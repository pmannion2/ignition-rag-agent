#!/usr/bin/env python3
import asyncio
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

# Import chunking functions from indexer
from indexer import create_chunks, enc
from indexer import mock_embedding as indexer_mock_embedding

# Import custom logger
from logger import LoggerMiddleware, get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("api")

# Global thread pool for CPU-bound operations
CPU_THREAD_POOL_SIZE = int(os.getenv("CPU_THREAD_POOL_SIZE", "10"))
thread_pool = ThreadPoolExecutor(max_workers=CPU_THREAD_POOL_SIZE)
logger.info(f"Initialized thread pool with {CPU_THREAD_POOL_SIZE} workers")

# Global semaphores for limiting concurrent OpenAI API calls
# Separate semaphores for querying and indexing to prevent resource contention
QUERY_CONCURRENCY_LIMIT = int(os.getenv("QUERY_CONCURRENCY_LIMIT", "10"))
INDEX_CONCURRENCY_LIMIT = int(os.getenv("INDEX_CONCURRENCY_LIMIT", "5"))
query_semaphore = asyncio.Semaphore(QUERY_CONCURRENCY_LIMIT)
index_semaphore = asyncio.Semaphore(INDEX_CONCURRENCY_LIMIT)
logger.info(f"Initialized query semaphore with {QUERY_CONCURRENCY_LIMIT} max concurrent operations")
logger.info(f"Initialized index semaphore with {INDEX_CONCURRENCY_LIMIT} max concurrent operations")

# Operation timeouts
DEFAULT_QUERY_TIMEOUT = int(os.getenv("DEFAULT_QUERY_TIMEOUT", "30"))
DEFAULT_INDEX_TIMEOUT = int(os.getenv("DEFAULT_INDEX_TIMEOUT", "60"))


# Global token rate limiter
class OpenAIRateLimiter:
    """Global rate limiter for OpenAI API token usage."""

    def __init__(self, tokens_per_minute=150000):
        """Initialize rate limiter with tokens per minute limit."""
        self.tokens_per_minute = tokens_per_minute
        self.tokens_used = 0
        self.reset_time = time.time() + 60
        self.lock = asyncio.Lock()
        logger.info(f"Initialized OpenAI rate limiter with {tokens_per_minute} tokens per minute")

    async def add_tokens(self, token_count):
        """Track token usage and wait if necessary to avoid rate limits."""
        async with self.lock:
            current_time = time.time()

            # Reset if minute has passed
            if current_time > self.reset_time:
                logger.debug(
                    f"Rate limit window reset. Used {self.tokens_used} tokens in previous window"
                )
                self.tokens_used = 0
                self.reset_time = current_time + 60

            # Check if adding would exceed limit
            if self.tokens_used + token_count > self.tokens_per_minute:
                wait_time = self.reset_time - current_time
                if wait_time > 0:
                    logger.info(
                        f"Rate limit approaching. Waiting {wait_time:.2f}s before processing"
                    )
                    await asyncio.sleep(wait_time)
                # Reset counter after waiting
                self.tokens_used = 0
                self.reset_time = time.time() + 60

            # Add tokens to counter
            self.tokens_used += token_count
            logger.debug(f"Added {token_count} tokens. Total in current window: {self.tokens_used}")
            return self.tokens_used


# Create global token rate limiter
openai_rate_limiter = OpenAIRateLimiter(
    tokens_per_minute=int(os.getenv("OPENAI_TOKENS_PER_MINUTE", "150000"))
)

# Global ChromaDB client to be initialized during startup
chroma_client = None

# Check if we're in mock mode (for testing without OpenAI API key)
MOCK_EMBEDDINGS = os.getenv("MOCK_EMBEDDINGS", "false").lower() == "true"
if MOCK_EMBEDDINGS:
    logger.info("Using mock embeddings for testing")

# Initialize OpenAI client only if we're not in mock mode or we have a key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not MOCK_EMBEDDINGS or openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("Initialized OpenAI client")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        if not MOCK_EMBEDDINGS:
            logger.warning(
                "No valid OpenAI API key and mock mode is not enabled, some features may not work"
            )
else:
    openai_client = None
    logger.info("OpenAI client not initialized (using mock mode)")

# Initialize Chroma settings
PERSIST_DIRECTORY = "chroma_index"
COLLECTION_NAME = "ignition_project"

# Check if running in Docker with external Chroma
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
USE_PERSISTENT_CHROMA = os.getenv("USE_PERSISTENT_CHROMA", "false").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="Ignition RAG API",
    description="API for querying Ignition project files using semantic search",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add logging middleware
app.add_middleware(LoggerMiddleware)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize global resources on application startup."""
    global chroma_client
    logger.info("Initializing global resources on startup")
    # Initialize the shared Chroma client
    chroma_client = get_chroma_client()
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down application and cleaning up resources")
    # Shutdown the thread pool
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")
    # Add any other cleanup code here as needed


def mock_embedding(text: str) -> List[float]:
    """Create a deterministic mock embedding based on the text content hash."""
    # Generate a deterministic hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Use the hash to seed a random generator for deterministic embeddings
    seed = int(text_hash, 16) % (2**32 - 1)
    rng = np.random.RandomState(seed)

    # Generate a random vector of length 1536 (same as text-embedding-ada-002)
    embedding = rng.rand(1536).astype(np.float32)

    # Normalize to unit length for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# Helper function for exponential backoff retry
async def generate_embedding_with_backoff(
    text,
    skip_rate_limiting=False,
    max_retries=5,
    initial_backoff=1,
    is_indexing=False,
    timeout=None,
):
    """Generate embedding with exponential backoff for rate limit handling."""
    if MOCK_EMBEDDINGS:
        return indexer_mock_embedding(text)

    # Set default timeout based on operation type
    if timeout is None:
        timeout = DEFAULT_INDEX_TIMEOUT if is_indexing else DEFAULT_QUERY_TIMEOUT

    # Calculate token count for rate limiting
    token_count = len(enc.encode(text))

    # Use both rate limiter and semaphore for concurrent OpenAI API calls
    # Skip rate limiting if requested
    if not skip_rate_limiting:
        await openai_rate_limiter.add_tokens(token_count)
    else:
        logger.debug(f"Skipping rate limiting for {token_count} tokens")

    # Choose appropriate semaphore based on operation type
    semaphore = index_semaphore if is_indexing else query_semaphore

    try:
        # Use timeout for the entire embedding operation
        async with asyncio.timeout(timeout):
            async with semaphore:
                logger.debug(
                    f"{'Indexing' if is_indexing else 'Query'} embedding generation: acquired semaphore, {token_count} tokens"
                )
                retries = 0
                backoff_time = initial_backoff

                while retries <= max_retries:
                    try:
                        response = openai_client.embeddings.create(
                            input=text, model="text-embedding-ada-002"
                        )
                        return response.data[0].embedding
                    except Exception as e:
                        error_str = str(e).lower()

                        # Check if this is a rate limit error
                        if (
                            "rate limit" in error_str
                            or "too many requests" in error_str
                            or "429" in error_str
                        ):
                            retries += 1
                            if retries > max_retries:
                                logger.error(
                                    f"Max retries reached for rate limit. Final error: {e!s}"
                                )
                                raise

                            logger.info(
                                f"Rate limit hit. Backing off for {backoff_time:.1f} seconds (retry {retries}/{max_retries})"
                            )
                            await asyncio.sleep(backoff_time)

                            # Exponential backoff: double the wait time for next retry
                            backoff_time *= 2
                        else:
                            # Not a rate limit error, re-raise
                            logger.error(f"Error generating embedding: {e!s}")
                            raise

                # This should not be reached due to the raise in the loop
                raise Exception("Failed to generate embedding after maximum retries")
    except asyncio.TimeoutError:
        logger.error(f"Embedding generation timed out after {timeout}s")
        if MOCK_EMBEDDINGS:
            return indexer_mock_embedding(text)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Embedding generation timed out after {timeout}s",
        ) from None


# Add custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all unhandled exceptions."""
    error_id = f"error-{time.time()}"
    logger.error(f"Error {error_id}: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error_id": error_id,
            "message": str(exc),
        },
    )


# Initialize Chroma client
def get_chroma_client():
    """Initialize and return a Chroma client."""
    try:
        # For tests, use in-memory client if specified
        if os.getenv("USE_IN_MEMORY_CHROMA", "false").lower() == "true":
            logger.info("Using in-memory Chroma client for testing")
            return chromadb.Client(Settings(anonymized_telemetry=False))

        # Due to version compatibility issues between client and server,
        # we'll consistently use local persistence mode regardless of environment
        logger.info(f"Using local Chroma with persistence at {PERSIST_DIRECTORY}")
        client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False)
        )

        # Test connection works
        heartbeat = client.heartbeat()
        logger.info(f"Successfully connected to local Chroma database. Heartbeat: {heartbeat}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Chroma: {e}")
        logger.error(f"Check if local storage at {PERSIST_DIRECTORY} is accessible")
        # Create an in-memory client for fallback
        logger.warning("Using in-memory Chroma as fallback (no persistence!)")
        return chromadb.Client(Settings(anonymized_telemetry=False))


# Initialize Chroma collection
def get_collection():
    """Get or create the collection for storing Ignition project data."""
    global chroma_client
    try:
        # Use the globally initialized client
        client = chroma_client

        # If client is None (which shouldn't happen in normal operation),
        # initialize a new client as fallback
        if client is None:
            logger.warning("Global Chroma client is None, creating a new client as fallback")
            client = get_chroma_client()

        # Check if collection exists, create it if it doesn't
        try:
            collection = client.get_collection(COLLECTION_NAME)
            doc_count = collection.count()
            logger.info(f"Connected to collection '{COLLECTION_NAME}' with {doc_count} documents")
        except ValueError:
            logger.info(f"Creating new collection '{COLLECTION_NAME}'")
            collection = client.create_collection(COLLECTION_NAME)

        return collection
    except Exception as e:
        logger.error(f"Error connecting to collection: {e}")
        # For API to start even without Chroma, return None
        # Endpoints will need to check if collection is None
        return None


# Define query request model
class QueryRequest(BaseModel):
    """Request model for querying the vector database."""

    query: str = Field(..., description="Query text to search for")
    top_k: int = Field(3, description="Number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default={}, description="Optional metadata filters for the query"
    )


# Define chunk response model
class Chunk(BaseModel):
    content: str = Field(..., description="The JSON content of the chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the chunk source")
    similarity: float = Field(..., description="Similarity score (lower is better)")


# Define query response model
class QueryResponse(BaseModel):
    results: List[Chunk] = Field(..., description="List of matching chunks")
    total: int = Field(..., description="Total number of results found")
    mock_used: bool = Field(False, description="Whether mock embeddings were used")


# Health check dependency to verify OpenAI and Chroma are working
async def verify_dependencies():
    """Verify that all dependencies are working."""
    errors = []

    # Check OpenAI API key if not in mock mode
    if not MOCK_EMBEDDINGS and not openai_api_key:
        errors.append("OpenAI API key not configured and mock mode is not enabled")

    # Check Chroma connection
    try:
        collection = get_collection()
        if collection:
            _ = collection.count()
        else:
            errors.append("Could not connect to Chroma collection")
    except Exception as e:
        errors.append(f"Chroma connection error: {e!s}")

    if errors:
        logger.error(f"Dependency check failed: {', '.join(errors)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"errors": errors, "message": "Service dependencies unavailable"},
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Ignition RAG API",
        "description": "API for querying Ignition project files using semantic search",
        "version": "1.0.0",
        "mock_mode": MOCK_EMBEDDINGS,
        "endpoints": {
            "/query": "POST - Query the vector database",
            "/stats": "GET - Get statistics about the indexed data",
            "/agent/query": "POST - Agent-optimized query endpoint for Cursor integration",
            "/agent/chat": "POST - Get conversational LLM responses enhanced with RAG context",
            "/health": "GET - Check API health status",
        },
    }


@app.get("/health")
async def health_check(deps: None = Depends(verify_dependencies)):
    """Health check endpoint that verifies all dependencies are working."""
    collection = get_collection()
    doc_count = collection.count() if collection else 0

    return {
        "status": "healthy",
        "chroma": {"connected": collection is not None, "documents": doc_count},
        "openai": {"configured": bool(openai_api_key) or MOCK_EMBEDDINGS},
        "mock_mode": MOCK_EMBEDDINGS,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, deps: None = Depends(verify_dependencies)):
    """Query the vector database for relevant Ignition project files."""
    try:
        # Start timing
        start_time = time.time()

        # Get collection
        collection = get_collection()
        if collection is None or collection.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents indexed. Please index the project first.",
            )

        # Extract query parameters
        query_text = req.query
        top_k = req.top_k if req.top_k else 3
        filter_metadata = req.filter_metadata if req.filter_metadata else {}

        # Generate embedding for the query
        use_mock = MOCK_EMBEDDINGS

        # CPU-bound operation: tokenizing and calculating embeddings
        # Run in thread pool to avoid blocking the event loop
        if use_mock:
            embedding = await asyncio.get_event_loop().run_in_executor(
                thread_pool, mock_embedding, query_text
            )
        else:
            embedding = await generate_embedding_with_backoff(
                query_text, is_indexing=False, timeout=DEFAULT_QUERY_TIMEOUT
            )

        # Query the collection with the embedding
        logger.info(f"Querying for: '{query_text}' with top_k={top_k}")
        try:
            async with asyncio.timeout(DEFAULT_QUERY_TIMEOUT):
                if filter_metadata:
                    logger.info(f"Applying filter: {filter_metadata}")
                    response = collection.query(
                        query_embeddings=[embedding],
                        n_results=top_k,
                        where=filter_metadata,
                    )
                else:
                    response = collection.query(
                        query_embeddings=[embedding],
                        n_results=top_k,
                    )
        except asyncio.TimeoutError:
            logger.error(f"Query operation timed out after {DEFAULT_QUERY_TIMEOUT}s")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Query operation timed out after {DEFAULT_QUERY_TIMEOUT}s",
            ) from None

        # Process results
        results = []
        if response["distances"] and len(response["distances"][0]) > 0:
            documents = response["documents"][0]
            metadatas = response["metadatas"][0]
            distances = response["distances"][0]

            for _i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Calculate similarity score (invert distance)
                similarity = 1.0 - dist
                # Format result
                result = {
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "source": meta.get("filepath", "Unknown source"),
                }
                results.append(result)

        # Calculate query time
        query_time = time.time() - start_time
        logger.info(f"Query processed in {query_time:.2f} seconds")

        # Return formatted results
        return QueryResponse(
            results=results,
            total=len(results),
            mock_used=use_mock,
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error processing query: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


# Agent-optimized query for Cursor integration
class AgentQueryRequest(BaseModel):
    """Request model for the agent query endpoint."""

    query: str = Field(..., description="Query string")
    top_k: int = Field(5, description="Number of results to return")
    filter_type: Optional[str] = Field(None, description="Optional filter for document type")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context for the query")
    use_mock: Optional[bool] = Field(
        None, description="Whether to use mock embeddings (for testing)"
    )
    artificial_delay: Optional[float] = Field(
        0, description="Artificial delay in seconds for testing concurrency"
    )


class AgentQueryResponse(BaseModel):
    context_chunks: List[Dict[str, Any]] = Field(
        ..., description="Relevant context chunks for the agent"
    )
    suggested_prompt: Optional[str] = Field(
        None, description="Suggested prompt incorporating the context"
    )
    mock_used: bool = Field(False, description="Whether mock embeddings were used")


# Define chat request and response models for conversational responses
class ChatRequest(BaseModel):
    query: str = Field(..., description="The natural language query to search for")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    filter_type: Optional[str] = Field(
        None, description="Filter results by document type (perspective or tag)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for the agent (e.g., current file being edited)",
    )
    use_mock: Optional[bool] = Field(None, description="Override mock mode setting (for testing)")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Conversational LLM response incorporating context")
    context_chunks: List[Dict[str, Any]] = Field(
        ..., description="Relevant context chunks used for the response"
    )
    mock_used: bool = Field(False, description="Whether mock embeddings were used")


@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(req: AgentQueryRequest, deps: None = Depends(verify_dependencies)):
    """Agent-optimized query endpoint for Cursor integration."""
    try:
        # Start timing
        start_time = time.time()

        # If artificial delay is specified, simulate a slow request
        artificial_delay = req.artificial_delay or 0
        if artificial_delay > 0:
            logger.info(f"Adding artificial delay of {artificial_delay}s for request")
            await asyncio.sleep(artificial_delay)

        # Get collection
        collection = get_collection()
        if collection is None or collection.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents indexed. Please index the project first.",
            )

        # Extract query parameters
        query_text = req.query
        top_k = req.top_k if req.top_k else 5
        filter_type = req.filter_type
        context = req.context

        # Generate embedding for the query
        use_mock = MOCK_EMBEDDINGS
        if req.use_mock is not None:
            use_mock = req.use_mock

        # Use thread pool for CPU-bound embedding generation
        if use_mock:
            embedding = await asyncio.get_event_loop().run_in_executor(
                thread_pool, mock_embedding, query_text
            )
        else:
            embedding = await generate_embedding_with_backoff(
                query_text, is_indexing=False, timeout=DEFAULT_QUERY_TIMEOUT
            )

        # Construct filter metadata based on filter_type
        filter_metadata = {}
        if filter_type:
            filter_metadata["type"] = filter_type

        # Apply additional filters based on context
        if context and context.get("current_file"):
            # If we have a current file context, we could boost files in the same directory
            # or related components (this is a simple example, can be expanded)
            logger.info(f"Query has current file context: {context['current_file']}")

        # Query the collection with appropriate timeout
        try:
            async with asyncio.timeout(DEFAULT_QUERY_TIMEOUT):
                if filter_metadata:
                    logger.info(f"Applying filter: {filter_metadata}")
                    response = collection.query(
                        query_embeddings=[embedding],
                        n_results=top_k,
                        where=filter_metadata,
                    )
                else:
                    response = collection.query(
                        query_embeddings=[embedding],
                        n_results=top_k,
                    )
        except asyncio.TimeoutError:
            logger.error(f"Agent query operation timed out after {DEFAULT_QUERY_TIMEOUT}s")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Query operation timed out after {DEFAULT_QUERY_TIMEOUT}s",
            ) from None

        # Process results
        context_chunks = []
        if response["distances"] and len(response["distances"][0]) > 0:
            documents = response["documents"][0]
            metadatas = response["metadatas"][0]
            distances = response["distances"][0]

            for _i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Calculate similarity score (invert distance)
                similarity = 1.0 - dist

                # Format source path to be more readable
                source_path = meta.get("filepath", "Unknown")
                if source_path != "Unknown":
                    source_path = os.path.basename(source_path)

                # Format chunk
                chunk = {
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "source": source_path,
                }
                context_chunks.append(chunk)

        # Generate suggested prompt enhancement with retrieved context
        # This is a CPU-bound operation, use thread pool
        def generate_suggested_prompt():
            prompt = f"Query: {query_text}\n\n"
            if context_chunks:
                prompt += "Relevant context from Ignition project:\n\n"
                for _i, chunk in enumerate(context_chunks, 1):
                    source = chunk["source"]
                    content_preview = chunk["content"]
                    if len(content_preview) > 200:
                        content_preview = content_preview[:197] + "..."
                    prompt += f"{_i}. From {source}: {content_preview}\n\n"
            return prompt

        suggested_prompt = await asyncio.get_event_loop().run_in_executor(
            thread_pool, generate_suggested_prompt
        )

        # Measure response time
        response_time = time.time() - start_time
        logger.info(
            f"Agent query completed in {response_time:.2f}s, found {len(context_chunks)} chunks, mock_mode={use_mock}"
        )

        # If there was an artificial delay, log it
        if artificial_delay > 0:
            logger.info(f"Query included artificial delay of {artificial_delay:.2f}s")

        return AgentQueryResponse(
            context_chunks=context_chunks,
            suggested_prompt=suggested_prompt,
            mock_used=use_mock,
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error processing agent query: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest, deps: None = Depends(verify_dependencies)):
    """Generate a conversational LLM response enhanced with RAG context."""
    start_time = time.time()
    logger.info(f"Chat query received: '{req.query}', top_k={req.top_k}")

    # Check if mock mode is requested for this query
    use_mock = MOCK_EMBEDDINGS
    if req.use_mock is not None:
        use_mock = req.use_mock
        logger.info(f"Mock mode overridden to: {use_mock}")

    try:
        # First, get relevant context chunks using the agent_query logic
        agent_req = AgentQueryRequest(
            query=req.query,
            top_k=req.top_k,
            filter_type=req.filter_type,
            context=req.context,
            use_mock=use_mock,
        )

        # Get the raw query results (re-using the agent_query logic)
        query_results = await agent_query(agent_req)

        # If we're using mock mode and no OpenAI API key, return a simple response
        if use_mock and not openai_client:
            mock_response = f"Here's what I found about '{req.query}':\n\n"
            for _i, chunk in enumerate(query_results.context_chunks, 1):
                mock_response += f"{_i}. From {chunk.get('source', 'unknown source')}: "
                content = chunk.get("content", "").strip()
                mock_response += f"{content[:100]}{'...' if len(content) > 100 else ''}\n\n"

            mock_response += "This is a mock response since no OpenAI API key is available."

            return ChatResponse(
                response=mock_response,
                context_chunks=query_results.context_chunks,
                mock_used=True,
            )

        # Prepare a prompt for the LLM that includes the relevant context
        context_prompt = "You are an assistant for Ignition SCADA systems. "
        context_prompt += "Answer the following question using the provided context. "
        context_prompt += "If the context doesn't provide enough information to answer fully, "
        context_prompt += (
            "acknowledge that and provide the best response based on what's available.\n\n"
        )

        # Add the context chunks
        if query_results.context_chunks:
            context_prompt += "Context information:\n"
            for _i, chunk in enumerate(query_results.context_chunks, 1):
                source = chunk.get("source", "Unknown source")
                content = chunk.get("content", "").strip()
                context_prompt += f"\n--- Context {_i}: {source} ---\n{content}\n"
        else:
            context_prompt += "No specific context found for this query.\n"

        # Add the user's question
        context_prompt += f"\nQuestion: {req.query}\n\nAnswer:"

        # Call the OpenAI API to generate a conversational response
        try:
            chat_response = openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Using GPT-4 for a larger context window
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in Ignition SCADA systems.",
                    },
                    {"role": "user", "content": context_prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # Extract the response text
            response_text = chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating chat response: {e}", exc_info=True)
            response_text = f"I encountered an error when trying to generate a response: {e}"
            if not use_mock:
                response_text += "\n\nHere's the raw context I found:\n"
                for _i, chunk in enumerate(query_results.context_chunks, 1):
                    source = chunk.get("source", "Unknown")
                    content = chunk.get("content", "")[:100]
                    response_text += f"{_i}. From {source}: {content}...\n"

        response_time = time.time() - start_time
        logger.info(f"Chat response generated in {response_time:.2f}s")

        return ChatResponse(
            response=response_text,
            context_chunks=query_results.context_chunks,
            mock_used=use_mock,
        )

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error processing chat query: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed data."""
    start_time = time.time()
    logger.info("Stats request received")

    try:
        # Get the collection
        collection = get_collection()
        if collection is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database is not available. Please check Chroma connection.",
            )

        # Get count of documents in the collection
        try:
            count = collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error counting documents: {e!s}",
            ) from e

        # Get some stats about document types if any documents exist
        type_stats = {}
        if count > 0:
            try:
                # Get all document types without limiting the results
                # First get metadata only for efficiency
                all_metadata = collection.get(include=["metadatas"])["metadatas"]

                # Count documents by type
                for meta in all_metadata:
                    doc_type = meta.get("type", "unknown")
                    if doc_type not in type_stats:
                        type_stats[doc_type] = 0
                    type_stats[doc_type] += 1
            except Exception as e:
                logger.error(f"Error getting document types: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting document types: {e!s}",
                ) from e

        response_time = time.time() - start_time
        logger.info(f"Stats request completed in {response_time:.2f}s")

        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "type_distribution": type_stats,
            "mock_mode": MOCK_EMBEDDINGS,
        }

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error getting stats: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


# Define index request model
class IndexRequest(BaseModel):
    """Request model for indexing the Ignition project files."""

    project_path: str = Field(
        "./whk-ignition-scada", description="Path to the Ignition project directory"
    )
    rebuild: bool = Field(False, description="Whether to rebuild the index from scratch")
    skip_rate_limiting: bool = Field(
        False, description="Skip rate limiting for faster processing (use with caution)"
    )


# Create a semaphore for limiting concurrent indexing operations
indexing_lock = asyncio.Lock()  # Only allow one indexing operation at a time


@app.post("/index", summary="Index Ignition project files")
async def index_project(req: IndexRequest, deps: None = Depends(verify_dependencies)):
    """Index Ignition project files."""
    logger.info(f"Index request received for {req.project_path}")

    try:
        # Use timeout to avoid waiting too long if another indexing operation is in progress
        try:
            async with asyncio.timeout(5.0):
                # Try to acquire the lock - only one indexing operation at a time
                async with indexing_lock:
                    logger.debug("Acquired indexing lock")
                    return await _perform_indexing(req)
        except asyncio.TimeoutError:
            logger.warning("Indexing already in progress, cannot acquire lock")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Another indexing operation is in progress. Please try again later.",
            ) from None
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error during indexing: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


async def _perform_indexing(req: IndexRequest):
    """Internal function to perform the actual indexing operation."""
    global MOCK_EMBEDDINGS

    # Set startup time
    start_time = time.time()

    # Verify the path exists
    project_path = os.path.expanduser(req.project_path)
    if not os.path.exists(project_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path {project_path} does not exist.",
        )

    # Get collection
    collection = get_collection()
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to vector database",
        )

    # Get parameters
    rebuild = req.rebuild
    skip_rate_limiting = req.skip_rate_limiting

    logger.info(
        f"Starting indexing of project at {project_path} (rebuild={rebuild}, skip_rate_limiting={skip_rate_limiting})"
    )

    # If rebuilding, delete existing collection and create a new one
    if rebuild and collection.count() > 0:
        logger.info(
            f"Rebuilding index - deleting existing collection with {collection.count()} documents"
        )
        client = get_chroma_client()
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME)
        logger.info("Created new empty collection")

    # Path to the Ignition project
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project directory not found: {project_path}",
        )

    # Find all JSON files in the project directory
    json_files = []
    for root, _, files in os.walk(project_path):
        for fname in files:
            if fname.endswith(".json"):
                json_files.append(os.path.join(root, fname))

    logger.info(f"Found {len(json_files)} JSON files to index")

    # Process each file
    total_files = len(json_files)
    doc_count = 0
    chunk_count = 0

    # Track token usage for rate limiting
    hard_token_limit = 7500  # Hard limit for safety

    # Helper function for character chunking (copied from main.py)
    def chunk_by_characters(text, max_chunk_size):
        """Chunk a text by a fixed number of characters, respecting JSON structure when possible."""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        current_pos = 0
        text_length = len(text)

        while current_pos < text_length:
            # Default end position
            end_pos = min(current_pos + max_chunk_size, text_length)

            # If we're not at the end of the text, try to find a better split point
            if end_pos < text_length:
                # Look for JSON structural elements to split on
                candidates = []

                # Find commas between array/object items
                comma_pos = text.rfind(",", current_pos, end_pos)
                if comma_pos > current_pos:
                    candidates.append((comma_pos + 1, "comma"))

                # Find closing braces/brackets followed by comma
                brace_pos = text.rfind("},", current_pos, end_pos)
                if brace_pos > current_pos:
                    candidates.append((brace_pos + 1, "brace"))

                bracket_pos = text.rfind("],", current_pos, end_pos)
                if bracket_pos > current_pos:
                    candidates.append((bracket_pos + 1, "bracket"))

                # If we found candidates, use the latest one
                if candidates:
                    candidates.sort(reverse=True)
                    end_pos = candidates[0][0]

            # Extract the chunk
            chunk = text[current_pos:end_pos]
            chunks.append(chunk)
            current_pos = end_pos

        return chunks

    for file_index, file_path in enumerate(json_files):
        file_start_time = time.time()
        try:
            logger.info(f"Processing {file_path}... [{file_index + 1}/{total_files}]")
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                logger.info(f"Skipping empty file: {file_path}")
                continue

            # Create metadata for this file
            metadata = {
                "filepath": file_path,
                "filename": os.path.basename(file_path),
                "directory": os.path.dirname(file_path),
                "type": "json",
                "source": file_path,
            }

            # Check token count
            token_count = len(enc.encode(content))
            logger.info(f"File has {token_count} tokens")

            # Always chunk files over 3000 tokens to ensure safer processing
            if token_count <= 3000:
                try:
                    # Generate embedding with backoff for rate limiting
                    embedding = await generate_embedding_with_backoff(
                        content,
                        skip_rate_limiting=skip_rate_limiting,
                        is_indexing=True,
                        timeout=DEFAULT_INDEX_TIMEOUT,
                    )

                    # Add to collection
                    file_path_replaced = file_path.replace("/", "_").replace("\\", "_")
                    collection.add(
                        ids=[file_path_replaced],
                        documents=[content],
                        embeddings=[embedding],
                        metadatas=[metadata],
                    )
                    doc_count += 1
                    chunk_count += 1
                    logger.info(f"Indexed {file_path} as a single chunk")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e!s}")
            else:
                # For large files, we need to chunk the content
                logger.info(f"File exceeds token limit, chunking: {file_path}")

                try:
                    # For extremely large files (>50K tokens), use a character-level chunking approach
                    if token_count > 50000:
                        logger.info(
                            f"Very large file ({token_count} tokens), using character-level chunking"
                        )
                        chunks = []

                        # For json files, try to intelligently split on braces or brackets
                        if file_path.endswith(".json"):
                            try:
                                # Try to parse as JSON first to extract key structures
                                json_content = json.loads(content)
                                logger.info(
                                    f"Successfully parsed JSON for {file_path} - type: {type(json_content).__name__}"
                                )

                                # For array-type JSONs, split at the top level
                                if isinstance(json_content, list) and len(json_content) > 1:
                                    logger.info(
                                        f"Using array-level chunking for JSON array with {len(json_content)} items"
                                    )
                                    sub_chunks = []
                                    current_array = []
                                    current_tokens = 0

                                    # Process each array element
                                    for item in json_content:
                                        item_str = json.dumps(item)
                                        item_tokens = len(enc.encode(item_str))

                                        # If this item alone exceeds the limit, we need to chunk it further
                                        if item_tokens > hard_token_limit:
                                            # Process any accumulated items
                                            if current_array:
                                                array_str = json.dumps(current_array)
                                                sub_chunks.append(array_str)
                                                current_array = []
                                                current_tokens = 0

                                            # Chunk this large item by characters, preserving JSON format
                                            item_chunks = chunk_by_characters(
                                                item_str,
                                                int(hard_token_limit / 1.2),
                                            )
                                            sub_chunks.extend(item_chunks)
                                        # If adding this would exceed limit, create a new chunk
                                        elif current_tokens + item_tokens > hard_token_limit:
                                            array_str = json.dumps(current_array)
                                            sub_chunks.append(array_str)
                                            current_array = [item]
                                            current_tokens = item_tokens
                                        # Otherwise add to current chunk
                                        else:
                                            current_array.append(item)
                                            current_tokens += item_tokens

                                    # Add any remaining items
                                    if current_array:
                                        array_str = json.dumps(current_array)
                                        sub_chunks.append(array_str)

                                chunks = [(chunk, metadata) for chunk in sub_chunks]
                            except json.JSONDecodeError:
                                # If JSON parsing fails, use character-level chunking
                                text_chunks = chunk_by_characters(
                                    content,
                                    int(hard_token_limit / 1.2),
                                )
                                chunks = [(chunk, metadata) for chunk in text_chunks]
                        else:
                            # For non-JSON files, use character-level chunking
                            text_chunks = chunk_by_characters(
                                content,
                                int(hard_token_limit / 1.2),
                            )
                            chunks = [(chunk, metadata) for chunk in text_chunks]
                    else:
                        # For moderately sized files, use create_chunks from indexer module
                        # Preprocess the document to match expected input format
                        doc = {"content": content, "metadata": metadata}
                        chunks = create_chunks([doc])

                    # Process chunks
                    for _i, (chunk_text, chunk_metadata) in enumerate(chunks):
                        try:
                            # Generate embedding with backoff
                            embedding = await generate_embedding_with_backoff(
                                chunk_text,
                                skip_rate_limiting=skip_rate_limiting,
                                is_indexing=True,
                                timeout=DEFAULT_INDEX_TIMEOUT,
                            )

                            # Create a unique ID for this chunk
                            file_path_replaced = file_path.replace("/", "_").replace("\\", "_")
                            chunk_id = f"{file_path_replaced}_chunk_{_i}"

                            # Add to collection
                            collection.add(
                                ids=[chunk_id],
                                documents=[chunk_text],
                                embeddings=[embedding],
                                metadatas=[chunk_metadata],
                            )

                            chunk_count += 1
                        except Exception as e:
                            logger.error(f"Error processing chunk {_i} of {file_path}: {e!s}")

                    doc_count += 1
                    logger.info(f"Indexed {file_path} into {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error chunking {file_path}: {e!s}")

            # Calculate time taken for this file
            file_time = time.time() - file_start_time
            logger.info(f"Processed {file_path} in {file_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e!s}")

    # Calculate total time
    total_time = time.time() - start_time
    avg_time = total_time / total_files if total_files > 0 else 0

    # Summary
    result = {
        "indexed_files": doc_count,
        "total_chunks": chunk_count,
        "total_files_found": total_files,
        "total_time_seconds": total_time,
        "average_time_per_file": avg_time,
        "project_path": str(project_path),
    }

    logger.info(f"Indexing completed in {total_time:.2f} seconds")
    logger.info(f"Indexed {doc_count}/{total_files} files into {chunk_count} chunks")

    return result


class DelayTestRequest(BaseModel):
    """Request model for the delay test endpoint."""

    delay_seconds: float = Field(1.0, description="Seconds to delay before responding")
    identifier: str = Field("test", description="Identifier for this request")


class DelayTestResponse(BaseModel):
    """Response model for the delay test endpoint."""

    identifier: str = Field(..., description="Echo of the request identifier")
    delay_seconds: float = Field(..., description="Actual delay applied")
    timestamp: float = Field(..., description="Server timestamp")


@app.post("/test/delay", response_model=DelayTestResponse)
async def test_delay(req: DelayTestRequest):
    """Test endpoint that delays for the specified time before responding.

    Used to test concurrency handling in the API.
    """
    start_time = time.time()

    # Log the request
    logger.info(f"Delay test request received: {req.identifier}, delay={req.delay_seconds}s")

    # Apply the delay
    await asyncio.sleep(req.delay_seconds)

    # Calculate actual delay
    actual_delay = time.time() - start_time

    # Log completion
    logger.info(f"Delay test completed: {req.identifier}, actual_delay={actual_delay:.2f}s")

    # Return response
    return DelayTestResponse(
        identifier=req.identifier, delay_seconds=actual_delay, timestamp=time.time()
    )


@app.post("/reset", summary="Reset the Chroma collection")
async def reset_collection():
    """Reset the entire Chroma collection, removing all indexed documents.
    This is a destructive operation and cannot be undone."""
    logger.warning("Received request to reset Chroma collection")

    try:
        # Get the Chroma client
        client = get_chroma_client()

        # Delete the collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Successfully deleted collection '{COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
            # Continue anyway - it might not exist

        # Create a new empty collection
        collection = client.create_collection(COLLECTION_NAME)
        logger.info(f"Created new empty collection '{COLLECTION_NAME}'")

        return {
            "status": "success",
            "message": f"Collection '{COLLECTION_NAME}' has been reset",
            "document_count": collection.count(),
        }

    except Exception as e:
        error_msg = f"Error resetting collection: {e!s}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        ) from e


if __name__ == "__main__":
    import uvicorn

    # Set debug mode from environment
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.debug = debug

    # Get port from environment
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"Starting API server on port {port}, debug={debug}")
    uvicorn.run(app, host="0.0.0.0", port=port)
