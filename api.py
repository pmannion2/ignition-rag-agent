#!/usr/bin/env python3
import hashlib
import os
import time
import traceback
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

# Import custom logger
from logger import LoggerMiddleware, get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("api")

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

        # Check if external Chroma server is specified
        if CHROMA_HOST:
            logger.info(f"Connecting to external Chroma at {CHROMA_HOST}:{CHROMA_PORT}")

            # Use different client init based on whether we're using persistent mode
            if USE_PERSISTENT_CHROMA:
                logger.info("Using persistent HTTP client mode")
                client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=int(CHROMA_PORT) if CHROMA_PORT else 8000,
                    tenant="default_tenant",
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
            else:
                # Standard HTTP client
                client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=int(CHROMA_PORT) if CHROMA_PORT else 8000,
                    tenant="default_tenant",
                    settings=Settings(
                        anonymized_telemetry=False,
                    ),
                )
        else:
            # Use local persistent Chroma
            logger.info(f"Using local Chroma with persistence at {PERSIST_DIRECTORY}")
            client = chromadb.PersistentClient(
                path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False)
            )

        # Test connection works
        heartbeat = client.heartbeat()
        logger.info(f"Chroma connection successful. Heartbeat: {heartbeat}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Chroma: {e}")
        logger.error(
            f"Check if Chroma is running at {CHROMA_HOST or 'localhost'}:{CHROMA_PORT or '8000'}"
        )
        # Create an in-memory client for fallback
        logger.warning("Using in-memory Chroma as fallback (no persistence!)")
        return chromadb.Client(Settings(anonymized_telemetry=False))


# Initialize Chroma collection
def get_collection():
    """Get or create the collection for storing Ignition project data."""
    try:
        client = get_chroma_client()
        # Check if collection exists, create it if it doesn't
        try:
            collection = client.get_collection(COLLECTION_NAME)
            doc_count = collection.count()
            logger.info(
                f"Connected to collection '{COLLECTION_NAME}' with {doc_count} documents"
            )
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


@app.post("/query", summary="Query the vector database")
async def query(request: QueryRequest):
    """
    Search for relevant context from Ignition project files.

    This endpoint performs semantic search using the query text.
    """
    try:
        # Get collection (may be None if Chroma connection fails)
        collection = get_collection()
        if collection is None:
            logger.error("Failed to connect to the vector database")
            raise HTTPException(
                status_code=503,
                detail="Vector database is not available. Please check Chroma connection.",
            )

        # Check if collection is empty
        if collection.count() == 0:
            logger.warning("Empty collection, no results will be returned")
            return {
                "results": [],
                "metadata": {
                    "total_chunks": 0,
                    "query": request.query,
                    "message": "The collection is empty. Please run the indexer to populate it.",
                },
            }

        # Generate embedding for the query
        query_embedding = None
        if MOCK_EMBEDDINGS or not openai_client:
            logger.info("Using mock embedding for query")
            query_embedding = mock_embedding(request.query)
        else:
            try:
                # Use OpenAI API to generate embedding
                embedding_response = openai_client.embeddings.create(
                    input=request.query, model="text-embedding-ada-002"
                )
                query_embedding = embedding_response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback to mock embedding
                logger.info("Falling back to mock embedding")
                query_embedding = mock_embedding(request.query)

        # Query the collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=request.top_k,
            where=request.filter_metadata or None,
            include=["metadatas", "documents", "distances"],
        )

        # Process and format results
        processed_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            document = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results["distances"] else 0

            # Convert distance to similarity score (cosine similarity)
            similarity = 1.0 - distance if distance <= 2.0 else 0

            # Add result
            processed_results.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "similarity": similarity,
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
            )

        return {
            "results": processed_results,
            "metadata": {
                "total_chunks": collection.count(),
                "query": request.query,
                "embedding_type": (
                    "mock" if MOCK_EMBEDDINGS or not openai_client else "openai"
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


# Agent-optimized query for Cursor integration
class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to search for")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    filter_type: Optional[str] = Field(
        None, description="Filter results by document type (perspective or tag)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for the agent (e.g., current file being edited)",
    )
    use_mock: Optional[bool] = Field(
        None,
        description="Use mock embeddings for testing (overrides environment variable)",
    )


class AgentQueryResponse(BaseModel):
    context_chunks: List[Dict[str, Any]] = Field(
        ..., description="Relevant context chunks for the agent"
    )
    suggested_prompt: Optional[str] = Field(
        None, description="Suggested prompt incorporating the context"
    )
    mock_used: bool = Field(False, description="Whether mock embeddings were used")


@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(
    req: AgentQueryRequest, deps: None = Depends(verify_dependencies)
):
    """Agent-optimized query endpoint for Cursor integration."""
    start_time = time.time()
    logger.info(
        f"Agent query received: '{req.query}', top_k={req.top_k}, context={req.context}"
    )

    # Check if mock mode is requested for this query
    use_mock = MOCK_EMBEDDINGS
    if req.use_mock is not None:
        use_mock = req.use_mock
        logger.info(f"Mock mode overridden to: {use_mock}")

    try:
        # Get the collection
        collection = get_collection()
        if collection is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database is not available. Please check Chroma connection.",
            )

        # Extract any filter from context
        where_filter = {}
        if req.filter_type:
            where_filter["type"] = req.filter_type

        # If context includes a file path, try to find related files
        current_file = req.context.get("current_file") if req.context else None
        if current_file:
            # This is a simple heuristic - in a real implementation you might
            # use more sophisticated matching based on file structure
            file_base = os.path.basename(current_file)
            if "." in file_base:
                file_base = file_base.split(".")[0]
                # Using simple equality instead of $contains
                where_filter["filepath"] = file_base

        # Get embedding for the query
        try:
            embedding_start = time.time()

            if use_mock:
                # Use mock embedding if in mock mode
                query_vector = mock_embedding(req.query)
                logger.debug("Using mock embedding for query")
            else:
                # Use OpenAI API for real embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002", input=[req.query]
                )
                query_vector = response.data[0].embedding

            logger.debug(f"Generated embedding in {time.time() - embedding_start:.2f}s")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            if not use_mock:
                logger.info("Falling back to mock embedding")
                query_vector = mock_embedding(req.query)
                use_mock = True
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Error generating embedding: {e!s}",
                ) from e

        # Perform similarity search in Chroma
        try:
            query_start = time.time()
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=req.top_k,
                where=where_filter if where_filter else None,
            )
            logger.debug(f"Chroma query completed in {time.time() - query_start:.2f}s")
        except Exception as e:
            logger.error(f"Error querying vector database: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error querying vector database: {e!s}",
            ) from e

        # Format results for agent consumption
        context_chunks = []
        if results and "documents" in results and results["documents"]:
            format_start = time.time()
            for doc_text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Format the chunk with its metadata for easy agent use
                chunk_type = meta.get("type", "unknown")
                source_info = f"{os.path.basename(meta.get('filepath', 'unknown'))}"

                if chunk_type == "perspective":
                    if "component" in meta:
                        source_info += f" - Component: {meta['component']}"
                elif chunk_type == "tag" and "folder" in meta:
                    source_info += f" - Folder: {meta['folder']}"

                context_chunks.append(
                    {
                        "source": source_info,
                        "content": doc_text,
                        "metadata": meta,
                        "similarity": distance,
                    }
                )
            logger.debug(f"Formatted results in {time.time() - format_start:.2f}s")

        # Create a suggested prompt that incorporates the context
        suggested_prompt = None
        if context_chunks:
            prompt_start = time.time()
            suggested_prompt = f"Query: {req.query}\n\nRelevant Ignition context:\n"
            for i, chunk in enumerate(context_chunks, 1):
                suggested_prompt += f"\n--- Context {i} ({chunk['source']}) ---\n"
                suggested_prompt += f"{chunk['content']}\n"

            suggested_prompt += (
                "\nBased on the above context, please help with the query."
            )
            logger.debug(f"Generated prompt in {time.time() - prompt_start:.2f}s")

        response_time = time.time() - start_time
        logger.info(
            f"Agent query completed in {response_time:.2f}s, found {len(context_chunks)} chunks, mock_mode={use_mock}"
        )

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
            # Limit query to avoid huge responses
            try:
                sample = collection.get(limit=1000)
            except Exception as e:
                logger.error(f"Error getting sample documents: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error getting sample documents: {e!s}",
                ) from e

            # Count documents by type
            for meta in sample["metadatas"]:
                doc_type = meta.get("type", "unknown")
                if doc_type not in type_stats:
                    type_stats[doc_type] = 0
                type_stats[doc_type] += 1

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


if __name__ == "__main__":
    import uvicorn

    # Set debug mode from environment
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.debug = debug

    # Get port from environment
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"Starting API server on port {port}, debug={debug}")
    uvicorn.run(app, host="0.0.0.0", port=port)
