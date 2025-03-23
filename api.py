#!/usr/bin/env python3
import os
import time
import hashlib
import numpy as np
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import traceback

# Import custom logger
from logger import get_logger, LoggerMiddleware

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
    logger.error(f"Unhandled exception: {error_id} - {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred",
            "error_id": error_id,
            "detail": str(exc) if app.debug else "Internal Server Error",
        },
    )


# Initialize Chroma client - either local or remote
try:
    if CHROMA_HOST and CHROMA_PORT:
        logger.info(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
    else:
        logger.info(f"Using local Chroma with persistence at {PERSIST_DIRECTORY}")
        # Updated client initialization for newer ChromaDB versions
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # Get collection or create if it doesn't exist
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info(
        f"Connected to collection '{COLLECTION_NAME}' with {collection.count()} documents"
    )
except Exception as e:
    logger.error(f"Error connecting to Chroma: {e}", exc_info=True)
    logger.info("Creating a new collection. Please run indexer.py to populate it.")
    try:
        # Try to create the collection if it doesn't exist
        if CHROMA_HOST and CHROMA_PORT:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        else:
            # Updated client initialization for newer ChromaDB versions
            chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
    except Exception as create_error:
        logger.critical(
            f"Failed to create Chroma collection: {create_error}", exc_info=True
        )
        # We'll continue execution, but API calls that use the collection will fail


# Define query request model
class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to search for")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    filter_type: Optional[str] = Field(
        None, description="Filter results by document type (perspective or tag)"
    )
    filter_path: Optional[str] = Field(
        None, description="Filter results by file path pattern"
    )
    use_mock: Optional[bool] = Field(
        None,
        description="Use mock embeddings for testing (overrides environment variable)",
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
        count = collection.count()
    except Exception as e:
        errors.append(f"Chroma connection error: {str(e)}")

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
    return {
        "status": "healthy",
        "chroma": {"connected": True, "documents": collection.count()},
        "openai": {"configured": bool(openai_api_key) or MOCK_EMBEDDINGS},
        "mock_mode": MOCK_EMBEDDINGS,
    }


@app.post("/query", response_model=QueryResponse)
async def query_vector_store(
    req: QueryRequest, deps: None = Depends(verify_dependencies)
):
    """Query the vector database for similar content."""
    start_time = time.time()
    logger.info(
        f"Query request received: '{req.query}', top_k={req.top_k}, filters={req.filter_type}/{req.filter_path}"
    )

    # Check if mock mode is requested for this query
    use_mock = MOCK_EMBEDDINGS
    if req.use_mock is not None:
        use_mock = req.use_mock
        logger.info(f"Mock mode overridden to: {use_mock}")

    try:
        # Prepare filter if any
        where_filter = {}
        if req.filter_type:
            where_filter["type"] = req.filter_type
        if req.filter_path:
            where_filter["filepath"] = {"$contains": req.filter_path}

        # Use where_filter only if it has any conditions
        where_document = where_filter if where_filter else None

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
                    detail=f"Error generating embedding: {str(e)}",
                )

        # Perform similarity search in Chroma
        try:
            query_start = time.time()
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=req.top_k,
                where=where_document,
            )
            logger.debug(f"Chroma query completed in {time.time() - query_start:.2f}s")
        except Exception as e:
            logger.error(f"Error querying vector database: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error querying vector database: {str(e)}",
            )

        chunks = []
        # Check if we got any results
        if results and "documents" in results and results["documents"]:
            # results["documents"][0] is list of document texts, [0] because we passed one query
            for doc_text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(
                    Chunk(content=doc_text, metadata=meta, similarity=distance)
                )

        response_time = time.time() - start_time
        logger.info(
            f"Query completed in {response_time:.2f}s, found {len(chunks)} results, mock_mode={use_mock}"
        )
        return QueryResponse(results=chunks, total=len(chunks), mock_used=use_mock)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve status code
        raise
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


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
                    detail=f"Error generating embedding: {str(e)}",
                )

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
                detail=f"Error querying vector database: {str(e)}",
            )

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
                elif chunk_type == "tag":
                    if "folder" in meta:
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
        error_msg = f"Error processing agent query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


@app.get("/stats")
async def get_stats():
    """Get statistics about the indexed data."""
    start_time = time.time()
    logger.info("Stats request received")

    try:
        # Get count of documents in the collection
        try:
            count = collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error counting documents: {str(e)}",
            )

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
                    detail=f"Error getting sample documents: {str(e)}",
                )

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
        error_msg = f"Error getting stats: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


if __name__ == "__main__":
    import uvicorn

    # Set debug mode from environment
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.debug = debug

    # Get port from environment
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"Starting API server on port {port}, debug={debug}")
    uvicorn.run(app, host="0.0.0.0", port=port)
