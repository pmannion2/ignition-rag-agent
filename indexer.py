#!/usr/bin/env python3
import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import tiktoken
import typer
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Check if we're in mock mode (for testing without OpenAI API key)
MOCK_EMBEDDINGS = os.getenv("MOCK_EMBEDDINGS", "false").lower() == "true"
if MOCK_EMBEDDINGS:
    print("Using mock embeddings for testing")

# Initialize OpenAI client only if we're not in mock mode or we have a key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not MOCK_EMBEDDINGS or openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        print("Initialized OpenAI client")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        if not MOCK_EMBEDDINGS:
            print(
                "No valid OpenAI API key and mock mode is not enabled, some features may not work"
            )
else:
    client = None
    print("OpenAI client not initialized (using mock mode)")

# Initialize tokenizer for GPT models
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 6000  # Increased from 400 to reduce API calls while staying under limits

# Initialize Chroma client
PERSIST_DIRECTORY = "chroma_index"
COLLECTION_NAME = "ignition_project"
LAST_INDEX_TIME_FILE = "last_index_time.pkl"

# Check if running in Docker with external Chroma
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
USE_PERSISTENT_CHROMA = os.getenv("USE_PERSISTENT_CHROMA", "false").lower() == "true"

app = typer.Typer()


def setup_chroma_client():
    """Set up and return a Chroma client with persistence."""
    # For tests, use in-memory client if specified
    if os.getenv("USE_IN_MEMORY_CHROMA", "false").lower() == "true":
        print("Using in-memory Chroma client for testing")
        return chromadb.Client()

    # For external Chroma connections
    if CHROMA_HOST and CHROMA_PORT:
        print(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
        if USE_PERSISTENT_CHROMA:
            # For persistent HTTP client mode (used by run_local.sh)
            print("Using persistent HTTP client mode")
            return chromadb.HttpClient(
                host=CHROMA_HOST,
                port=int(CHROMA_PORT),
                tenant="default_tenant",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        else:
            # Standard HTTP client
            return chromadb.HttpClient(
                host=CHROMA_HOST,
                port=int(CHROMA_PORT),
            )
    else:
        # Use local persistent Chroma
        print(f"Using local Chroma with persistence at {PERSIST_DIRECTORY}")
        return chromadb.PersistentClient(path=PERSIST_DIRECTORY)


def get_collection(client, rebuild=False):
    """Get or create a collection for the Ignition project."""
    if (
        rebuild
        and client.list_collections()
        and any(c.name == COLLECTION_NAME for c in client.list_collections())
    ):
        client.delete_collection(COLLECTION_NAME)

    return client.get_or_create_collection(name=COLLECTION_NAME)


def mock_embedding(text: str) -> List[float]:
    """Create a deterministic mock embedding based on the text content hash.

    This is used for testing without a valid OpenAI API key.
    """
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


def find_json_files(project_dir: str) -> List[str]:
    """Find all JSON files in the project directory that are likely Perspective views or Tag configurations."""
    json_files = []
    for root, _, files in os.walk(project_dir):
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                # Only include files that are likely Perspective views or Tag configurations
                # This is a simple heuristic and may need to be adjusted based on your project structure
                if "views" in fpath or "tag" in fpath.lower():
                    json_files.append(fpath)
    return json_files


def load_json_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load JSON content from each file with metadata."""
    documents = []
    for fpath in file_paths:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            # Determine metadata based on file path and content
            meta = {"filepath": fpath}
            if "views" in fpath:
                meta["type"] = "perspective"
                meta["name"] = os.path.splitext(os.path.basename(fpath))[0]
            elif "tag" in fpath.lower():
                meta["type"] = "tag"
                # Try to extract tag provider or folder from path
                parts = fpath.split(os.sep)
                if len(parts) > 1:
                    meta["folder"] = parts[-2]

            documents.append({"content": data, "metadata": meta})
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {fpath}")
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return documents


def chunk_perspective_view(
    view_json: Dict[str, Any], view_meta: Dict[str, str]
) -> List[tuple]:
    """Split a Perspective view JSON into semantically meaningful chunks."""
    chunks = []

    # Extract root components or use the whole JSON if no clear structure
    if isinstance(view_json, dict) and "root" in view_json:
        # Handle case where root has direct children
        if "children" in view_json["root"]:
            components = view_json["root"]["children"]
        else:
            components = [view_json["root"]]

        # Process root-level parameters if they exist
        if "params" in view_json["root"]:
            params_json = json.dumps(view_json["root"]["params"], ensure_ascii=False)
            if len(enc.encode(params_json)) <= MAX_TOKENS:
                meta = {**view_meta, "section": "params"}
                chunks.append((params_json, meta))
    else:
        components = [view_json]

    # Process view-level properties if they exist (non-root)
    if isinstance(view_json, dict) and "params" in view_json:
        params_json = json.dumps(view_json["params"], ensure_ascii=False)
        if len(enc.encode(params_json)) <= MAX_TOKENS:
            meta = {**view_meta, "section": "params"}
            chunks.append((params_json, meta))

    # Process each component
    for comp in components:
        process_component(comp, view_meta, chunks)

    return chunks


def process_component(comp, view_meta, chunks, parent_path=""):
    """Process a component and its children recursively."""
    comp_name = comp.get("name", "unnamed")
    current_path = f"{parent_path}/{comp_name}" if parent_path else comp_name

    # Create a copy with metadata for this component
    comp_meta = {**view_meta, "component": comp_name, "path": current_path}

    # Convert to string for token counting
    comp_json_str = json.dumps(comp, ensure_ascii=False)
    tokens = len(enc.encode(comp_json_str))

    if tokens <= MAX_TOKENS:
        # If small enough, add as one chunk
        chunks.append((comp_json_str, comp_meta))
    else:
        # If too large, split it
        if comp.get("children"):
            # Process children separately
            for child in comp["children"]:
                process_component(child, view_meta, chunks, current_path)

            # Also add the component without its children
            comp_copy = {k: v for k, v in comp.items() if k != "children"}
            comp_without_children = json.dumps(comp_copy, ensure_ascii=False)
            if len(enc.encode(comp_without_children)) <= MAX_TOKENS:
                chunks.append(
                    (comp_without_children, {**comp_meta, "section": "properties"})
                )
        else:
            # Split properties if no children but still too large
            props = list(comp.items())
            mid = len(props) // 2
            part1 = dict(props[:mid])
            part2 = dict(props[mid:])

            part1_str = json.dumps(part1, ensure_ascii=False)
            part2_str = json.dumps(part2, ensure_ascii=False)

            chunks.append((part1_str, {**comp_meta, "section": "properties_part1"}))
            chunks.append((part2_str, {**comp_meta, "section": "properties_part2"}))


def chunk_tag_config(tag_json: Any, tag_meta: Dict[str, str]) -> List[tuple]:
    """Split a Tag JSON into semantically meaningful chunks."""
    chunks = []
    tags_list = []

    # Extract the list of tags based on the structure
    if isinstance(tag_json, dict) and "tags" in tag_json:
        tags_list = tag_json["tags"]
    elif isinstance(tag_json, list):
        tags_list = tag_json
    else:
        tags_list = [tag_json]  # Treat as a single tag

    # Group tags by folder/path if possible
    tag_groups = {}
    for tag in tags_list:
        # Try to group by tag path or folder
        if isinstance(tag, dict):
            path = tag.get("path", "")
            folder = path.split("/")[0] if path and "/" in path else "root"
        else:
            folder = "root"

        if folder not in tag_groups:
            tag_groups[folder] = []
        tag_groups[folder].append(tag)

    # Process each group
    for folder, folder_tags in tag_groups.items():
        current_batch = []
        current_token_count = 0

        for tag in folder_tags:
            tag_str = json.dumps(tag, ensure_ascii=False)
            tag_tokens = len(enc.encode(tag_str))

            # If adding this tag would exceed MAX_TOKENS, start a new chunk
            if current_batch and current_token_count + tag_tokens > MAX_TOKENS:
                batch_str = "[\n" + ",\n".join(current_batch) + "\n]"
                folder_meta = {**tag_meta, "folder": folder}
                chunks.append((batch_str, folder_meta))
                current_batch = []
                current_token_count = 0

            current_batch.append(tag_str)
            current_token_count += tag_tokens

        # Add any remaining tags as the last chunk
        if current_batch:
            batch_str = "[\n" + ",\n".join(current_batch) + "\n]"
            folder_meta = {**tag_meta, "folder": folder}
            chunks.append((batch_str, folder_meta))

    return chunks


def create_chunks(documents: List[Dict[str, Any]]) -> List[tuple]:
    """Create chunks from all documents based on their type."""
    all_chunks = []

    for doc in documents:
        content = doc["content"]
        meta = doc["metadata"]

        if meta.get("type") == "perspective":
            chunks = chunk_perspective_view(content, meta)
        elif meta.get("type") == "tag":
            chunks = chunk_tag_config(content, meta)
        else:
            # Default chunking for unknown types
            content_str = json.dumps(content, ensure_ascii=False)
            chunks = [(content_str, meta)]

        all_chunks.extend(chunks)

    return all_chunks


def generate_embeddings(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI's API."""
    embeddings = []

    # Use mock embeddings if in mock mode
    if MOCK_EMBEDDINGS:
        print("Using mock embeddings for testing")
        for text in texts:
            embeddings.append(mock_embedding(text))
        return embeddings

    # Use OpenAI API for real embeddings
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            # Updated for OpenAI v1.0+
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(
                f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
            )
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            print("Falling back to mock embeddings for this batch")
            # Fall back to mock embeddings if the API call fails
            for text in batch:
                embeddings.append(mock_embedding(text))

    return embeddings


def index_documents(chunks: List[tuple], collection, rebuild: bool = False):
    """Index the chunks in the Chroma collection."""
    texts = [chunk[0] for chunk in chunks]
    metadatas = [chunk[1] for chunk in chunks]

    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embeddings(texts)

    # Create unique IDs for each chunk
    ids = []
    for idx, meta in enumerate(metadatas):
        # Create a unique ID based on filepath and chunk index
        file_id = os.path.basename(meta["filepath"])
        chunk_id = f"{file_id}-chunk{idx}"
        ids.append(chunk_id)

    # Add all embeddings, documents, and metadata to Chroma collection
    print("Adding chunks to vector database...")

    # If rebuilding, we already created a new collection earlier
    # Otherwise, we need to handle updates differently
    if not rebuild:
        # Get existing IDs to determine what to delete
        existing_ids = collection.get()["ids"] if collection.count() > 0 else []

        # Find IDs that belong to files we're re-indexing
        file_paths = {meta["filepath"] for meta in metadatas}
        ids_to_delete = [
            eid
            for eid in existing_ids
            if any(eid.startswith(os.path.basename(fp)) for fp in file_paths)
        ]

        # Delete those IDs before adding new ones
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} outdated chunks...")
            collection.delete(ids=ids_to_delete)

    # Add the new chunks
    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

    print(f"Successfully indexed {len(texts)} chunks.")


def save_last_index_time():
    """Save the current time as the last index time."""
    with open(LAST_INDEX_TIME_FILE, "wb") as f:
        pickle.dump(time.time(), f)


def load_last_index_time() -> float:
    """Load the last index time or return 0 if not available."""
    try:
        with open(LAST_INDEX_TIME_FILE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        return 0


@app.command()
def main(
    path: str = typer.Argument(..., help="Path to the Ignition project directory"),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Rebuild the index from scratch"
    ),
    changed_only: bool = typer.Option(
        False, "--changed-only", help="Only index files changed since last run"
    ),
    file: Optional[str] = typer.Option(
        None, "--file", help="Index only a specific file"
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock embeddings for testing without an OpenAI API key",
    ),
):
    """Main function to index Ignition project files."""
    print(f"Indexing Ignition project at: {path}")

    # Set mock mode if requested
    global MOCK_EMBEDDINGS
    if mock:
        MOCK_EMBEDDINGS = True
        print("Mock embedding mode enabled")

    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist")
        return

    # Initialize Chroma client and get collection
    client = setup_chroma_client()
    collection = get_collection(client, rebuild)

    # Find JSON files to index
    all_json_files = find_json_files(path)

    if file:
        # Index only a specific file
        file_path = os.path.join(path, file) if not os.path.isabs(file) else file
        if file_path in all_json_files:
            json_files = [file_path]
        else:
            print(f"Error: File {file_path} not found or is not a valid JSON file")
            return
    elif changed_only and not rebuild:
        # Index only files changed since last run
        last_index_time = load_last_index_time()
        json_files = [
            f for f in all_json_files if os.path.getmtime(f) > last_index_time
        ]
        print(
            f"Found {len(json_files)} changed files since {datetime.fromtimestamp(last_index_time)}"
        )
    else:
        # Index all files
        json_files = all_json_files
        print(f"Found {len(json_files)} JSON files to index")

    if not json_files:
        print("No files to index")
        return

    # Load and process the files
    documents = load_json_files(json_files)
    chunks = create_chunks(documents)

    # Index the chunks
    index_documents(chunks, collection, rebuild)

    # Save the current time as the last index time
    save_last_index_time()

    print("Indexing complete!")


if __name__ == "__main__":
    app()
