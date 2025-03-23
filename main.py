#!/usr/bin/env python3
import os
from pathlib import Path

import chromadb
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    print("Starting codebase indexing with OpenAI embeddings...")

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    print("Initialized OpenAI client")

    # Initialize Chroma client - using PersistentClient for local storage
    chroma_client = chromadb.PersistentClient(path="./chroma_index")
    print("Initialized Chroma client")

    # Get or create collection
    collection_name = "codebase"
    # Delete if exists
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    # Find Python files in the project (excluding .venv directory)
    project_path = Path(".")
    py_files = []
    for path in project_path.rglob("*.py"):
        if ".venv" not in str(path) and "__pycache__" not in str(path):
            py_files.append(str(path))

    print(f"Found {len(py_files)} Python files in the project")

    # Process each file
    doc_count = 0
    for file_path in py_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                continue

            # Generate embedding using OpenAI
            response = client.embeddings.create(
                input=content, model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            # Add to collection
            file_id = file_path.replace("/", "_")
            collection.add(
                ids=[file_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"source": file_path}],
            )
            doc_count += 1

            print(f"Indexed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Indexing complete. Collection now has {collection.count()} documents")
    print(f"Total documents indexed: {doc_count}")

    # Test a simple query
    query_text = "How does the indexer handle JSON files?"
    query_response = client.embeddings.create(
        input=query_text, model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    print("\nTest Query Results:")
    print(f"Query: '{query_text}'")
    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\nResult {i+1} from {metadata['source']}:")
        print(f"{doc[:150]}...")


if __name__ == "__main__":
    main()
