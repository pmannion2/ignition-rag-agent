#!/usr/bin/env python3
import os
import json
import time
import typer
from pathlib import Path
from typing import Optional

import chromadb
import openai
import tiktoken
from dotenv import load_dotenv
from indexer import create_chunks, enc, MAX_TOKENS

# Load environment variables
load_dotenv()

# Max tokens per chunk - must stay under model limit of 8192
MAX_CHUNK_SIZE = 7000
# Hard limit for token count safety (OpenAI's limit is 8192)
HARD_TOKEN_LIMIT = 7500

app = typer.Typer()


@app.command()
def main(
    skip_rate_limiting: bool = typer.Option(
        False,
        "--skip-rate-limiting",
        help="Skip rate limiting for faster processing (use with caution)",
    )
):
    print("Starting Ignition project indexing with OpenAI embeddings...")

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    print("Initialized OpenAI client")

    # Initialize Chroma client - using PersistentClient for local storage
    chroma_client = chromadb.PersistentClient(path="./chroma_index")
    print("Initialized Chroma client")

    # Get or create collection
    collection_name = "ignition_project"
    # Delete if exists
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    # Path to the Ignition project
    project_path = Path("./whk-ignition-scada")

    # Find JSON files in the Ignition project
    json_files = []
    for path in project_path.rglob("*.json"):
        json_files.append(str(path))

    print(f"Found {len(json_files)} JSON files in the Ignition project")

    # Process each file
    doc_count = 0
    chunk_count = 0

    # Rate limiting variables
    tokens_in_minute = 0
    minute_start = time.time()
    MAX_TOKENS_PER_MINUTE = 80000  # Conservative limit below OpenAI's 100K TPM

    for file_path in json_files:
        try:
            print(f"Processing {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                print(f"Skipping empty file: {file_path}")
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
            print(f"File has {token_count} tokens")

            # Always chunk files over 3000 tokens to ensure safer processing
            # This avoids context length errors that can happen with certain file structures
            if token_count <= 3000:
                # Check rate limits
                if (
                    not skip_rate_limiting
                    and tokens_in_minute + token_count > MAX_TOKENS_PER_MINUTE
                ):
                    # Wait until the minute is up
                    elapsed = time.time() - minute_start
                    if elapsed < 60:
                        sleep_time = 60 - elapsed
                        print(
                            f"Rate limit approaching. Sleeping for {sleep_time:.1f} seconds..."
                        )
                        time.sleep(sleep_time)
                    # Reset rate limit counter
                    tokens_in_minute = 0
                    minute_start = time.time()

                try:
                    # Generate embedding using OpenAI
                    response = client.embeddings.create(
                        input=content, model="text-embedding-ada-002"
                    )
                    embedding = response.data[0].embedding

                    # Update rate limit counter
                    tokens_in_minute += token_count

                    # Add to collection
                    file_id = file_path.replace("/", "_").replace("\\", "_")
                    collection.add(
                        ids=[file_id],
                        documents=[content],
                        embeddings=[embedding],
                        metadatas=[metadata],
                    )
                    doc_count += 1
                    chunk_count += 1
                    print(f"Indexed {file_path} as a single chunk")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                # For large files, we need to chunk the content
                print(f"File exceeds token limit, chunking: {file_path}")

                try:
                    # For extremely large files (>50K tokens), use a character-level chunking approach
                    # This is more reliable for massive files
                    if token_count > 50000:
                        print(
                            f"Very large file ({token_count} tokens), using character-level chunking"
                        )
                        chunks = []

                        # For json files, try to intelligently split on braces or brackets for better semantic chunks
                        if file_path.endswith(".json"):
                            try:
                                # Try to parse as JSON first to extract key structures
                                json_content = json.loads(content)
                                print(
                                    f"Successfully parsed JSON for {file_path} - type: {type(json_content).__name__}"
                                )

                                # For array-type JSONs, split at the top level
                                if (
                                    isinstance(json_content, list)
                                    and len(json_content) > 1
                                ):
                                    print(
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
                                        if item_tokens > HARD_TOKEN_LIMIT:
                                            # Process any accumulated items
                                            if current_array:
                                                array_str = json.dumps(current_array)
                                                sub_chunks.append(array_str)
                                                current_array = []
                                                current_tokens = 0

                                            # Chunk this large item by characters, preserving JSON format
                                            item_chunks = chunk_by_characters(
                                                item_str,
                                                int(
                                                    HARD_TOKEN_LIMIT / 1.2
                                                ),  # Fixed integer division
                                            )
                                            sub_chunks.extend(item_chunks)
                                        # If adding this would exceed limit, create a new chunk
                                        elif (
                                            current_tokens + item_tokens
                                            > HARD_TOKEN_LIMIT
                                        ):
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
                                else:
                                    # For other JSON structures, fall back to character-level chunking
                                    text_chunks = chunk_by_characters(
                                        content,
                                        int(
                                            HARD_TOKEN_LIMIT / 1.2
                                        ),  # Fixed integer division
                                    )
                                    chunks = [
                                        (chunk, metadata) for chunk in text_chunks
                                    ]
                            except json.JSONDecodeError:
                                # If JSON parsing fails, use character-level chunking
                                text_chunks = chunk_by_characters(
                                    content,
                                    int(
                                        HARD_TOKEN_LIMIT / 1.2
                                    ),  # Fixed integer division
                                )
                                chunks = [(chunk, metadata) for chunk in text_chunks]
                        else:
                            # For non-JSON files, use character-level chunking
                            text_chunks = chunk_by_characters(
                                content,
                                int(HARD_TOKEN_LIMIT / 1.2),  # Fixed integer division
                            )
                            chunks = [(chunk, metadata) for chunk in text_chunks]
                    else:
                        # Parse JSON content
                        json_content = json.loads(content)

                        # Prepare document for chunking
                        document = {"content": json_content, "metadata": metadata}

                        # Create chunks using indexer's chunking logic
                        chunks = create_chunks([document])

                    print(f"Created {len(chunks)} chunks for {file_path}")

                    # Process each chunk with rate limiting
                    for i, (chunk_text, chunk_meta) in enumerate(chunks):
                        try:
                            # Verify chunk size isn't too large
                            chunk_token_count = len(enc.encode(chunk_text))

                            # Skip chunks that are still too large
                            if chunk_token_count > 8000:
                                print(
                                    f"Warning: Chunk {i} is too large ({chunk_token_count} tokens). Skipping."
                                )
                                continue

                            # Check and handle rate limits
                            if (
                                not skip_rate_limiting
                                and tokens_in_minute + chunk_token_count
                                > MAX_TOKENS_PER_MINUTE
                            ):
                                elapsed = time.time() - minute_start
                                if elapsed < 60:
                                    sleep_time = 60 - elapsed
                                    print(
                                        f"Rate limit approaching. Sleeping for {sleep_time:.1f} seconds..."
                                    )
                                    time.sleep(sleep_time)
                                # Reset rate limit counter
                                tokens_in_minute = 0
                                minute_start = time.time()

                            # Generate embedding using OpenAI
                            response = client.embeddings.create(
                                input=chunk_text, model="text-embedding-ada-002"
                            )
                            embedding = response.data[0].embedding

                            # Update rate limit counter
                            if not skip_rate_limiting:
                                tokens_in_minute += chunk_token_count

                            # Create a unique ID for this chunk
                            chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"

                            # Add to collection
                            collection.add(
                                ids=[chunk_id],
                                documents=[chunk_text],
                                embeddings=[embedding],
                                metadatas=[chunk_meta],
                            )
                            chunk_count += 1

                            # No delay between chunks for faster processing
                            if (
                                not skip_rate_limiting and i > 0 and i % 20 == 0
                            ):  # Changed from 10 to 20
                                print(
                                    f"Processed {i}/{len(chunks)} chunks for {file_path}"
                                )

                        except Exception as e:
                            print(f"Error processing chunk {i} of {file_path}: {e}")

                    doc_count += 1
                    print(f"Indexed {len(chunks)} chunks from {file_path}")

                except json.JSONDecodeError:
                    print(f"Error parsing JSON content for {file_path}, skipping")
                except Exception as e:
                    print(f"Error chunking {file_path}: {e}")
        except Exception as e:
            print(f"Error opening/reading {file_path}: {e}")

    print(
        f"Indexing complete. Collection now has {collection.count()} documents/chunks"
    )
    print(f"Total files indexed: {doc_count}")
    print(f"Total chunks indexed: {chunk_count}")

    # Test a simple query
    if chunk_count > 0:
        query_text = "Show me all the tank level configurations"
        query_response = client.embeddings.create(
            input=query_text, model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        print("\nTest Query Results:")
        print(f"Query: '{query_text}'")
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            print(f"\nResult {i+1} from {metadata['source']}:")
            print(f"{doc[:250]}...")


def chunk_by_characters(text, max_chunk_size):
    """Chunk text by characters, ensuring no chunk exceeds the token limit."""
    chunks = []

    # Convert max_chunk_size from tokens to approximate characters (rough estimate)
    # Typically 1 token ≈ 4 characters for English text
    max_chars = int(max_chunk_size * 3)

    # Initialize chunking variables
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = min(start + max_chars, text_length)

        # If we're not at the end, try to find a good break point
        if end < text_length:
            # Try to find a natural break (newline, period, comma, etc.)
            natural_breaks = ["\n\n", "\n", ". ", ", ", " ", ".", ","]

            for separator in natural_breaks:
                # Look for the separator within a window near the end
                window_size = min(200, max_chars // 4)
                window_start = max(start, end - window_size)

                # Find the last occurrence of the separator in this window
                last_sep = text.rfind(separator, window_start, end)

                if last_sep > window_start:
                    end = last_sep + len(separator)
                    break

        # Extract the chunk
        chunk = text[start:end]

        # Verify token count for safety
        token_count = len(enc.encode(chunk))
        if token_count > HARD_TOKEN_LIMIT:
            # If still too large, use a more aggressive approach
            print(
                f"Warning: Chunk still too large ({token_count} tokens). Forcing smaller size."
            )
            # Reduce max_chars and try again from this starting point
            max_chars = max_chars // 2
            continue

        # Add chunk and move to next position
        chunks.append(chunk)
        start = end

    return chunks


if __name__ == "__main__":
    app()
