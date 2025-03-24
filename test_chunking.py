#!/usr/bin/env python3
import json
import os
import sys

import tiktoken
from dotenv import load_dotenv

# Add the current directory to path so we can import the indexer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from indexer import (
    HARD_TOKEN_LIMIT,
    chunk_by_characters,
    chunk_perspective_view,
    chunk_tag_config,
)

# Load environment variables
load_dotenv()

# Initialize tokenizer for GPT models
enc = tiktoken.get_encoding("cl100k_base")


def test_chunking_strategies(file_path, size_limit=None):
    """Test different chunking methods on a file and compare results.

    Args:
        file_path: Path to the file to test
        size_limit: If set, limit testing to files smaller than this size (in MB)
    """
    # Check file size if limit is set
    if size_limit:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > size_limit:
            print(
                f"Skipping {file_path} as it exceeds size limit ({file_size_mb:.1f} MB > {size_limit} MB)"
            )
            return

    print(f"Testing chunking on {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Count tokens in the file
        token_count = len(enc.encode(content))
        print(f"File has {token_count} tokens")

        # Try to parse as JSON to understand structure
        try:
            json_content = json.loads(content)
            print(f"Successfully parsed as JSON. Type: {type(json_content).__name__}")

            # Test appropriate chunking strategy based on file type
            if "view.json" in file_path:
                print("\nTesting perspective view chunking (Context-Preserving)...")
                meta = {
                    "filepath": file_path,
                    "name": os.path.basename(os.path.dirname(file_path)),
                }
                chunks = chunk_perspective_view(json_content, meta)
                print(f"Created {len(chunks)} chunks using Context-Preserving chunking")

                # Print chunk statistics
                print_chunk_stats(chunks, file_path, "View Context-Preserving")

            elif "tags" in file_path.lower():
                print("\nTesting tag hierarchical chunking...")
                meta = {
                    "filepath": file_path,
                    "folder": os.path.basename(file_path).split(".")[0],
                }
                chunks = chunk_tag_config(json_content, meta)
                print(f"Created {len(chunks)} chunks using Tag Hierarchy chunking")

                # Print chunk statistics
                print_chunk_stats(chunks, file_path, "Tag Hierarchy")

            else:
                # Fallback to character chunking for unknown content
                print("\nTesting fallback character-based chunking...")
                chunks = test_character_chunking(content)
                print(f"Created {len(chunks)} chunks via character chunking")

        except json.JSONDecodeError as e:
            print(f"Failed to parse as JSON: {e}")
            # Fall back to character chunking for non-JSON content
            chunks = test_character_chunking(content)
            print(f"Created {len(chunks)} chunks via character chunking")

    except Exception as e:
        print(f"Error processing file: {e}")


def test_character_chunking(content):
    """Test the character-based chunking strategy."""
    chunks = chunk_by_characters(content, int(HARD_TOKEN_LIMIT / 1.2))

    # Print chunk statistics
    for i, chunk in enumerate(chunks[:3]):  # Just show first 3 chunks
        chunk_tokens = len(enc.encode(chunk))
        print(f"Chunk {i}: {chunk_tokens} tokens, {len(chunk)} characters")

    return chunks


def print_chunk_stats(chunks, file_path, strategy_name):
    """Print statistics about the chunks."""
    print(f"\n{strategy_name} chunking results for {os.path.basename(file_path)}:")

    # Get total tokens in file for compression ratio calculation
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    file_token_count = len(enc.encode(file_content))

    total_tokens = 0
    max_tokens = 0
    min_tokens = float("inf")

    # Collect some statistics
    for i, (chunk_text, chunk_meta) in enumerate(chunks[:10]):  # Analyze up to 10 chunks
        tokens = len(enc.encode(chunk_text))
        total_tokens += tokens
        max_tokens = max(max_tokens, tokens)
        min_tokens = min(min_tokens, tokens)

        # Print context info for context-preserving chunks
        if "context" in chunk_meta:
            context = json.loads(chunk_meta["context"])
            context_preview = (
                str(context)[:100] + "..." if len(str(context)) > 100 else str(context)
            )
            print(f"Chunk {i}: {tokens} tokens, section: {chunk_meta.get('section', 'unknown')}")
            print(f"  Context: {context_preview}")
        else:
            print(f"Chunk {i}: {tokens} tokens, section: {chunk_meta.get('section', 'unknown')}")

    # Print overall statistics
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"Average tokens per chunk: {total_tokens / min(len(chunks), 10):.1f}")
        print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}")
        print(f"Compression ratio: {file_token_count / total_tokens:.2f}x")


def main():
    # Define a size limit for testing (in MB)
    size_limit = 20  # Skip files larger than 20MB for this test

    # Process problematic large files
    large_tags = [
        "whk-ignition-scada/tags/WHK01.json",
        "whk-ignition-scada/tags/MQTT Engine.json",
        "whk-ignition-scada/tags/default.json",
        "whk-ignition-scada/tags/QSI_UDTs.json",
    ]

    # Process view files
    view_files = [
        "whk-ignition-scada/com.inductiveautomation.perspective/views/Main/sidemenu/view.json",
        "whk-ignition-scada/com.inductiveautomation.perspective/views/Exchange/MettlerToledoLibrary/Popups/IND/IND360/view.json",
        "whk-ignition-scada/com.inductiveautomation.perspective/views/Exchange/MettlerToledoLibrary/Popups/Embedded Views/StatusIndicator/view.json",
    ]

    # Test both types of files
    for file_path in large_tags:
        if os.path.exists(file_path):
            test_chunking_strategies(file_path, size_limit)
            print("\n" + "=" * 50 + "\n")
        else:
            print(f"File not found: {file_path}")

    for file_path in view_files:
        if os.path.exists(file_path):
            test_chunking_strategies(file_path, size_limit)
            print("\n" + "=" * 50 + "\n")
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
