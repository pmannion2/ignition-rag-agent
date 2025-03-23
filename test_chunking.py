#!/usr/bin/env python3
import os
import json
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tokenizer for GPT models
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 7000  # Target max tokens per chunk
HARD_TOKEN_LIMIT = 7500  # Hard limit for safety


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


def process_large_file(file_path):
    """Process a large file and test the chunking mechanism."""
    print(f"Testing chunking on {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count tokens in the file
        token_count = len(enc.encode(content))
        print(f"File has {token_count} tokens")

        # Try to parse as JSON to understand structure
        try:
            json_content = json.loads(content)
            print(f"Successfully parsed as JSON. Type: {type(json_content).__name__}")

            if isinstance(json_content, list):
                print(f"JSON is an array with {len(json_content)} items")

                # Sample a few items
                for i in range(min(3, len(json_content))):
                    item = json_content[i]
                    item_str = json.dumps(item)
                    item_tokens = len(enc.encode(item_str))
                    print(
                        f"Item {i}: {item_tokens} tokens, type: {type(item).__name__}"
                    )

            # Test chunking - character based
            print("\nTesting character-based chunking...")
            chunks = chunk_by_characters(content, int(HARD_TOKEN_LIMIT / 1.2))
            print(f"Created {len(chunks)} chunks")

            # Print chunk statistics
            for i, chunk in enumerate(chunks[:3]):  # Just show first 3 chunks
                chunk_tokens = len(enc.encode(chunk))
                print(f"Chunk {i}: {chunk_tokens} tokens, {len(chunk)} characters")

        except json.JSONDecodeError as e:
            print(f"Failed to parse as JSON: {e}")
            # Fall back to character chunking for non-JSON content
            chunks = chunk_by_characters(content, int(HARD_TOKEN_LIMIT / 1.2))
            print(f"Created {len(chunks)} chunks via character chunking")

    except Exception as e:
        print(f"Error processing file: {e}")


def main():
    # Process a few problematic large files
    large_files = [
        "whk-ignition-scada/tags/WHK01.json",
        "whk-ignition-scada/tags/MQTT Engine.json",
        "whk-ignition-scada/tags/default.json",
        "whk-ignition-scada/tags/QSI_UDTs.json",
    ]

    for file_path in large_files:
        if os.path.exists(file_path):
            process_large_file(file_path)
            print("\n" + "=" * 50 + "\n")
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
