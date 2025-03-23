#!/usr/bin/env python3
"""
Cursor Client for Ignition RAG

This script provides a client for integrating Ignition RAG with Cursor IDE.
It connects directly to the RAG API and can be used as a standalone script or imported.
"""

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001")


def get_rag_context(query, current_file=None, top_k=3, filter_type=None):
    """
    Get context from the RAG API for a given query.

    Args:
        query (str): The query to search for
        current_file (str, optional): Path to the current file being edited
        top_k (int, optional): Number of results to return
        filter_type (str, optional): Filter by document type (perspective or tag)

    Returns:
        str: Context to be used in Cursor
    """
    try:
        # Prepare the request
        endpoint = f"{RAG_API_URL}/agent/query"
        payload = {
            "query": query,
            "top_k": top_k,
            "filter_type": filter_type,
            "context": {"current_file": current_file} if current_file else None,
        }

        # Make the request
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()

        # Parse the response
        data = response.json()

        # Use suggested prompt if available
        if data.get("suggested_prompt"):
            return data["suggested_prompt"]

        # Otherwise, format the context manually
        context_chunks = data.get("context_chunks", [])
        if not context_chunks:
            return f"No relevant context found for query: {query}"

        # Format the context as a Cursor-friendly string
        context_text = f"# RAG Context for: {query}\n\n"

        for i, chunk in enumerate(context_chunks):
            content = chunk.get("content", "")
            source = chunk.get("source", "Unknown")
            metadata = chunk.get("metadata", {})
            file_path = metadata.get("filepath", "Unknown file")

            context_text += f"## Context {i+1}: {source} ({file_path})\n"
            context_text += "```json\n"
            context_text += content + "\n"
            context_text += "```\n\n"

        return context_text

    except Exception as e:
        return f"Error retrieving context: {str(e)}"


def cursor_integration(query, current_file=None):
    """
    Format the RAG context for Cursor integration.
    This function follows Cursor's expected format for agent responses.

    Args:
        query (str): User query
        current_file (str, optional): Current file being edited

    Returns:
        dict: Response in Cursor format
    """
    context = get_rag_context(query, current_file)

    return {"content": context, "role": "assistant"}


def main():
    """Command line interface for the Cursor client."""
    parser = argparse.ArgumentParser(description="Cursor Client for Ignition RAG")
    parser.add_argument("query", help="The query to search for")
    parser.add_argument("--file", "-f", help="Path to the current file")
    parser.add_argument(
        "--top-k", "-k", type=int, default=3, help="Number of results to return"
    )
    parser.add_argument("--filter", help="Filter by document type (perspective or tag)")
    parser.add_argument(
        "--output", "-o", help="Output format (text or json)", default="text"
    )

    args = parser.parse_args()

    if args.output == "json":
        result = cursor_integration(args.query, args.file)
        print(json.dumps(result, indent=2))
    else:
        context = get_rag_context(args.query, args.file, args.top_k, args.filter)
        print(context)


if __name__ == "__main__":
    main()
