#!/usr/bin/env python3
"""
Cursor Agent Integration for Ignition RAG

This module provides functions that can be used by Cursor's Agent mode to
retrieve context from the Ignition RAG system when generating code or answering questions.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default API endpoint
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001")


def query_rag(
    query: str,
    top_k: int = 3,
    filter_type: Optional[str] = None,
    current_file: Optional[str] = None,
) -> dict:
    """
    Query the RAG system for relevant Ignition project context.

    Args:
        query: The natural language query to search for
        top_k: Number of results to return
        filter_type: Optional filter for document type (perspective or tag)
        current_file: The file currently being edited (for contextual relevance)

    Returns:
        Dictionary containing context chunks and a suggested prompt
    """
    try:
        # Prepare the request to the agent-optimized endpoint
        endpoint = f"{RAG_API_URL}/agent/query"
        payload = {
            "query": query,
            "top_k": top_k,
            "filter_type": filter_type,
            "context": {"current_file": current_file} if current_file else None,
        }

        # Make the request
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Return the result data
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error querying Ignition RAG API: {e}")
        return {"context_chunks": [], "suggested_prompt": None, "error": str(e)}


def get_cursor_context(
    query: str, cursor_context: Dict[str, Any], top_k: int = 3
) -> str:
    """
    Get context from the RAG system specifically formatted for Cursor Agent.
    This function integrates with Cursor's context format.

    Args:
        query: The user's query to search for relevant context
        cursor_context: Context provided by Cursor about the current environment
        top_k: Number of results to return

    Returns:
        String containing the formatted context for the LLM prompt
    """
    # Extract relevant information from cursor context
    current_file = cursor_context.get("current_file")

    # Determine if we should filter by type based on file extension
    filter_type = None
    if current_file:
        if current_file.endswith(".java"):
            # If working with Java, more likely to be interested in Tags
            filter_type = "tag"
        elif current_file.endswith(".js") or current_file.endswith(".ts"):
            # If working with JS/TS, more likely interested in Perspective views
            filter_type = "perspective"

    # Query the RAG system
    result = query_rag(
        query=query, top_k=top_k, filter_type=filter_type, current_file=current_file
    )

    # Extract or build the context string
    if result.get("suggested_prompt"):
        return result["suggested_prompt"]

    # If no suggested prompt, but we have context chunks, build our own context
    context_chunks = result.get("context_chunks", [])
    if context_chunks:
        context_str = f"Relevant Ignition project context for: {query}\n\n"

        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown source")
            content = chunk.get("content", "")

            context_str += f"--- Context {i}: {source} ---\n"
            context_str += f"{content}\n\n"

        context_str += "Use the above context to help answer the query or generate appropriate code.\n"
        return context_str

    # No context available
    return f"No relevant Ignition project context found for: {query}"


def get_ignition_tag_info(tag_name: str) -> dict:
    """
    Get specific information about an Ignition tag by name.

    Args:
        tag_name: The name of the tag to look up

    Returns:
        Dictionary with tag information or empty dict if not found
    """
    result = query_rag(
        query=f"Tag configuration for {tag_name}", top_k=1, filter_type="tag"
    )

    context_chunks = result.get("context_chunks", [])
    if not context_chunks:
        return {}

    # Try to parse the tag JSON
    try:
        chunk = context_chunks[0]
        content = chunk.get("content", "{}")
        tag_data = json.loads(content)

        # If it's a list (common for tag exports), look for the specific tag
        if isinstance(tag_data, list):
            for tag in tag_data:
                if tag.get("name") == tag_name:
                    return tag

        # If it's just the tag itself
        elif isinstance(tag_data, dict) and tag_data.get("name") == tag_name:
            return tag_data

        # Otherwise return the first chunk as best effort
        return tag_data

    except (json.JSONDecodeError, IndexError):
        return {}


def get_ignition_view_component(
    view_name: str, component_name: Optional[str] = None
) -> dict:
    """
    Get information about a Perspective view or specific component.

    Args:
        view_name: The name of the Perspective view
        component_name: Optional specific component name to look for

    Returns:
        Dictionary with view/component information or empty dict if not found
    """
    # Build query based on parameters
    if component_name:
        query = f"Component {component_name} in view {view_name}"
    else:
        query = f"Perspective view {view_name}"

    result = query_rag(query=query, top_k=3, filter_type="perspective")

    context_chunks = result.get("context_chunks", [])
    if not context_chunks:
        return {}

    # Process results
    try:
        # If looking for a specific component
        if component_name:
            for chunk in context_chunks:
                content = chunk.get("content", "{}")
                component_data = json.loads(content)

                if component_data.get("name") == component_name:
                    return component_data

        # Just return the first chunk's content as best effort
        chunk = context_chunks[0]
        content = chunk.get("content", "{}")
        return json.loads(content)

    except (json.JSONDecodeError, IndexError):
        return {}


# Example of how to use in Cursor Agent mode
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cursor_agent.py '<query>'")
        sys.exit(1)

    query = sys.argv[1]
    # Mock cursor context for testing
    mock_context = {"current_file": "ignition_project/views/tank_view.json"}

    context = get_cursor_context(query, mock_context)
    print(context)
