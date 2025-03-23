#!/usr/bin/env python3
"""
Cursor Agent Integration for Ignition RAG

This module provides functions that can be used by Cursor's Agent mode to
retrieve context from the Ignition RAG system when generating code or answering questions.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from logger import get_logger

# Load environment variables
load_dotenv()

# Set up logging
logger = get_logger("cursor_agent")

# RAG API endpoint configuration
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001")

# Check if we should use mock embeddings
USE_MOCK_EMBEDDINGS = os.environ.get("MOCK_EMBEDDINGS", "").lower() in (
    "true",
    "1",
    "t",
)


def mock_embedding(text: str) -> List[float]:
    """Mock embedding function that returns a fixed vector."""
    return [0.1] * 1536  # Same dimensionality as OpenAI's text-embedding-ada-002


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

        # Check for mock mode
        if USE_MOCK_EMBEDDINGS:
            return {
                "context_chunks": [
                    {
                        "content": "This is mock content for testing",
                        "source": "Mock source",
                        "metadata": {"type": "mock", "filepath": "mock_file.json"},
                        "similarity": 0.95,
                    }
                ],
                "suggested_prompt": f"Query: {query}\n\nRelevant context from mock data",
                "mock_used": True,
            }

        # Make the request
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Return the result data
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ignition RAG API: {e}")
        return {"context_chunks": [], "suggested_prompt": None, "error": str(e)}


def get_cursor_context(
    user_query: str,
    cursor_context: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> str:
    """
    Get relevant context for a Cursor query based on the current cursor position
    and user query.

    Args:
        user_query: The user's query text
        cursor_context: Dictionary containing cursor context (current file, selection, etc.)
        top_k: Number of top results to return

    Returns:
        A string containing the suggested prompt with context
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
        query=user_query,
        top_k=top_k,
        filter_type=filter_type,
        current_file=current_file,
    )

    # Extract or build the context string
    if result.get("suggested_prompt"):
        return result["suggested_prompt"]

    # If no suggested prompt, but we have context chunks, build our own context
    context_chunks = result.get("context_chunks", [])
    if context_chunks:
        context_str = f"Relevant Ignition project context for: {user_query}\n\n"

        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown source")
            content = chunk.get("content", "")

            context_str += f"--- Context {i}: {source} ---\n"
            context_str += f"{content}\n\n"

        context_str += "Use the above context to help answer the query or generate appropriate code.\n"
        return context_str

    # No context available
    return f"No relevant Ignition project context found for: {user_query}"


def get_ignition_tag_info(tag_name: str) -> dict:
    """
    Get information about a specific Ignition tag.

    Args:
        tag_name: The name of the tag to look up

    Returns:
        Dictionary containing tag information
    """
    # Check if mock mode is enabled
    if USE_MOCK_EMBEDDINGS:
        logger.info(f"Using mock data for tag: {tag_name}")
        return {
            "name": tag_name,
            "value": 42.0,
            "path": f"Tags/{tag_name}",
            "tagType": "AtomicTag",
            "dataType": "Float8",
            "parameters": {
                "engUnit": "%",
                "description": f"Mock description for {tag_name}",
                "engHigh": 100,
                "engLow": 0,
            },
            "mock_used": True,
        }

    # Get tag information from the RAG system
    rag_results = query_rag(
        query=f"Tag configuration for {tag_name}", top_k=1, filter_type="tag"
    )

    # Extract tag info from the context
    context_chunks = rag_results.get("context_chunks", [])
    if not context_chunks:
        logger.warning(f"No tag information found for {tag_name}")
        return {"error": f"No tag information found for {tag_name}"}

    # Process the first matching chunk
    for chunk in context_chunks:
        content = chunk.get("content", "")
        try:
            # Try to parse the tag information from the content
            tag_info_str = content.strip()
            tag_info = json.loads(tag_info_str)
            if "name" in tag_info and tag_info["name"].lower() == tag_name.lower():
                return tag_info
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing tag information: {e}")
            continue

    return {"error": f"Could not parse tag information for {tag_name}"}


def get_ignition_view_component(
    view_name: str, component_name: Optional[str] = None
) -> dict:
    """
    Get information about a specific view or component in an Ignition project.

    Args:
        view_name: The name of the view to look up
        component_name: Optional name of the component within the view

    Returns:
        Dictionary containing view or component information
    """
    # Check if mock mode is enabled
    if USE_MOCK_EMBEDDINGS:
        logger.info(
            f"Using mock data for view: {view_name}, component: {component_name}"
        )
        # Create a mock response
        if component_name:
            return {
                "view": view_name,
                "component": component_name,
                "type": "Label" if "label" in component_name.lower() else "Tank",
                "properties": {
                    "x": 100,
                    "y": 100,
                    "width": 200,
                    "height": 150,
                    "text": (
                        f"Mock {component_name}"
                        if "label" in component_name.lower()
                        else None
                    ),
                },
                "mock_used": True,
            }
        else:
            return {
                "name": view_name,
                "path": f"views/{view_name}.json",
                "components": [
                    {"name": "Tank1", "type": "Tank"},
                    {"name": "Label1", "type": "Label"},
                ],
                "size": {"width": 800, "height": 600},
                "mock_used": True,
            }

    # Build the query based on what we're looking for
    if component_name:
        query = f"Component {component_name} in view {view_name}"
    else:
        query = f"View configuration for {view_name}"

    # Get view/component information from the RAG system
    rag_results = query_rag(query=query, top_k=2, filter_type="perspective")

    # Extract view/component info from the context
    context_chunks = rag_results.get("context_chunks", [])
    if not context_chunks:
        logger.warning(f"No view information found for {view_name}")
        return {"error": f"No view information found for {view_name}"}

    # Combine the relevant context
    view_info = {}
    for chunk in context_chunks:
        content = chunk.get("content", "")
        try:
            # Try to parse the JSON content
            content_obj = json.loads(content.strip())

            # For component search
            if (
                component_name
                and "name" in content_obj
                and content_obj["name"] == component_name
            ):
                return content_obj

            # For view search
            if "name" in content_obj and content_obj["name"] == view_name:
                view_info = content_obj
                break

            # For partial view information
            view_info.update(content_obj)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing view information: {e}")
            continue

    if not view_info:
        return {"error": f"Could not parse view information for {view_name}"}

    return view_info


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
