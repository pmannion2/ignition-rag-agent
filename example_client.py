#!/usr/bin/env python3
import argparse
import json

import requests


def query_rag_api(
    query, top_k=5, filter_type=None, filter_path=None, api_url="http://localhost:8000"
):
    """Query the RAG API and return results."""
    # Prepare the request
    endpoint = f"{api_url}/query"
    payload = {"query": query, "top_k": top_k}

    # Add optional filters if provided
    if filter_type:
        payload["filter_type"] = filter_type
    if filter_path:
        payload["filter_path"] = filter_path

    # Make the request
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the JSON response
        data = response.json()

        return data
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        return None


def format_results(response_data):
    """Format the API response for display."""
    if not response_data or "results" not in response_data:
        return "No results found."

    results = response_data["results"]
    total = response_data["total"]

    if not results:
        return "No matching results found."

    output = f"Found {total} relevant results:\n\n"

    for i, result in enumerate(results, 1):
        # Format the content (pretty-print JSON if possible)
        try:
            content_json = json.loads(result["content"])
            formatted_content = json.dumps(content_json, indent=2)
        except json.JSONDecodeError:
            formatted_content = result["content"]

        # Format the metadata
        metadata = result["metadata"]
        filepath = metadata.get("filepath", "unknown")
        doc_type = metadata.get("type", "unknown")

        # Add additional context based on the document type
        if doc_type == "perspective":
            component = metadata.get("component", "")
            component_info = f"\nComponent: {component}" if component else ""
            section = metadata.get("section", "")
            section_info = f"\nSection: {section}" if section else ""
            context = component_info + section_info
        elif doc_type == "tag":
            folder = metadata.get("folder", "")
            folder_info = f"\nFolder: {folder}" if folder else ""
            context = folder_info
        else:
            context = ""

        # Format the entire result
        output += f"Result {i} (Similarity: {result['similarity']:.4f}):\n"
        output += f"Source: {filepath} (Type: {doc_type}){context}\n"
        output += "Content:\n"
        output += f"{formatted_content}\n\n"
        output += "-" * 80 + "\n\n"

    return output


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Query the Ignition RAG API")
    parser.add_argument("query", help="The natural language query to search for")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of results to return (default: 3)"
    )
    parser.add_argument(
        "--filter-type", choices=["perspective", "tag"], help="Filter by document type"
    )
    parser.add_argument("--filter-path", help="Filter by file path pattern")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Query the API
    results = query_rag_api(
        query=args.query,
        top_k=args.top_k,
        filter_type=args.filter_type,
        filter_path=args.filter_path,
        api_url=args.api_url,
    )

    # Format and display results
    if results:
        print(format_results(results))
    else:
        print("Failed to get results from the API.")
        print("Make sure the API server is running and accessible.")


if __name__ == "__main__":
    main()
