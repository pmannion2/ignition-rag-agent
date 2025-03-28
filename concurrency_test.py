#!/usr/bin/env python3
"""
Concurrency test script for the Ignition RAG API.

This script sends multiple concurrent requests to the API to test how well
it handles concurrency.
"""

import argparse
import asyncio
import os
import time

import requests

# Set environment variable to force in-memory Chroma
os.environ["USE_IN_MEMORY_CHROMA"] = "true"
os.environ["MOCK_EMBEDDINGS"] = "true"

# Default API URL
API_URL = "http://localhost:8001"


async def query_api(session_id, query_text, delay=0):
    """Make a query request to the API."""
    start_time = time.time()
    try:
        # Use agent/query endpoint with artificial_delay parameter
        response = requests.post(
            f"{API_URL}/agent/query",
            json={
                "query": f"{query_text} (request {session_id})",
                "top_k": 5,
                "use_mock": True,
                "artificial_delay": delay if delay > 0 else 0,
            },
            headers={"X-Mock-Embeddings": "true"},
            timeout=delay + 30 if delay > 0 else 30,
        )

        # Print response status and content if there's an error
        if response.status_code != 200:
            print(f"Request {session_id} error: Status {response.status_code}")
            print(f"Response: {response.text}")

        response.raise_for_status()
        duration = time.time() - start_time
        result = response.json()
        chunk_count = len(result.get("context_chunks", []))

        if delay > 0:
            print(
                f"Request {session_id} completed in {duration:.2f}s with artificial delay of {delay:.2f}s"
            )
        else:
            print(f"Request {session_id} completed in {duration:.2f}s, found {chunk_count} chunks")

        return {
            "session_id": session_id,
            "duration": duration,
            "status": response.status_code,
            "chunks": chunk_count,
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"Request {session_id} failed after {duration:.2f}s: {e}")
        return {
            "session_id": session_id,
            "duration": duration,
            "status": "error",
            "error": str(e),
        }


async def run_concurrent_queries(query_text, num_requests, delay=0):
    """Run multiple concurrent queries."""
    tasks = []
    for i in range(num_requests):
        tasks.append(asyncio.create_task(query_api(i + 1, query_text, delay)))

    print(f"Started {num_requests} concurrent requests...")
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    # Calculate statistics
    successful = sum(1 for r in results if r["status"] == 200)
    durations = [r["duration"] for r in results if r["status"] == 200]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Calculate concurrency benefit - how much faster than sequential
    sequential_estimate = sum(durations) if durations else 0
    speedup = sequential_estimate / total_time if total_time > 0 else 0

    # Calculate average chunks returned
    chunks = [r["chunks"] for r in results if r["status"] == 200]
    avg_chunks = sum(chunks) / len(chunks) if chunks else 0

    print("\nConcurrency Test Results:")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {successful}")
    print(f"Failed requests: {num_requests - successful}")
    print(f"Average request duration: {avg_duration:.2f}s")
    print(f"Average chunks per request: {avg_chunks:.1f}")
    print(f"Total test duration: {total_time:.2f}s")
    print(f"Estimated sequential time: {sequential_estimate:.2f}s")
    print(f"Speedup factor: {speedup:.2f}x")

    # Check for rate limiting or other errors
    errors = [r for r in results if r["status"] != 200]
    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  Request {err['session_id']}: {err.get('error', 'Unknown error')}")

    return results


def index_test_data():
    """Index the sample project."""
    print("Indexing sample project data with mock embeddings...")
    try:
        response = requests.post(
            f"{API_URL}/index",
            json={
                "project_path": "./sample_project",
                "rebuild": True,
                "skip_rate_limiting": True,
            },
            headers={"X-Mock-Embeddings": "true"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        print(
            f"Indexing completed: {result['indexed_files']} files, {result['total_chunks']} chunks"
        )
        return True
    except Exception as e:
        print(f"Error indexing data: {e}")
        return False


def main():
    global API_URL
    parser = argparse.ArgumentParser(description="Test API concurrency")
    parser.add_argument("--requests", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument(
        "--query", type=str, default="How does the tank level work", help="Query text"
    )
    parser.add_argument("--url", type=str, default=API_URL, help="API URL")
    parser.add_argument("--index", action="store_true", help="Index sample data before testing")
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Artificial delay in seconds to simulate slow requests",
    )
    args = parser.parse_args()

    API_URL = args.url

    # Check API health
    try:
        health = requests.get(f"{API_URL}/health")
        health.raise_for_status()
        print(f"API is healthy: {health.json()}")
    except Exception as e:
        print(f"API is not available: {e}")
        return

    # Index data if requested
    if args.index and not index_test_data():
        print("Failed to index test data, aborting test.")
        return

    # Run concurrency test
    asyncio.run(run_concurrent_queries(args.query, args.requests, args.delay))


if __name__ == "__main__":
    main()
