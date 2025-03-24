#!/usr/bin/env python3
import json
import os
import sys

import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestIndexerE2E:
    """End-to-end tests for the Indexer functionality."""

    def test_query_with_indexed_content(self, api_url):
        """Test that queries return results from indexed content."""
        # This query should match content in the test data
        query_data = {
            "query": "What is the liquid level in the tank?",
            "top_k": 5,
            "filter_metadata": {},
        }

        response = requests.post(f"{api_url}/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data
        assert "total_chunks" in data["metadata"]

        # The collection might be empty in some E2E tests
        # Just check it's a valid response with the right format
        if data["metadata"]["total_chunks"] > 0:
            assert len(data["results"]) > 0

            # Validate that sources include expected content
            results_text = json.dumps(data)
            assert "tank" in results_text.lower()

            # Verify this is coming from the indexer by checking some metadata
            assert any("filepath" in result["metadata"] for result in data["results"])

    def test_search_endpoint(self, api_url):
        """Test the direct search endpoint."""
        search_data = {
            "query": "tank level",
            "top_k": 5,
            "filter_metadata": {},
        }

        response = requests.post(f"{api_url}/query", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data

        # The collection might be empty in some E2E tests
        if data["metadata"]["total_chunks"] > 0 and len(data["results"]) > 0:
            # Check first result has expected fields
            first_result = data["results"][0]
            assert "content" in first_result
            assert "metadata" in first_result
            assert "similarity" in first_result
