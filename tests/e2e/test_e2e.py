#!/usr/bin/env python3
import json
import os
import sys

import pytest
import requests

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestE2E:
    """End-to-end tests for the Ignition RAG system."""

    def test_api_health(self, api_url):
        """Test API health endpoint."""
        response = requests.get(f"{api_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_query_endpoint(self, api_url):
        """Test the query endpoint with a basic question."""
        query_data = {
            "query": "Tell me about the tank system",
            "top_k": 3,
            "filter_metadata": {},
        }

        response = requests.post(f"{api_url}/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data
        assert "total_chunks" in data["metadata"]

        # The collection might be empty in some E2E tests
        # Only check result structure if we have results
        if data["metadata"]["total_chunks"] > 0 and len(data["results"]) > 0:
            # Check first result has the expected structure
            first_result = data["results"][0]
            assert "content" in first_result
            assert "metadata" in first_result
            assert "similarity" in first_result

    def test_multi_turn_conversation(self, api_url):
        """Test a multi-turn conversation."""
        # First query
        query1_data = {
            "query": "What is in the tank view?",
            "top_k": 3,
            "filter_metadata": {"type": "perspective"},
        }

        response1 = requests.post(f"{api_url}/query", json=query1_data)

        assert response1.status_code == 200
        data1 = response1.json()
        assert "results" in data1
        assert "metadata" in data1
        assert "total_chunks" in data1["metadata"]

        # Follow-up query
        query2_data = {
            "query": "Tell me more about its components",
            "top_k": 3,
            "filter_metadata": {},
        }

        response2 = requests.post(f"{api_url}/query", json=query2_data)

        assert response2.status_code == 200
        data2 = response2.json()
        assert "results" in data2
        assert "metadata" in data2
        assert "total_chunks" in data2["metadata"]
