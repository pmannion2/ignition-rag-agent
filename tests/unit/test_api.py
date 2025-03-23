#!/usr/bin/env python3
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path to import api
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import after path setup
import api
from api import app

# Test client
client = TestClient(app)


@pytest.fixture
def mock_collection():
    """Mock collection for testing."""
    mock = MagicMock()
    mock.count.return_value = 2
    return mock


# Apply mock patches at module level for all tests
@pytest.fixture(autouse=True, scope="module")
def mock_dependencies():
    """Mock API dependencies for all tests."""
    # Apply mock patches at module level
    with patch.object(api, "MOCK_EMBEDDINGS", True), patch.object(
        api, "verify_dependencies", return_value=None
    ), patch("api.get_collection"), patch("api.mock_embedding"), patch(
        "api.openai_client"
    ):
        yield


def test_health_check(mock_collection):
    """Test the /health endpoint."""
    # Setup mock collection
    with patch("api.get_collection", return_value=mock_collection):
        # Call the API
        response = client.get("/health")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["chroma"]["connected"] is True
        assert data["chroma"]["documents"] == 2


def test_query():
    """Test the /query endpoint."""
    # Setup mock collection and response
    mock_collection = MagicMock()

    # Configure the mock collection to return test results
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{"type": "perspective"}, {"type": "tag"}]],
        "documents": [["doc1 content", "doc2 content"]],
    }

    # Call the API
    with patch("api.get_collection", return_value=mock_collection):
        response = client.post("/query", json={"query": "Test query", "top_k": 2})

        # Validate response
        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 2
        assert "metadata" in data
        assert data["metadata"]["total_chunks"] == 2

        # Check that results have the expected structure
        result = data["results"][0]
        assert "content" in result
        assert "metadata" in result
        assert "similarity" in result

        # Verify collection.query was called with the embedding
        mock_collection.query.assert_called_once()


def test_query_filter():
    """Test the /query endpoint with filters."""
    # Setup mock collection
    mock_collection = MagicMock()

    # Configure the mock collection response
    mock_collection.query.return_value = {
        "ids": [["id1"]],
        "distances": [[0.1]],
        "metadatas": [[{"type": "tag"}]],
        "documents": [["doc content"]],
    }

    # Call the API with filter
    with patch("api.get_collection", return_value=mock_collection):
        response = client.post(
            "/query",
            json={
                "query": "Test query",
                "top_k": 1,
                "filter_metadata": {"type": "tag", "filepath": {"$contains": "tags"}},
            },
        )

        # Validate response
        assert response.status_code == 200

        # Verify collection.query was called with the filter
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        assert "where" in kwargs
        assert kwargs["where"]["type"] == "tag"
        assert kwargs["where"]["filepath"]["$contains"] == "tags"


def test_agent_query():
    """Test the /agent/query endpoint."""
    # Setup mock collection
    mock_collection = MagicMock()

    # Configure the mock collection to return test results
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{"type": "perspective"}, {"type": "tag"}]],
        "documents": [["doc1 content", "doc2 content"]],
    }

    # Call the API
    with patch("api.get_collection", return_value=mock_collection):
        response = client.post(
            "/agent/query",
            json={"query": "Test query", "top_k": 2, "filter_type": "perspective"},
        )

        # Validate response
        assert response.status_code == 200
        data = response.json()

        assert "context_chunks" in data
        assert len(data["context_chunks"]) == 2
        assert "suggested_prompt" in data

        # Check that context chunks have the expected structure
        chunk = data["context_chunks"][0]
        assert "source" in chunk
        assert "content" in chunk
        assert "metadata" in chunk
        assert "similarity" in chunk


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "description" in data
    assert "version" in data
    assert "endpoints" in data


def test_stats():
    """Test the /stats endpoint."""
    # Setup mock collection
    mock_collection = MagicMock()

    # Configure mocks
    mock_collection.count.return_value = 42
    mock_collection.get.return_value = {
        "metadatas": [
            {"type": "perspective"},
            {"type": "perspective"},
            {"type": "tag"},
        ]
    }

    # Call the API
    with patch("api.get_collection", return_value=mock_collection):
        response = client.get("/stats")

        # Validate response
        assert response.status_code == 200
        data = response.json()

        assert data["total_documents"] == 42
        assert data["collection_name"] == "ignition_project"
        assert "type_distribution" in data

        # Check the type distribution
        assert data["type_distribution"]["perspective"] == 2
        assert data["type_distribution"]["tag"] == 1


if __name__ == "__main__":
    unittest.main()
