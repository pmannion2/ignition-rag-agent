#!/usr/bin/env python3
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Add parent directory to path to import api
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import after path setup
import api
from api import app, get_collection, mock_embedding


class TestAPI(unittest.TestCase):
    """Test cases for the API endpoints."""

    mock_patches = []

    @classmethod
    def setUpClass(cls):
        """Set up test class with patches."""
        # Apply mock patches at class level
        cls.mock_embeddings_patch = patch.object(api, "MOCK_EMBEDDINGS", True)
        cls.mock_embeddings_patch.start()
        cls.mock_patches.append(cls.mock_embeddings_patch)

        # Directly patch the verify_dependencies function to do nothing
        # This ensures tests aren't affected by dependency checks
        async def mock_verify_dependencies():
            pass

        cls.verify_dependencies_patch = patch.object(
            api, "verify_dependencies", mock_verify_dependencies
        )
        cls.verify_dependencies_patch.start()
        cls.mock_patches.append(cls.verify_dependencies_patch)

    @classmethod
    def tearDownClass(cls):
        """Clean up patches."""
        # Stop all patches
        for mock_patch in cls.mock_patches:
            mock_patch.stop()

    def setUp(self):
        """Set up test environment."""
        self.client = TestClient(app)

        # Mock collection.query response
        self.mock_query_response = {
            "ids": [["sample-1", "sample-2"]],
            "documents": [
                [
                    '{"name": "Tank1/Level", "value": 75.5}',
                    '{"type": "gauge", "name": "TankLevelGauge"}',
                ]
            ],
            "metadatas": [
                [
                    {
                        "type": "tag",
                        "filepath": "tags/sample_tags.json",
                        "folder": "Tanks",
                    },
                    {
                        "type": "perspective",
                        "filepath": "views/sample_view.json",
                        "component": "TankLevelGauge",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

    @patch("api.get_collection")
    @patch("api.mock_embedding", return_value=[0.1] * 1536)
    def test_query_endpoint(self, mock_embedding_fn, mock_get_collection):
        """Test the /query endpoint."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = self.mock_query_response
        mock_collection.count.return_value = 2
        # Setup the get_collection function to return our mock
        mock_get_collection.return_value = mock_collection

        # Test the endpoint
        response = self.client.post(
            "/query", json={"query": "Tank Level", "top_k": 2, "use_mock": True}
        )

        # Validate response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["total_chunks"], 2)

        # Check that results have the expected structure
        result = data["results"][0]
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("similarity", result)

        # Verify collection.query was called with the embedding
        mock_collection.query.assert_called_once()

    @patch("api.get_collection")
    @patch("api.mock_embedding", return_value=[0.1] * 1536)
    def test_query_with_filter(self, mock_embedding_fn, mock_get_collection):
        """Test the /query endpoint with filters."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = self.mock_query_response
        mock_collection.count.return_value = 2
        # Setup the get_collection function to return our mock
        mock_get_collection.return_value = mock_collection

        # Test the endpoint with type filter
        response = self.client.post(
            "/query",
            json={
                "query": "Tank Level",
                "top_k": 2,
                "filter_metadata": {"type": "tag", "filepath": {"$contains": "tags"}},
                "use_mock": True,
            },
        )

        # Validate response
        self.assertEqual(response.status_code, 200)

        # Verify collection.query was called with the filter
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        self.assertIn("where", kwargs)
        self.assertEqual(kwargs["where"]["type"], "tag")
        self.assertEqual(kwargs["where"]["filepath"]["$contains"], "tags")

    @patch("api.get_collection")
    @patch("api.mock_embedding", return_value=[0.1] * 1536)
    def test_agent_query_endpoint(self, mock_embedding_fn, mock_get_collection):
        """Test the /agent/query endpoint."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = self.mock_query_response
        # Setup the get_collection function to return our mock
        mock_get_collection.return_value = mock_collection

        # Test the endpoint
        response = self.client.post(
            "/agent/query",
            json={
                "query": "Tank Level",
                "top_k": 2,
                "context": {"current_file": "views/tank_view.json"},
                "use_mock": True,
            },
        )

        # Validate response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("context_chunks", data)
        self.assertEqual(len(data["context_chunks"]), 2)
        self.assertIn("suggested_prompt", data)

        # Check that context chunks have the expected structure
        chunk = data["context_chunks"][0]
        self.assertIn("source", chunk)
        self.assertIn("content", chunk)
        self.assertIn("metadata", chunk)
        self.assertIn("similarity", chunk)

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("name", data)
        self.assertIn("description", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)

    @patch("api.get_collection")
    def test_stats_endpoint(self, mock_get_collection):
        """Test the /stats endpoint."""
        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.get.return_value = {
            "metadatas": [
                {"type": "perspective"},
                {"type": "perspective"},
                {"type": "tag"},
            ]
        }
        # Setup the get_collection function to return our mock
        mock_get_collection.return_value = mock_collection

        # Test the endpoint
        response = self.client.get("/stats")

        # Validate response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_documents"], 42)
        self.assertEqual(data["collection_name"], "ignition_project")
        self.assertIn("type_distribution", data)

        # Check the type distribution
        self.assertEqual(data["type_distribution"]["perspective"], 2)
        self.assertEqual(data["type_distribution"]["tag"], 1)


if __name__ == "__main__":
    unittest.main()
