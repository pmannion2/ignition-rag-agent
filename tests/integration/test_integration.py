#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import requests

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from fastapi.testclient import TestClient

import indexer
from api import MOCK_EMBEDDINGS, app, mock_embedding


class TestIntegration(unittest.TestCase):
    """Integration tests for the Ignition RAG system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for all tests."""
        # Create a temporary directory for test data and index
        cls.test_dir = tempfile.mkdtemp()
        cls.index_dir = os.path.join(cls.test_dir, "chroma_index")
        cls.project_dir = os.path.join(cls.test_dir, "ignition_project")
        cls.views_dir = os.path.join(cls.project_dir, "views")
        cls.tags_dir = os.path.join(cls.project_dir, "tags")

        # Create directories
        os.makedirs(cls.index_dir, exist_ok=True)
        os.makedirs(cls.views_dir, exist_ok=True)
        os.makedirs(cls.tags_dir, exist_ok=True)

        # Get sample data
        cls.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

        # Copy sample files to test project
        shutil.copy(
            os.path.join(cls.data_dir, "sample_view.json"),
            os.path.join(cls.views_dir, "tank_view.json"),
        )
        shutil.copy(
            os.path.join(cls.data_dir, "sample_tags.json"),
            os.path.join(cls.tags_dir, "tank_tags.json"),
        )

        # Mock environment variables for testing
        os.environ["MOCK_EMBEDDINGS"] = "true"
        os.environ["CHROMA_DB_PATH"] = cls.index_dir

        # Create a test client for the API
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        shutil.rmtree(cls.test_dir)

        # Remove environment variables
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "CHROMA_DB_PATH" in os.environ:
            del os.environ["CHROMA_DB_PATH"]

    @patch("indexer.mock_embedding", return_value=[0.1] * 1536)
    @patch("api.mock_embedding", return_value=[0.1] * 1536)
    def test_full_pipeline(self, mock_api_embedding, mock_indexer_embedding):
        """Test the full indexing and querying pipeline."""
        # Step 1: Run the indexer on the test project
        with patch.object(indexer, "PERSIST_DIRECTORY", self.index_dir):
            client = indexer.setup_chroma_client()
            collection = indexer.get_collection(client, rebuild=True)

            # Find JSON files
            json_files = indexer.find_json_files(self.project_dir)
            self.assertEqual(len(json_files), 2)

            # Load and chunk the files
            documents = indexer.load_json_files(json_files)
            chunks = indexer.create_chunks(documents)

            # Index the chunks
            indexer.index_documents(chunks, collection, rebuild=True)

            # Verify chunks were indexed
            self.assertGreater(collection.count(), 0)

        # Step 2: Test the API with indexed data
        # We'll bypass the actual HTTP server and use the TestClient directly

        # Test query endpoint
        response = self.client.post("/query", json={"query": "Tank Level", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        query_data = response.json()
        self.assertIn("results", query_data)

        # Test agent query endpoint
        response = self.client.post(
            "/agent/query",
            json={"query": "How is the Tank Level configured?", "top_k": 2},
        )
        self.assertEqual(response.status_code, 200)
        agent_data = response.json()
        self.assertIn("context_chunks", agent_data)
        self.assertIn("suggested_prompt", agent_data)

        # Test stats endpoint
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        stats_data = response.json()
        self.assertIn("total_documents", stats_data)
        self.assertIn("type_distribution", stats_data)

    @patch("indexer.mock_embedding", return_value=[0.1] * 1536)
    def test_incremental_indexing(self, mock_embedding_fn):
        """Test incremental indexing when a file changes."""
        # First indexing run
        with patch.object(indexer, "PERSIST_DIRECTORY", self.index_dir):
            client = indexer.setup_chroma_client()
            collection = indexer.get_collection(client, rebuild=True)

            # Index initial files
            json_files = indexer.find_json_files(self.project_dir)
            documents = indexer.load_json_files(json_files)
            chunks = indexer.create_chunks(documents)
            indexer.index_documents(chunks, collection, rebuild=True)

            # Record the initial document count
            initial_count = collection.count()

            # Create a completely new tag file instead of modifying an existing one
            new_tags_path = os.path.join(self.tags_dir, "new_tags.json")
            new_tags_data = [
                {
                    "name": "Tank2/Level",
                    "tagType": "AtomicTag",
                    "dataType": "Float8",
                    "value": 55.5,
                    "path": "Tanks/Tank2/Level",
                    "parameters": {
                        "engHigh": 100,
                        "engLow": 0,
                        "engUnit": "%",
                        "description": "Current fill level of Tank 2",
                    },
                },
                {
                    "name": "Tank2/Pressure",
                    "tagType": "AtomicTag",
                    "dataType": "Float8",
                    "value": 1.75,
                    "path": "Tanks/Tank2/Pressure",
                    "parameters": {
                        "engHigh": 5,
                        "engLow": 0,
                        "engUnit": "bar",
                        "description": "Current pressure in Tank 2",
                    },
                },
            ]

            # Write the new file
            with open(new_tags_path, "w") as f:
                json.dump(new_tags_data, f)

            # Run incremental indexing on the new file
            documents = indexer.load_json_files([new_tags_path])
            chunks = indexer.create_chunks(documents)
            indexer.index_documents(chunks, collection, rebuild=False)

            # Verify the document count increased
            final_count = collection.count()
            self.assertGreater(final_count, initial_count)

    @patch("api.mock_embedding", return_value=[0.1] * 1536)
    def test_cursor_agent_integration(self, mock_embedding_fn):
        """Test the Cursor agent integration."""
        # Import cursor_agent here to avoid affecting other tests
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        import cursor_agent

        # Ensure mock embeddings are enabled
        cursor_agent.USE_MOCK_EMBEDDINGS = True

        # Test the get_cursor_context function
        context = cursor_agent.get_cursor_context(
            "How is the Tank Level configured?",
            cursor_context={"current_file": "views/tank_view.json"},
            top_k=1,
        )

        # Verify we got a suggested prompt with mock data
        self.assertTrue("mock" in context.lower())

        # Test the get_ignition_tag_info function
        tag_info = cursor_agent.get_ignition_tag_info("Tank1/Level")

        # Verify we got mock tag data
        self.assertEqual(tag_info["name"], "Tank1/Level")
        self.assertEqual(tag_info["value"], 42.0)
        self.assertTrue(tag_info["mock_used"])

        # Test the get_ignition_view_component function
        view_info = cursor_agent.get_ignition_view_component("Main")

        # Verify we got mock view data
        self.assertEqual(view_info["name"], "Main")
        self.assertTrue(view_info["mock_used"])


if __name__ == "__main__":
    unittest.main()
