#!/usr/bin/env python3
import json
import os
import shutil
import sys
import tempfile
import unittest

import pytest

# Add parent directory to path to import indexer
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from indexer import (
    chunk_perspective_view,
    chunk_tag_config,
    create_chunks,
    find_json_files,
    load_json_files,
    process_component,
)


class TestIndexer(unittest.TestCase):
    """Test cases for the indexer functions."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

        # Load sample data
        with open(os.path.join(self.data_dir, "sample_view.json")) as f:
            self.sample_view = json.load(f)

        with open(os.path.join(self.data_dir, "sample_tags.json")) as f:
            self.sample_tags = json.load(f)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_find_json_files(self):
        """Test finding JSON files."""
        # Create some test files
        views_dir = os.path.join(self.test_dir, "views")
        tags_dir = os.path.join(self.test_dir, "tags")
        other_dir = os.path.join(self.test_dir, "other")

        os.makedirs(views_dir)
        os.makedirs(tags_dir)
        os.makedirs(other_dir)

        # Create test files
        view_file = os.path.join(views_dir, "test_view.json")
        tag_file = os.path.join(tags_dir, "test_tag.json")
        other_file = os.path.join(other_dir, "other.json")

        with open(view_file, "w") as f:
            f.write("{}")
        with open(tag_file, "w") as f:
            f.write("{}")
        with open(other_file, "w") as f:
            f.write("{}")

        # Test the function
        json_files = find_json_files(self.test_dir)

        # Should find the view and tag JSON files
        assert len(json_files) == 2
        assert view_file in json_files
        assert tag_file in json_files
        assert other_file not in json_files  # Should not include other JSON files

    def test_load_json_files(self):
        """Test loading JSON files with metadata."""
        # Create test files
        views_dir = os.path.join(self.test_dir, "views")
        tags_dir = os.path.join(self.test_dir, "tags")

        os.makedirs(views_dir)
        os.makedirs(tags_dir)

        view_file = os.path.join(views_dir, "test_view.json")
        tag_file = os.path.join(tags_dir, "test_tag.json")

        with open(view_file, "w") as f:
            json.dump({"test": "view"}, f)
        with open(tag_file, "w") as f:
            json.dump({"test": "tag"}, f)

        # Test the function
        documents = load_json_files([view_file, tag_file])

        # Should have loaded both files with correct metadata
        assert len(documents) == 2

        view_doc = next(
            (doc for doc in documents if doc["metadata"]["filepath"] == view_file), None
        )
        tag_doc = next(
            (doc for doc in documents if doc["metadata"]["filepath"] == tag_file), None
        )

        assert view_doc is not None
        assert tag_doc is not None

        assert view_doc["metadata"]["type"] == "perspective"
        assert tag_doc["metadata"]["type"] == "tag"

        assert view_doc["content"]["test"] == "view"
        assert tag_doc["content"]["test"] == "tag"

    def test_chunk_perspective_view(self):
        """Test chunking a Perspective view."""
        view_meta = {
            "filepath": "views/test_view.json",
            "type": "perspective",
            "name": "test_view",
        }

        # Make sure the sample_view has the correct structure
        if "root" in self.sample_view and "params" in self.sample_view["root"]:
            # Structure is already correct
            pass
        else:
            # Add parameters to the root for the test
            self.sample_view = {
                "root": {
                    **self.sample_view,
                    "params": {
                        "title": "Test View",
                        "description": "This is a test view",
                    },
                }
            }

        chunks = chunk_perspective_view(self.sample_view, view_meta)

        # Should have created chunks for the view
        assert len(chunks) > 0

        # Check that params are included
        params_chunk = next(
            (chunk for chunk in chunks if chunk[1].get("section") == "params"), None
        )
        assert params_chunk is not None

        # Check that a component is included
        component_chunk = next(
            (chunk for chunk in chunks if "component" in chunk[1]), None
        )
        assert component_chunk is not None

        # Verify chunk contents are JSON strings
        for chunk, _ in chunks:
            try:
                json.loads(chunk)
            except json.JSONDecodeError:
                pytest.fail("Chunk is not valid JSON")

    def test_chunk_tag_config(self):
        """Test chunking a Tag configuration."""
        tag_meta = {
            "filepath": "tags/test_tags.json",
            "type": "tag",
            "name": "test_tags",
        }

        chunks = chunk_tag_config(self.sample_tags, tag_meta)

        # Should have created chunks for the tags
        assert len(chunks) > 0

        # Verify chunk contents
        for chunk, metadata in chunks:
            try:
                parsed = json.loads(chunk)
                # Each chunk should be a list of tags or a single tag
                if isinstance(parsed, list):
                    for tag in parsed:
                        assert "name" in tag
                else:
                    assert "name" in parsed
            except json.JSONDecodeError:
                pytest.fail("Chunk is not valid JSON")

            # Check that folder information is included
            assert "folder" in metadata

    def test_create_chunks(self):
        """Test the create_chunks function that handles different document types."""
        documents = [
            {
                "content": self.sample_view,
                "metadata": {
                    "filepath": "views/test_view.json",
                    "type": "perspective",
                    "name": "test_view",
                },
            },
            {
                "content": self.sample_tags,
                "metadata": {
                    "filepath": "tags/test_tags.json",
                    "type": "tag",
                    "name": "test_tags",
                },
            },
        ]

        all_chunks = create_chunks(documents)

        # Should have created chunks for both documents
        assert len(all_chunks) > 0

        # Check that we have both types of chunks
        perspective_chunks = [
            chunk for chunk in all_chunks if chunk[1]["type"] == "perspective"
        ]
        tag_chunks = [chunk for chunk in all_chunks if chunk[1]["type"] == "tag"]

        assert len(perspective_chunks) > 0
        assert len(tag_chunks) > 0

    def test_process_component(self):
        """Test processing a component and its children."""
        # Use the MainContainer component from sample view
        component = self.sample_view["root"]["children"][0]
        view_meta = {
            "filepath": "views/test_view.json",
            "type": "perspective",
            "name": "test_view",
        }
        chunks = []

        process_component(component, view_meta, chunks)

        # Should have created chunks for the component and its children
        assert len(chunks) > 0

        # Check that component metadata is correct
        for _, metadata in chunks:
            assert metadata["type"] == "perspective"
            assert "component" in metadata
            assert "path" in metadata


if __name__ == "__main__":
    unittest.main()
