#!/usr/bin/env python3
"""
Sample Indexer for Ignition RAG

This script adds some sample data to the Chroma database for testing.
Run it after starting the Chroma service to have some data to query.
"""

import hashlib
import os
import sys
from datetime import datetime

# Use the Chroma client directly
import chromadb
import numpy as np
import requests
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Mock embedding function to create sample embeddings
def mock_embedding(text):
    """Create deterministic mock embeddings"""
    # Generate a deterministic hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Use the hash to seed a random generator for deterministic embeddings
    seed = int(text_hash, 16) % (2**32 - 1)
    rng = np.random.RandomState(seed)

    # Generate a random vector of length 1536 (same as text-embedding-ada-002)
    embedding = rng.rand(1536).astype(np.float32)

    # Normalize to unit length for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# Sample content for Ignition tags
SAMPLE_TAG_DATA = [
    {
        "name": "Tank1Level",
        "type": "tag",
        "content": """
        {
            "name": "Tank1Level",
            "tagType": "AtomicTag",
            "dataType": "Float8",
            "value": 75.5,
            "alarms": [
                {"name": "High", "setpoint": 90.0, "mode": "GreaterThan"},
                {"name": "Low", "setpoint": 10.0, "mode": "LessThan"}
            ],
            "engineering_units": "gallons",
            "description": "Level sensor for main storage tank 1"
        }
        """,
        "metadata": {
            "filepath": "tags/Tanks/Tank1Level.json",
            "type": "tag",
            "folder": "Tanks",
            "created": "2023-01-15",
            "modified": "2023-02-20",
        },
    },
    {
        "name": "Pump1Status",
        "type": "tag",
        "content": """
        {
            "name": "Pump1Status",
            "tagType": "AtomicTag",
            "dataType": "Boolean",
            "value": true,
            "alarms": [
                {"name": "Offline", "mode": "StateChange"}
            ],
            "description": "Status indicator for main pump 1"
        }
        """,
        "metadata": {
            "filepath": "tags/Pumps/Pump1Status.json",
            "type": "tag",
            "folder": "Pumps",
            "created": "2023-01-10",
            "modified": "2023-03-05",
        },
    },
]

# Sample content for Ignition perspective views
SAMPLE_PERSPECTIVE_DATA = [
    {
        "name": "TankView",
        "type": "perspective",
        "content": """
        {
            "name": "TankView",
            "type": "view",
            "root": {
                "type": "flex-container",
                "children": [
                    {
                        "type": "tank-level-indicator",
                        "props": {
                            "tankTag": "Tank1Level",
                            "height": 300,
                            "width": 150,
                            "showLabels": true
                        }
                    },
                    {
                        "type": "numeric-display",
                        "props": {
                            "value": "{Tank1Level}",
                            "units": "gallons",
                            "decimalPlaces": 1
                        }
                    }
                ]
            }
        }
        """,
        "metadata": {
            "filepath": "perspective/views/TankView.json",
            "type": "perspective",
            "component": "view",
            "created": "2023-02-01",
            "modified": "2023-03-10",
        },
    },
    {
        "name": "PumpControl",
        "type": "perspective",
        "content": """
        {
            "name": "PumpControl",
            "type": "view",
            "root": {
                "type": "flex-container",
                "children": [
                    {
                        "type": "button",
                        "props": {
                            "text": "Start Pump",
                            "action": {
                                "actionType": "script",
                                "script": "system.tag.write('Pump1Status', true)"
                            }
                        }
                    },
                    {
                        "type": "button",
                        "props": {
                            "text": "Stop Pump",
                            "action": {
                                "actionType": "script",
                                "script": "system.tag.write('Pump1Status', false)"
                            }
                        }
                    },
                    {
                        "type": "status-indicator",
                        "props": {
                            "status": "{Pump1Status}",
                            "onLabel": "Running",
                            "offLabel": "Stopped"
                        }
                    }
                ]
            }
        }
        """,
        "metadata": {
            "filepath": "perspective/views/PumpControl.json",
            "type": "perspective",
            "component": "view",
            "created": "2023-02-05",
            "modified": "2023-03-15",
        },
    },
]


def check_chroma_status(host, port):
    """Check if Chroma is up and retrieve server configuration info."""
    try:
        # Check heartbeat
        heartbeat_url = f"http://{host}:{port}/api/v1/heartbeat"
        heartbeat_response = requests.get(heartbeat_url)
        heartbeat_data = heartbeat_response.json()
        print(f"Chroma heartbeat: {heartbeat_data}")

        # List available tenants (this endpoint might not exist in all versions)
        try:
            tenants_url = f"http://{host}:{port}/api/v1/tenants"
            tenants_response = requests.get(tenants_url)
            if tenants_response.status_code == 200:
                tenants_data = tenants_response.json()
                print(f"Available tenants: {tenants_data}")
            else:
                print(f"Could not retrieve tenants: {tenants_response.status_code}")
        except Exception as e:
            print(f"Error checking tenants: {e}")

        # Try to create default tenant for backward compatibility
        try:
            create_tenant_url = f"http://{host}:{port}/api/v1/tenants"
            create_tenant_data = {"name": "default_tenant"}
            create_response = requests.post(create_tenant_url, json=create_tenant_data)
            print(
                f"Create tenant response: {create_response.status_code} - {create_response.text}"
            )
        except Exception as e:
            print(f"Error creating tenant: {e}")

        return True
    except Exception as e:
        print(f"Error checking Chroma status: {e}")
        return False


def main():
    """Main function to populate the database"""
    print("Starting sample data indexing...")

    # Connect to Chroma
    try:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        print(f"Connecting to Chroma at {chroma_host}:{chroma_port}...")

        # Check basic connectivity
        try:
            heartbeat_url = f"http://{chroma_host}:{chroma_port}/api/v1/heartbeat"
            heartbeat_response = requests.get(heartbeat_url)
            print(f"Chroma is responding with heartbeat: {heartbeat_response.json()}")
        except Exception as e:
            print(f"Couldn't connect to Chroma at {chroma_host}:{chroma_port}: {e}")
            return 1

        print("Attempting to connect to the Chroma client...")

        # Attempt to create tenant first
        try:
            create_tenant_url = f"http://{chroma_host}:{chroma_port}/api/v1/tenants"
            create_tenant_data = {"name": "default_tenant"}
            requests.post(create_tenant_url, json=create_tenant_data)
            print("Created or verified default_tenant")
        except Exception as e:
            print(f"Note: Couldn't create tenant (may already exist): {e}")

        # Connect to latest Chroma version with tenant
        client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            tenant="default_tenant",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Test if connection works
        heartbeat = client.heartbeat()
        print(f"Connected to Chroma. Heartbeat: {heartbeat}")

        print("Connected to Chroma successfully.")

        # Create or get collection
        collection_name = os.getenv("COLLECTION_NAME", "ignition_project")

        try:
            print(f"Getting collection '{collection_name}'...")
            collection = client.get_collection(collection_name)
            print(f"Found existing collection with {collection.count()} documents.")
        except Exception as e:
            print(f"Creating new collection '{collection_name}'...")
            print(f"Error was: {e}")
            collection = client.create_collection(collection_name)

        # Combine sample data
        sample_data = SAMPLE_TAG_DATA + SAMPLE_PERSPECTIVE_DATA

        # Add sample data to collection
        docs = []
        ids = []
        embeddings = []
        metadatas = []

        for item in sample_data:
            doc_id = f"{item['type']}_{item['name']}_{datetime.now().timestamp()}"
            doc_content = item["content"].strip()
            doc_embedding = mock_embedding(doc_content)

            docs.append(doc_content)
            ids.append(doc_id)
            embeddings.append(doc_embedding)
            metadatas.append(item["metadata"])

        # Add to collection
        print(f"Adding {len(docs)} documents to collection...")
        collection.add(
            documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        # Verify documents were added
        count = collection.count()
        print(f"Collection now has {count} documents.")

        print("Sample data indexed successfully!")
        return 0

    except Exception as e:
        print(f"Error indexing sample data: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
