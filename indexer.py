#!/usr/bin/env python3
import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import tiktoken
import typer
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Check if we're in mock mode (for testing without OpenAI API key)
MOCK_EMBEDDINGS = os.getenv("MOCK_EMBEDDINGS", "false").lower() == "true"
if MOCK_EMBEDDINGS:
    print("Using mock embeddings for testing")

# Initialize OpenAI client only if we're not in mock mode or we have a key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not MOCK_EMBEDDINGS or openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        print("Initialized OpenAI client")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        if not MOCK_EMBEDDINGS:
            print(
                "No valid OpenAI API key and mock mode is not enabled, some features may not work"
            )
else:
    client = None
    print("OpenAI client not initialized (using mock mode)")

# Initialize tokenizer for GPT models
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 6000  # Increased from 400 to reduce API calls while staying under limits
HARD_TOKEN_LIMIT = 7500  # Hard limit for safety

# Initialize Chroma client
PERSIST_DIRECTORY = "chroma_index"
COLLECTION_NAME = "ignition_project"
LAST_INDEX_TIME_FILE = "last_index_time.pkl"

# Check if running in Docker with external Chroma
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
USE_PERSISTENT_CHROMA = os.getenv("USE_PERSISTENT_CHROMA", "false").lower() == "true"

app = typer.Typer()


def setup_chroma_client():
    """Set up and return a Chroma client with persistence."""
    # For tests, use in-memory client if specified
    if os.getenv("USE_IN_MEMORY_CHROMA", "false").lower() == "true":
        print("Using in-memory Chroma client for testing")
        return chromadb.Client()

    # For external Chroma connections
    if CHROMA_HOST and CHROMA_PORT:
        print(f"Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
        if USE_PERSISTENT_CHROMA:
            # For persistent HTTP client mode (used by run_local.sh)
            print("Using persistent HTTP client mode")
            return chromadb.HttpClient(
                host=CHROMA_HOST,
                port=int(CHROMA_PORT),
                tenant="default_tenant",
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        else:
            # Standard HTTP client
            return chromadb.HttpClient(
                host=CHROMA_HOST,
                port=int(CHROMA_PORT),
            )
    else:
        # Use local persistent Chroma
        print(f"Using local Chroma with persistence at {PERSIST_DIRECTORY}")
        return chromadb.PersistentClient(path=PERSIST_DIRECTORY)


def get_collection(client, rebuild=False):
    """Get or create a collection for the Ignition project."""
    if (
        rebuild
        and client.list_collections()
        and any(c.name == COLLECTION_NAME for c in client.list_collections())
    ):
        client.delete_collection(COLLECTION_NAME)

    return client.get_or_create_collection(name=COLLECTION_NAME)


def mock_embedding(text: str) -> List[float]:
    """Create a deterministic mock embedding based on the text content hash.

    This is used for testing without a valid OpenAI API key.
    """
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


def find_json_files(project_dir: str) -> List[str]:
    """Find all JSON files in the project directory that are likely Perspective views or Tag configurations."""
    json_files = []
    for root, _, files in os.walk(project_dir):
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                # Only include files that are likely Perspective views or Tag configurations
                # This is a simple heuristic and may need to be adjusted based on your project structure
                if "views" in fpath or "tag" in fpath.lower():
                    json_files.append(fpath)
    return json_files


def load_json_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load JSON content from each file with metadata."""
    documents = []
    for fpath in file_paths:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            # Determine metadata based on file path and content
            meta = {"filepath": fpath}
            if "views" in fpath:
                meta["type"] = "perspective"
                meta["name"] = os.path.splitext(os.path.basename(fpath))[0]
            elif "tag" in fpath.lower():
                meta["type"] = "tag"
                # Try to extract tag provider or folder from path
                parts = fpath.split(os.sep)
                if len(parts) > 1:
                    meta["folder"] = parts[-2]

            documents.append({"content": data, "metadata": meta})
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {fpath}")
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return documents


def chunk_perspective_view(view_json: Dict[str, Any], view_meta: Dict[str, str]) -> List[tuple]:
    """Split a Perspective view JSON into semantically meaningful chunks with context preservation."""
    chunks = []
    component_map = {}  # Store component references for cross-linking

    # Extract basic view information for context
    view_context = {
        "name": view_meta.get("name", ""),
        "filepath": view_meta.get("filepath", ""),
        "type": "perspective_view",
    }

    # Extract root component and children for a view
    if "root" in view_json:
        root = view_json["root"]
        components = root.get("children", [root])
    else:
        components = [view_json]

    # First pass - build component map
    component_map = {}

    # Process each component to build the map
    for component_id, comp in enumerate(components):
        build_component_map(comp, component_map, component_id, "root")

    # Second pass - process components with context preservation
    for comp in components:
        process_component_with_context(comp, view_meta, chunks, component_map, view_context)

    return chunks


def build_component_map(comp, component_map, comp_id, parent_path):
    """Build a map of component IDs to their metadata for context linking."""
    if not isinstance(comp, dict):
        return

    comp_name = comp.get("meta", {}).get("name", f"unnamed_{comp_id}")
    current_path = f"{parent_path}/{comp_name}" if parent_path else comp_name

    component_map[current_path] = {
        "id": comp_id,
        "name": comp_name,
        "parent": parent_path,
        "children": [],
    }

    # Process children
    if "children" in comp and isinstance(comp["children"], list):
        for child_id, child in enumerate(comp["children"]):
            child_path = f"{current_path}/child_{child_id}"
            component_map[current_path]["children"].append(child_path)
            build_component_map(child, component_map, child_id, current_path)


def process_component_with_context(
    comp, view_meta, chunks, component_map, view_context, parent_path=""
):
    """Process a component and its children recursively while preserving context."""
    if not isinstance(comp, dict):
        return

    comp_name = comp.get("meta", {}).get("name", "unnamed")
    current_path = f"{parent_path}/{comp_name}" if parent_path else comp_name

    # Create context information for this component
    component_context = {
        "view": view_context,
        "component_path": current_path,
        "parent_path": parent_path,
    }

    # Add child references if any
    if current_path in component_map and component_map[current_path]["children"]:
        component_context["children"] = component_map[current_path]["children"]

    # Add parent relationship
    if parent_path and parent_path in component_map:
        component_context["parent"] = {
            "path": parent_path,
            "name": component_map[parent_path].get("name", ""),
        }

    # Create a copy with metadata for this component
    comp_meta = {
        **view_meta,
        "component": comp_name,
        "path": current_path,
        "context": json.dumps(component_context),
    }

    # Convert to string for token counting
    comp_json_str = json.dumps(comp, ensure_ascii=False)
    tokens = len(enc.encode(comp_json_str))

    if tokens <= MAX_TOKENS:
        # If small enough, add as one chunk
        chunks.append((comp_json_str, comp_meta))
    else:
        # If too large, split it while preserving context
        if "children" in comp and isinstance(comp["children"], list):
            # Process children separately with context
            for child in comp["children"]:
                process_component_with_context(
                    child, view_meta, chunks, component_map, view_context, current_path
                )

            # Also add the component without its children but with references
            comp_copy = {k: v for k, v in comp.items() if k != "children"}
            comp_copy["_childrenRefs"] = [
                c.get("meta", {}).get("name", "unnamed") for c in comp.get("children", [])
            ]
            comp_without_children = json.dumps(comp_copy, ensure_ascii=False)

            if len(enc.encode(comp_without_children)) <= MAX_TOKENS:
                chunks.append(
                    (
                        comp_without_children,
                        {**comp_meta, "section": "properties_with_child_refs"},
                    )
                )
        else:
            # For large components with no children, split by logical sections
            sections = {}

            # Group by logical sections
            if "meta" in comp:
                sections["meta"] = {"meta": comp["meta"]}

            if "props" in comp:
                sections["props"] = {"props": comp["props"]}

            if "position" in comp:
                if "style" in comp:
                    sections["layout"] = {
                        "position": comp["position"],
                        "style": comp["style"],
                    }
                else:
                    sections["layout"] = {"position": comp["position"]}

            if "events" in comp:
                sections["events"] = {"events": comp["events"]}

            if "custom" in comp:
                sections["custom"] = {"custom": comp["custom"]}

            if "propConfig" in comp:
                # Split propConfig into smaller chunks if needed
                prop_config = comp["propConfig"]
                prop_items = list(prop_config.items())

                # Use multiple chunks if propConfig is large
                if len(json.dumps(prop_config)) > 3000:  # Approximate character limit
                    chunk_size = len(prop_items) // 2
                    for i in range(0, len(prop_items), chunk_size):
                        end_idx = min(i + chunk_size, len(prop_items))
                        section_name = f"propConfig_part{i // chunk_size + 1}"
                        sections[section_name] = {"propConfig": dict(prop_items[i:end_idx])}
                else:
                    sections["propConfig"] = {"propConfig": prop_config}

            # Add rest of properties if any important ones were missed
            remaining_props = {
                k: v
                for k, v in comp.items()
                if k
                not in [
                    "meta",
                    "props",
                    "position",
                    "style",
                    "events",
                    "custom",
                    "propConfig",
                    "children",
                ]
            }
            if remaining_props:
                sections["otherProps"] = remaining_props

            # Create chunks for each logical section with context
            for section_name, section_content in sections.items():
                section_str = json.dumps(section_content, ensure_ascii=False)
                if len(enc.encode(section_str)) <= MAX_TOKENS:
                    section_meta = {
                        **comp_meta,
                        "section": section_name,
                        "full_component": False,
                    }
                    chunks.append((section_str, section_meta))
                else:
                    # If still too large, fall back to character-based chunking for this section
                    text_chunks = chunk_by_characters(section_str, int(MAX_TOKENS / 1.2))
                    for i, chunk in enumerate(text_chunks):
                        chunk_meta = {
                            **comp_meta,
                            "section": f"{section_name}_part{i + 1}",
                            "full_component": False,
                            "total_parts": len(text_chunks),
                        }
                        chunks.append((chunk, chunk_meta))


def chunk_tag_config(tag_json: Any, tag_meta: Dict[str, str], max_depth=3) -> List[tuple]:
    """Split a Tag JSON into semantically meaningful chunks based on tag hierarchy.

    Args:
        tag_json: The tag JSON data to chunk
        tag_meta: Metadata about the tags
        max_depth: Maximum recursion depth to prevent infinite recursion
    """
    chunks = []

    # Extract the list of tags based on the structure
    tags_list = []
    if isinstance(tag_json, dict) and "tags" in tag_json:
        tags_list = tag_json["tags"]
    elif isinstance(tag_json, list):
        tags_list = tag_json
    else:
        tags_list = [tag_json]  # Treat as a single tag

    # Special case for extremely large tag files - pre-split if needed
    all_tags_str = json.dumps(tags_list, ensure_ascii=False)
    total_tokens = len(enc.encode(all_tags_str))

    # If the entire tag list is extremely large and we haven't exceeded max recursion depth
    if total_tokens > MAX_TOKENS * 5 and max_depth > 0:  # Reduced threshold and added depth check
        print(
            f"Large tag file detected ({total_tokens} tokens). Pre-chunking into {(total_tokens // MAX_TOKENS) + 1} blocks..."
        )

        # For very large files with many tags, split by index
        if len(tags_list) > 100:  # Only split large lists
            chunk_size = max(1, len(tags_list) // ((total_tokens // MAX_TOKENS) + 1))
            for i in range(0, len(tags_list), chunk_size):
                end_idx = min(i + chunk_size, len(tags_list))
                sub_tags = tags_list[i:end_idx]
                sub_meta = {
                    **tag_meta,
                    "sub_section": f"block_{i // chunk_size + 1}_of_{(len(tags_list) + chunk_size - 1) // chunk_size}",
                }
                # Recursively process with reduced depth
                sub_chunks = chunk_tag_config(sub_tags, sub_meta, max_depth - 1)
                chunks.extend(sub_chunks)
            return chunks
        else:
            # For files with few but very large tags, we'll continue with normal processing
            # but will use character chunking as fallback for individual large tags
            print(
                "File contains few but very large tags. Using direct processing with fallback chunking."
            )

    # Prepare for hierarchy analysis
    tag_map = {}  # All tags indexed by path
    udt_instances = {}  # UDT instance tags
    udt_definitions = {}  # UDT definition tags
    tag_hierarchies = {}  # Tags organized by hierarchy
    atomic_tags = []  # Tags without hierarchy relationships

    # First pass - categorize tags and build relationships
    for tag in tags_list:
        if not isinstance(tag, dict):
            atomic_tags.append(tag)
            continue

        # Get tag path and type information
        tag_path = tag.get("path", "")
        tag_type = tag.get("typeId", "")
        tag_map[tag_path] = tag

        if "parameters" in tag or "udt" in tag_type.lower():
            # This is likely a UDT instance
            udt_type = tag.get("typeId", "").split(":")[-1] if ":" in tag.get("typeId", "") else ""
            if udt_type:
                if udt_type not in udt_instances:
                    udt_instances[udt_type] = []
                udt_instances[udt_type].append(tag)
        elif "definition" in tag:
            # This is likely a UDT definition
            udt_name = tag.get("name", "")
            if udt_name:
                udt_definitions[udt_name] = tag
        else:
            # Extract parent path to build hierarchy
            parent_path = "/".join(tag_path.split("/")[:-1]) if "/" in tag_path else ""
            if parent_path:
                if parent_path not in tag_hierarchies:
                    tag_hierarchies[parent_path] = []
                tag_hierarchies[parent_path].append(tag)
            else:
                atomic_tags.append(tag)

    # Process UDT definitions with their instances
    for udt_name, udt_def in udt_definitions.items():
        udt_context = {
            "type": "udt_definition",
            "name": udt_name,
            "instances": [
                {"path": instance.get("path", ""), "name": instance.get("name", "")}
                for instance in udt_instances.get(udt_name, [])
            ],
        }

        # Add UDT definition with context
        udt_str = json.dumps(udt_def, ensure_ascii=False)
        if len(enc.encode(udt_str)) <= MAX_TOKENS:
            udt_meta = {
                **tag_meta,
                "udt": udt_name,
                "section": "udt_definition",
                "context": json.dumps(udt_context),
            }
            chunks.append((udt_str, udt_meta))
        else:
            # Split large UDT definitions by sections
            if "definition" in udt_def and isinstance(udt_def["definition"], dict):
                # Extract key sections of UDT definition
                udt_sections = {}
                definition = udt_def["definition"]

                if "parameters" in definition:
                    udt_sections["parameters"] = {
                        "name": udt_def.get("name", ""),
                        "parameters": definition["parameters"],
                    }

                if "tags" in definition:
                    # Split UDT member tags into manageable chunks
                    member_tags = definition["tags"]
                    if len(json.dumps(member_tags)) > 3000:
                        tag_items = list(enumerate(member_tags))
                        chunk_size = max(1, len(tag_items) // 3)

                        for i in range(0, len(tag_items), chunk_size):
                            end_idx = min(i + chunk_size, len(tag_items))
                            section_tags = [t for _, t in tag_items[i:end_idx]]
                            udt_sections[f"member_tags_part{i // chunk_size + 1}"] = {
                                "name": udt_def.get("name", ""),
                                "member_tags": section_tags,
                            }
                    else:
                        udt_sections["member_tags"] = {
                            "name": udt_def.get("name", ""),
                            "member_tags": member_tags,
                        }

                # Add other properties
                remaining_props = {
                    k: v for k, v in definition.items() if k not in ["parameters", "tags"]
                }
                if remaining_props:
                    udt_sections["other_props"] = {
                        "name": udt_def.get("name", ""),
                        "properties": remaining_props,
                    }

                # Create chunks for each section
                for section_name, section_content in udt_sections.items():
                    section_str = json.dumps(section_content, ensure_ascii=False)
                    if len(enc.encode(section_str)) <= MAX_TOKENS:
                        section_meta = {
                            **tag_meta,
                            "udt": udt_name,
                            "section": section_name,
                            "context": json.dumps(udt_context),
                        }
                        chunks.append((section_str, section_meta))
                    else:
                        # If still too large, apply character chunking
                        char_chunks = chunk_by_characters(section_str, int(MAX_TOKENS * 0.9))
                        for i, char_chunk in enumerate(char_chunks):
                            char_meta = {
                                **tag_meta,
                                "udt": udt_name,
                                "section": f"{section_name}_part{i + 1}",
                                "total_parts": len(char_chunks),
                                "context": json.dumps(udt_context),
                            }
                            chunks.append((char_chunk, char_meta))

    # Process UDT instances grouped by UDT type
    for udt_type, instances in udt_instances.items():
        # Group instances by parameter values or other similarities
        instance_groups = group_similar_udt_instances(instances)

        for group_name, group_instances in instance_groups.items():
            group_context = {
                "type": "udt_instances",
                "udt_type": udt_type,
                "group": group_name,
                "count": len(group_instances),
            }

            # Process each group
            process_tag_batch(
                group_instances,
                tag_meta,
                chunks,
                {
                    "context": json.dumps(group_context),
                    "udt_type": udt_type,
                    "group": group_name,
                },
            )

    # Process tag hierarchies (parent-child relationships)
    for parent_path, child_tags in tag_hierarchies.items():
        hierarchy_context = {
            "type": "tag_hierarchy",
            "parent_path": parent_path,
            "parent_name": (parent_path.split("/")[-1] if "/" in parent_path else parent_path),
            "child_count": len(child_tags),
        }

        # Include parent tag in context if available
        if parent_path in tag_map:
            parent_tag = tag_map[parent_path]
            parent_info = {
                "name": parent_tag.get("name", ""),
                "type": parent_tag.get("typeId", ""),
            }
            hierarchy_context["parent_info"] = parent_info

        # Process children in batches
        process_tag_batch(
            child_tags,
            tag_meta,
            chunks,
            {
                "context": json.dumps(hierarchy_context),
                "parent": parent_path,
                "section": "hierarchy",
            },
        )

    # Process atomic tags (not part of a hierarchy)
    if atomic_tags:
        # Group by tag type if possible
        grouped_atomic = {}
        for tag in atomic_tags:
            if isinstance(tag, dict):
                tag_type = tag.get("typeId", "unknown")
                if tag_type not in grouped_atomic:
                    grouped_atomic[tag_type] = []
                grouped_atomic[tag_type].append(tag)
            else:
                if "non_dict" not in grouped_atomic:
                    grouped_atomic["non_dict"] = []
                grouped_atomic["non_dict"].append(tag)

        # Process each group
        for type_name, type_tags in grouped_atomic.items():
            # For very large groups of atomic tags, process in smaller batches
            if len(type_tags) > 500:  # Arbitrary threshold for large groups
                batch_size = 100  # Process in batches of 100
                for i in range(0, len(type_tags), batch_size):
                    end_idx = min(i + batch_size, len(type_tags))
                    process_tag_batch(
                        type_tags[i:end_idx],
                        tag_meta,
                        chunks,
                        {
                            "tag_type": type_name,
                            "section": f"atomic_batch_{i // batch_size + 1}",
                        },
                    )
            else:
                # Process this batch of tags
                process_tag_batch(
                    type_tags,
                    tag_meta,
                    chunks,
                    {"tag_type": type_name, "section": "atomic"},
                )

    return chunks


def process_tag_batch(tag_batch, base_meta, chunks, extra_meta=None):
    """Process a batch of tags, ensuring no chunk exceeds token limit."""
    if not tag_batch:
        return

    # Initial metadata
    meta = {**base_meta}
    if extra_meta:
        meta.update(extra_meta)

    # First check if the entire batch fits within token limit
    batch_str = json.dumps(tag_batch, ensure_ascii=False)
    batch_tokens = len(enc.encode(batch_str))

    if batch_tokens <= MAX_TOKENS:
        # The whole batch fits in one chunk
        chunks.append((batch_str, meta))
        return

    # If batch is too large, try to split by size
    if len(tag_batch) > 1:
        mid = len(tag_batch) // 2
        process_tag_batch(tag_batch[:mid], base_meta, chunks, extra_meta)
        process_tag_batch(tag_batch[mid:], base_meta, chunks, extra_meta)
        return

    # If we have just one tag that's too large, process it as a single item
    if len(tag_batch) == 1:
        tag = tag_batch[0]
        tag_str = json.dumps(tag, ensure_ascii=False)
        tag_tokens = len(enc.encode(tag_str))

        if tag_tokens <= MAX_TOKENS:
            # Single tag fits within limit
            chunks.append((tag_str, meta))
        else:
            # Single tag is too large, use character chunking as fallback
            print(
                f"Warning: Found a single tag too large for context window ({tag_tokens} tokens)."
            )
            char_chunks = chunk_by_characters(tag_str, int(MAX_TOKENS * 0.9))

            for i, char_chunk in enumerate(char_chunks):
                chunk_meta = {
                    **meta,
                    "section": f"{meta.get('section', 'atomic')}_part{i + 1}",
                    "total_parts": len(char_chunks),
                }
                chunks.append((char_chunk, chunk_meta))

    return


def group_similar_udt_instances(instances):
    """Group UDT instances by similarity of their parameters or path structure."""
    groups = {"default": []}

    for instance in instances:
        # Skip if not a dictionary
        if not isinstance(instance, dict):
            groups["default"].append(instance)
            continue

        # Try to find meaningful grouping criteria
        path = instance.get("path", "")
        area = path.split("/")[0] if "/" in path else ""

        # Check if instance has parameters for grouping
        params = instance.get("parameters", {})

        if params and isinstance(params, dict) and len(params) > 0:
            # Group by key parameter values if available
            key_param = next(iter(params.keys()))  # Use first parameter as key
            key_value = params[key_param]

            if isinstance(key_value, (str, int, float, bool)):
                group_key = f"param_{key_param}_{key_value}"
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(instance)
                continue

        # If no parameters or complex parameter values, group by area
        if area:
            if area not in groups:
                groups[area] = []
            groups[area].append(instance)
        else:
            # Fall back to default group
            groups["default"].append(instance)

    return groups


def chunk_by_characters(text, max_chunk_size):
    """Chunk text by characters, ensuring no chunk exceeds the token limit."""
    chunks = []

    # Convert max_chunk_size from tokens to approximate characters (rough estimate)
    # Typically 1 token ≈ 4 characters for English text
    max_chars = int(max_chunk_size * 3)

    # Initialize chunking variables
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = min(start + max_chars, text_length)

        # If we're not at the end, try to find a good break point
        if end < text_length:
            # Try to find a natural break (newline, period, comma, etc.)
            natural_breaks = ["\n\n", "\n", ". ", ", ", " ", ".", ","]

            for separator in natural_breaks:
                # Look for the separator within a window near the end
                window_size = min(200, max_chars // 4)
                window_start = max(start, end - window_size)

                # Find the last occurrence of the separator in this window
                last_sep = text.rfind(separator, window_start, end)

                if last_sep > window_start:
                    end = last_sep + len(separator)
                    break

        # Extract the chunk
        chunk = text[start:end]

        # Verify token count for safety
        token_count = len(enc.encode(chunk))
        if token_count > max_chunk_size:
            # If still too large, use a more aggressive approach
            print(f"Warning: Chunk still too large ({token_count} tokens). Forcing smaller size.")
            # Reduce max_chars and try again from this starting point
            max_chars = max_chars // 2
            continue

        # Add chunk and move to next position
        chunks.append(chunk)
        start = end

    return chunks


def create_chunks(documents: List[Dict[str, Any]]) -> List[tuple]:
    """Create chunks from all documents based on their type."""
    all_chunks = []

    for doc in documents:
        content = doc["content"]
        meta = doc["metadata"]

        # Use the appropriate chunking strategy based on document type
        if meta.get("type") == "perspective":
            print(f"Using Context-Preserving chunking for view: {meta.get('name', 'unknown')}")
            chunks = chunk_perspective_view(content, meta)
        elif meta.get("type") == "tag":
            print(f"Using Tag Hierarchy chunking for tag file: {meta.get('filepath', 'unknown')}")
            chunks = chunk_tag_config(content, meta)
        else:
            # Default chunking for unknown types - character based fallback
            print(f"Using fallback character chunking for: {meta.get('filepath', 'unknown')}")
            content_str = json.dumps(content, ensure_ascii=False)
            tokens = len(enc.encode(content_str))

            if tokens <= MAX_TOKENS:
                chunks = [(content_str, meta)]
            else:
                # Split large content into smaller chunks
                text_chunks = chunk_by_characters(content_str, int(MAX_TOKENS / 1.2))
                chunks = [
                    (
                        chunk,
                        {
                            **meta,
                            "section": f"part{i + 1}",
                            "total_parts": len(text_chunks),
                        },
                    )
                    for i, chunk in enumerate(text_chunks)
                ]

        # Add a summary of the chunking to each chunk's metadata
        summary_chunks = []
        for chunk_text, chunk_meta in chunks:
            # Add summary information about the total chunking
            enhanced_meta = {
                **chunk_meta,
                "total_chunks": len(chunks),
                "chunking_strategy": meta.get("type", "character_based"),
            }

            # Compress context data to save space if it exists
            if "context" in enhanced_meta and len(enhanced_meta["context"]) > 500:
                # Parse and reserialize with only essential context info
                try:
                    context_data = json.loads(enhanced_meta["context"])
                    concise_context = {
                        "view": context_data.get("view", {}),
                        "component_path": context_data.get("component_path", ""),
                    }
                    # Add limited parent info if available
                    if "parent" in context_data:
                        concise_context["parent"] = {
                            "path": context_data["parent"].get("path", ""),
                            "name": context_data["parent"].get("name", ""),
                        }
                    # Replace with concise context
                    enhanced_meta["context"] = json.dumps(concise_context)
                except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                    # If there's an error parsing context, just keep it as is
                    pass

            summary_chunks.append((chunk_text, enhanced_meta))

        all_chunks.extend(summary_chunks)

    return all_chunks


def generate_embeddings(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI's API."""
    embeddings = []

    # Use mock embeddings if in mock mode
    if MOCK_EMBEDDINGS:
        print("Using mock embeddings for testing")
        for text in texts:
            embeddings.append(mock_embedding(text))
        return embeddings

    # Use OpenAI API for real embeddings
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            # Updated for OpenAI v1.0+
            response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(
                f"Generated embeddings for batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
            )
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            print("Falling back to mock embeddings for this batch")
            # Fall back to mock embeddings if the API call fails
            for text in batch:
                embeddings.append(mock_embedding(text))

    return embeddings


def index_documents(chunks: List[tuple], collection, rebuild: bool = False):
    """Index document chunks in ChromaDB."""
    # Check if there are any chunks to process
    if not chunks:
        print("No chunks to index.")
        return

    # Prepare data for indexing
    texts = [chunk[0] for chunk in chunks]
    metadatas = [chunk[1] for chunk in chunks]
    ids = [f"{os.path.basename(meta['filepath'])}_{i}" for i, meta in enumerate(metadatas)]

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embeddings(texts)

    # Handle re-indexing logic
    if not rebuild:
        existing_ids = collection.get()["ids"] if collection.count() > 0 else []

        # Find IDs that belong to files we're re-indexing
        file_paths = {meta["filepath"] for meta in metadatas}
        ids_to_delete = [
            eid
            for eid in existing_ids
            if any(eid.startswith(os.path.basename(fp)) for fp in file_paths)
        ]

        # Delete those IDs before adding new ones
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} outdated chunks...")
            collection.delete(ids=ids_to_delete)

    # Add the new chunks
    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

    print(f"Successfully indexed {len(texts)} chunks.")


def save_last_index_time():
    """Save the current time as the last index time."""
    with open(LAST_INDEX_TIME_FILE, "wb") as f:
        pickle.dump(time.time(), f)


def load_last_index_time() -> float:
    """Load the last index time or return 0 if not available."""
    try:
        with open(LAST_INDEX_TIME_FILE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        return 0


@app.command()
def main(
    path: str = typer.Argument(..., help="Path to the Ignition project directory"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the index from scratch"),
    changed_only: bool = typer.Option(
        False, "--changed-only", help="Only index files changed since last run"
    ),
    file: Optional[str] = typer.Option(None, "--file", help="Index only a specific file"),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock embeddings for testing without an OpenAI API key",
    ),
):
    """Main function to index Ignition project files."""
    print(f"Indexing Ignition project at: {path}")

    # Set mock mode if requested
    global MOCK_EMBEDDINGS
    if mock:
        MOCK_EMBEDDINGS = True
        print("Mock embedding mode enabled")

    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist")
        return

    # Initialize Chroma client and get collection
    client = setup_chroma_client()
    collection = get_collection(client, rebuild)

    # Find JSON files to index
    all_json_files = find_json_files(path)

    if file:
        # Index only a specific file
        file_path = os.path.join(path, file) if not os.path.isabs(file) else file
        if file_path in all_json_files:
            json_files = [file_path]
        else:
            print(f"Error: File {file_path} not found or is not a valid JSON file")
            return
    elif changed_only and not rebuild:
        # Index only files changed since last run
        last_index_time = load_last_index_time()
        json_files = [f for f in all_json_files if os.path.getmtime(f) > last_index_time]
        print(
            f"Found {len(json_files)} changed files since {datetime.fromtimestamp(last_index_time)}"
        )
    else:
        # Index all files
        json_files = all_json_files
        print(f"Found {len(json_files)} JSON files to index")

    if not json_files:
        print("No files to index")
        return

    # Load and process the files
    documents = load_json_files(json_files)
    chunks = create_chunks(documents)

    # Index the chunks
    index_documents(chunks, collection, rebuild)

    # Save the current time as the last index time
    save_last_index_time()

    print("Indexing complete!")


if __name__ == "__main__":
    app()
