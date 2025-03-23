#!/usr/bin/env python3
import os
import subprocess
import time

import typer
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

# Initialize Typer app
app = typer.Typer()


class IgnitionFileHandler(FileSystemEventHandler):
    """Handler for Ignition file changes."""

    def __init__(self, project_path, indexer_path, debounce_seconds=5):
        self.project_path = project_path
        self.indexer_path = indexer_path
        self.debounce_seconds = debounce_seconds
        self.last_event_time = 0
        self.pending_files = set()

    def on_any_event(self, event):
        """Handle all file system events."""
        # Only handle JSON files for perspective views and tags
        if not event.src_path.endswith(".json"):
            return

        # Skip temporary files and hidden files
        if event.src_path.startswith(".") or "/.git/" in event.src_path:
            return

        # Only process certain event types
        if isinstance(event, (FileCreatedEvent, FileModifiedEvent, FileDeletedEvent)):
            current_time = time.time()
            file_path = event.src_path

            print(f"Change detected in: {file_path}")

            # Add to pending files
            self.pending_files.add(file_path)

            # Update last event time
            self.last_event_time = current_time

            # Check if we need to reindex after debounce period
            if current_time - self.last_event_time >= self.debounce_seconds:
                self.check_and_reindex()

    def check_and_reindex(self):
        """Check if enough time has passed since last event and reindex if needed."""
        if not self.pending_files:
            return

        if time.time() - self.last_event_time >= self.debounce_seconds:
            print(f"Re-indexing {len(self.pending_files)} changed files...")

            # Convert to absolute paths if needed
            abs_files = []
            for file_path in self.pending_files:
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
                abs_files.append(file_path)

            # Run the indexer for each file
            for file_path in abs_files:
                # Extract relative path from project root
                rel_path = os.path.relpath(file_path, self.project_path)

                # Run indexer command
                try:
                    cmd = [
                        "python",
                        self.indexer_path,
                        "--path",
                        self.project_path,
                        "--file",
                        rel_path,
                    ]
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    print(f"Successfully re-indexed: {rel_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error re-indexing {rel_path}: {e}")

            # Clear pending files
            self.pending_files.clear()


@app.command()
def main(
    project_path: str = typer.Argument(
        ..., help="Path to the Ignition project directory to watch"
    ),
    indexer_path: str = typer.Option("indexer.py", help="Path to the indexer script"),
    debounce: int = typer.Option(5, help="Debounce period in seconds"),
):
    """Watch an Ignition project directory and re-index files when they change."""
    # Resolve paths
    abs_project_path = os.path.abspath(project_path)
    abs_indexer_path = os.path.abspath(indexer_path)

    # Verify paths exist
    if not os.path.isdir(abs_project_path):
        print(
            f"Error: Project path '{abs_project_path}' does not exist or is not a directory"
        )
        return

    if not os.path.isfile(abs_indexer_path):
        print(f"Error: Indexer script '{abs_indexer_path}' does not exist")
        return

    # Create event handler and observer
    event_handler = IgnitionFileHandler(abs_project_path, abs_indexer_path, debounce)
    observer = Observer()

    # Start watching the directory
    observer.schedule(event_handler, abs_project_path, recursive=True)
    observer.start()

    print(f"Watching {abs_project_path} for changes...")
    print("Press Ctrl+C to stop")

    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
            # Check for pending reindexing
            event_handler.check_and_reindex()
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
    print("Watcher stopped")


if __name__ == "__main__":
    app()
