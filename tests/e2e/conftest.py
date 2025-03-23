import logging
import os
import time

import pytest
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("e2e-tests")

# Constants
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8001")
API_URL = f"http://{API_HOST}:{API_PORT}"
MAX_RETRIES = 30
RETRY_INTERVAL = 5


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    """Wait for the API to be available before running tests."""
    logger.info(f"Waiting for API at {API_URL} to be ready")

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                logger.info(f"API is ready after {attempt + 1} attempts")
                # Give the API a moment to fully initialize
                time.sleep(2)
                return
        except requests.RequestException as e:
            logger.info(f"API connection error: {e}")

        logger.info(
            f"API not ready yet, retrying in {RETRY_INTERVAL} seconds... (attempt {attempt + 1}/{MAX_RETRIES})"
        )
        time.sleep(RETRY_INTERVAL)

    pytest.fail("API service did not become available in time")


@pytest.fixture(scope="session")
def api_url():
    """Provide API URL to test functions."""
    return API_URL
