#!/bin/bash

# Run the FastAPI server with hot reload for development
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8081

# To use the indexing endpoint, you can make a POST request to the /index endpoint:
# curl -X POST http://localhost:8081/index -H "Content-Type: application/json" -d '{"project_path":"./whk-ignition-scada", "rebuild":true, "skip_rate_limiting":false}' 