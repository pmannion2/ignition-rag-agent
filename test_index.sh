#!/bin/bash

# Test the index endpoint
curl -X POST "http://localhost:8081/index" \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "./whk-ignition-scada",
    "rebuild": true,
    "skip_rate_limiting": false
  }' 