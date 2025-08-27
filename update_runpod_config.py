#!/usr/bin/env python3
"""
Update RunPod endpoint to use GHCR image instead of building from source.
"""

import requests
import json
import os

# Configuration
ENDPOINT_ID = "25v1qttrao3ikz"
API_KEY = os.getenv("RUNPOD_API_KEY")
GHCR_IMAGE = "ghcr.io/trmquang93/artyx-edit-image-api:latest"

if not API_KEY:
    print("❌ Please set RUNPOD_API_KEY environment variable")
    exit(1)

def update_endpoint_config():
    """Update endpoint to use pre-built container image."""
    
    url = f"https://api.runpod.ai/graphql"
    
    # GraphQL mutation to update endpoint
    mutation = """
    mutation updateEndpoint($input: EndpointInput!) {
        updateEndpoint(input: $input) {
            id
            name
            template {
                containerDiskInGb
                dockerArgs
                env {
                    key
                    value
                }
                imageName
                name
                ports
                volumeInGb
                volumeMountPath
            }
        }
    }
    """
    
    # Configuration for using pre-built image
    variables = {
        "input": {
            "endpointId": ENDPOINT_ID,
            "template": {
                "imageName": GHCR_IMAGE,
                "containerDiskInGb": 20,
                "volumeInGb": 0,
                "env": [
                    {"key": "RUNPOD_MODE", "value": "true"},
                    {"key": "LOG_LEVEL", "value": "INFO"}
                ],
                "dockerArgs": "",
                "ports": "8000/http",
                "volumeMountPath": "/workspace"
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": mutation,
        "variables": variables
    }
    
    print(f"🔄 Updating endpoint {ENDPOINT_ID} to use image: {GHCR_IMAGE}")
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        if "errors" in result:
            print("❌ Error updating endpoint:")
            print(json.dumps(result["errors"], indent=2))
        else:
            print("✅ Endpoint updated successfully!")
            print(json.dumps(result, indent=2))
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    update_endpoint_config()