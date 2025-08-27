#!/usr/bin/env python3
"""
Test RunPod API connection and create endpoint.
"""

import json
import os
import time
import requests

# API configuration
API_KEY = os.getenv("RUNPOD_API_KEY", "your-runpod-api-key")
BASE_URL = "https://api.runpod.ai/graphql"

def test_api_connection():
    """Test basic API connection."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Query to test connection
    query = """
    query {
        myself {
            id
            email
        }
    }
    """
    
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]["myself"]:
                print("✅ RunPod API connection successful!")
                print(f"   User ID: {data['data']['myself']['id']}")
                print(f"   Email: {data['data']['myself']['email']}")
                return True
            else:
                print(f"❌ API error: {data}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def get_gpu_types():
    """Get available GPU types."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    query = """
    query {
        gpuTypes {
            id
            displayName
            memoryInGb
        }
    }
    """
    
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "gpuTypes" in data["data"]:
                gpu_types = data["data"]["gpuTypes"]
                print(f"✅ Found {len(gpu_types)} GPU types:")
                for gpu in gpu_types[:5]:  # Show first 5
                    print(f"   {gpu['displayName']} ({gpu['memoryInGb']}GB)")
                return gpu_types
            else:
                print(f"❌ GPU types error: {data}")
                return []
        else:
            print(f"❌ GPU types HTTP error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Failed to get GPU types: {e}")
        return []

def create_template():
    """Create a template for the Qwen-Image server."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Template creation mutation
    mutation = """
    mutation saveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
        }
    }
    """
    
    template_input = {
        "name": f"qwen-image-server-{int(time.time())}",
        "imageName": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        "dockerArgs": "",
        "containerDiskInGb": 50,
        "volumeInGb": 0,
        "volumeMountPath": "",
        "ports": "8000/http",
        "env": [
            {"key": "LOG_LEVEL", "value": "INFO"},
            {"key": "RUNPOD_MODE", "value": "true"},
            {"key": "TORCH_HOME", "value": "/tmp/.torch"},
            {"key": "HF_HOME", "value": "/tmp/.huggingface"}
        ],
        "isPublic": False,
        "readme": "Qwen-Image AI editing server"
    }
    
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json={
                "query": mutation,
                "variables": {"input": template_input}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "saveTemplate" in data["data"]:
                template = data["data"]["saveTemplate"]
                print(f"✅ Template created!")
                print(f"   ID: {template['id']}")
                print(f"   Name: {template['name']}")
                return template["id"]
            else:
                print(f"❌ Template creation error: {data}")
                return None
        else:
            print(f"❌ Template creation HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to create template: {e}")
        return None

def create_endpoint(template_id):
    """Create a serverless endpoint."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    mutation = """
    mutation saveEndpoint($input: SaveEndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            url
        }
    }
    """
    
    endpoint_input = {
        "name": f"qwen-image-endpoint-{int(time.time())}",
        "templateId": template_id,
        "workersMin": 0,
        "workersMax": 2,
        "idleTimeout": 5,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "locations": ["US"],
        "flashboot": True
    }
    
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json={
                "query": mutation,
                "variables": {"input": endpoint_input}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "saveEndpoint" in data["data"]:
                endpoint = data["data"]["saveEndpoint"]
                print(f"✅ Endpoint created!")
                print(f"   ID: {endpoint['id']}")
                print(f"   Name: {endpoint['name']}")
                if endpoint.get('url'):
                    print(f"   URL: {endpoint['url']}")
                return endpoint
            else:
                print(f"❌ Endpoint creation error: {data}")
                return None
        else:
            print(f"❌ Endpoint creation HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to create endpoint: {e}")
        return None

def main():
    print("🚀 RunPod API Test and Deployment")
    print("=" * 50)
    
    # Test API connection
    if not test_api_connection():
        return
    
    print()
    
    # Get available GPU types
    gpu_types = get_gpu_types()
    if not gpu_types:
        print("⚠️  Continuing without GPU type info...")
    
    print()
    
    # Create template
    print("Creating template...")
    template_id = create_template()
    if not template_id:
        print("❌ Failed to create template")
        return
    
    print()
    
    # Create endpoint
    print("Creating endpoint...")
    endpoint = create_endpoint(template_id)
    if not endpoint:
        print("❌ Failed to create endpoint")
        return
    
    # Save deployment info
    deployment_info = {
        "endpoint_id": endpoint["id"],
        "endpoint_name": endpoint["name"],
        "template_id": template_id,
        "deployed_at": time.time(),
        "api_key": "rpa_***" + API_KEY[-6:],  # Partial key for reference
        "status": "created"
    }
    
    with open("runpod_deployment.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n🎉 Deployment completed!")
    print(f"Endpoint ID: {endpoint['id']}")
    print(f"Template ID: {template_id}")
    print("\n📝 Next steps:")
    print("1. The endpoint is created with base PyTorch image")
    print("2. You need to deploy custom code to run Qwen-Image")
    print("3. Use RunPod CLI or web interface to add your code")
    print("\nDeployment info saved to runpod_deployment.json")

if __name__ == "__main__":
    main()