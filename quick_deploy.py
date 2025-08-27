#!/usr/bin/env python3
"""
Quick deployment script for RunPod without local Docker build.
"""

import json
import os
import sys
import time

try:
    import runpod
except ImportError:
    print("Installing runpod package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod"])
    import runpod

# Set API key from environment variable
runpod.api_key = os.getenv("RUNPOD_API_KEY", "your-runpod-api-key")

def test_api_connection():
    """Test RunPod API connection."""
    try:
        # List available GPU types
        gpu_types = runpod.get_gpu_types()
        print("✅ RunPod API connection successful!")
        print(f"Available GPU types: {len(gpu_types)} found")
        return True
    except Exception as e:
        print(f"❌ RunPod API connection failed: {e}")
        return False

def create_template_from_github():
    """Create template using GitHub repository."""
    print("Creating RunPod template from source...")
    
    template_config = {
        "name": "qwen-image-editing-server",
        "description": "AI Image Editing Server using Qwen-Image model",
        "readme": "# Qwen-Image AI Editing Server\n\nAdvanced AI image generation and editing using Qwen-Image 20B model.",
        "dockerArgs": "",
        "containerDiskInGb": 50,
        "volumeInGb": 0,
        "volumeMountPath": "",
        "ports": "8000/http",
        "env": [
            {
                "key": "LOG_LEVEL",
                "value": "INFO",
                "description": "Logging level"
            },
            {
                "key": "RUNPOD_MODE",
                "value": "true",
                "description": "Enable RunPod serverless mode"
            },
            {
                "key": "TORCH_HOME",
                "value": "/tmp/.torch",
                "description": "PyTorch cache directory"
            },
            {
                "key": "HF_HOME", 
                "value": "/tmp/.huggingface",
                "description": "HuggingFace cache directory"
            }
        ],
        "isPublic": False,
        # Use a base PyTorch image for now
        "imageName": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
    }
    
    try:
        template = runpod.create_template(template_config)
        template_id = template["id"]
        print(f"✅ Template created: {template_id}")
        return template_id
    except Exception as e:
        print(f"❌ Failed to create template: {e}")
        return None

def create_endpoint(template_id):
    """Create RunPod serverless endpoint."""
    endpoint_name = f"qwen-image-{int(time.time())}"
    print(f"Creating endpoint: {endpoint_name}")
    
    endpoint_config = {
        "name": endpoint_name,
        "template_id": template_id,
        "locations": {
            "US": {
                "workers_min": 0,
                "workers_max": 2,
                "idle_timeout": 5,
                "scaler_type": "QUEUE_DELAY",
                "scaler_value": 4
            }
        },
        "network_volume_id": None,
        "flashboot": True
    }
    
    try:
        endpoint = runpod.create_endpoint(endpoint_config)
        print(f"✅ Endpoint created!")
        print(f"   ID: {endpoint['id']}")
        print(f"   Name: {endpoint_name}")
        return endpoint
    except Exception as e:
        print(f"❌ Failed to create endpoint: {e}")
        return None

def test_endpoint(endpoint_id):
    """Test the endpoint with a simple request."""
    print(f"Testing endpoint: {endpoint_id}")
    print("Note: This may take a few minutes for the first request...")
    
    try:
        # Simple health check
        result = runpod.run_sync(endpoint_id, {
            "task": "health"
        }, timeout=300)  # 5 minutes timeout
        
        print("Test result:", result)
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if the custom code isn't deployed yet.")
        return False

def main():
    print("🚀 Quick RunPod Deployment")
    print("=" * 50)
    
    # Test API connection
    if not test_api_connection():
        return
    
    # Create template
    template_id = create_template_from_github()
    if not template_id:
        return
    
    # Create endpoint
    endpoint = create_endpoint(template_id)
    if not endpoint:
        return
    
    # Save deployment info
    deployment_info = {
        "endpoint_id": endpoint["id"],
        "template_id": template_id,
        "deployed_at": time.time(),
        "status": "created"
    }
    
    with open("deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n✅ Deployment completed!")
    print(f"Endpoint ID: {endpoint['id']}")
    print(f"Template ID: {template_id}")
    print("\n📝 Next steps:")
    print("1. The endpoint is created but needs custom code deployment")
    print("2. You'll need to build a custom Docker image with the Qwen-Image code")
    print("3. Or use RunPod's GitHub integration to deploy from repository")
    print("\nDeployment info saved to deployment_info.json")

if __name__ == "__main__":
    main()