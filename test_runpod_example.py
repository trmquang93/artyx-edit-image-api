#!/usr/bin/env python3
"""
Example test script for RunPod endpoint.
Set your API key as environment variable: export RUNPOD_API_KEY=your_key_here
"""

import requests
import json
import time
import base64
import os
from PIL import Image, ImageDraw
import io

# Configuration - set these environment variables
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "25v1qttrao3ikz")
API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

if not API_KEY:
    print("❌ Please set RUNPOD_API_KEY environment variable")
    print("Example: export RUNPOD_API_KEY=rpa_your_key_here")
    exit(1)

def test_health():
    """Test health check endpoint."""
    print("🏥 Testing health check...")
    
    payload = {
        "input": {
            "task": "health"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Health check successful!")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Health check failed!")
        print(f"Response: {response.text}")

def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (512, 512), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple pattern
    draw.rectangle([100, 100, 400, 400], fill='white', outline='blue', width=5)
    draw.text((200, 250), "TEST IMAGE", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_b64

def test_generate():
    """Test image generation."""
    print("\n🎨 Testing image generation...")
    
    payload = {
        "input": {
            "task": "generate",
            "prompt": "A beautiful sunset over mountains, photorealistic",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 4.0
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Generation request successful!")
        print(f"Job ID: {result.get('id')}")
        
        # Poll for results (this is a sync endpoint, but showing the pattern)
        if 'output' in result:
            print("🖼️ Generation complete!")
            print(f"Success: {result['output'].get('success')}")
            if result['output'].get('image'):
                print("📸 Image generated successfully (base64 data received)")
    else:
        print("❌ Generation failed!")
        print(f"Response: {response.text}")

def test_edit():
    """Test image editing."""
    print("\n✏️ Testing image editing...")
    
    # Create test image
    test_img_b64 = create_test_image()
    
    payload = {
        "input": {
            "task": "edit",
            "image": test_img_b64,
            "prompt": "Transform this into a beautiful landscape painting",
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "strength": 0.8
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Edit request successful!")
        print(f"Job ID: {result.get('id')}")
        
        if 'output' in result:
            print("🖼️ Edit complete!")
            print(f"Success: {result['output'].get('success')}")
            if result['output'].get('image'):
                print("📸 Image edited successfully (base64 data received)")
    else:
        print("❌ Edit failed!")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    print("🚀 RunPod Qwen-Image API Test")
    print(f"Endpoint: {BASE_URL}")
    print("-" * 50)
    
    test_health()
    test_generate()
    test_edit()
    
    print("\n✅ All tests completed!")