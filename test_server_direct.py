#!/usr/bin/env python3
"""
Test the deployed Qwen Image Edit server directly.
"""

import requests
import json
import base64
import io
from PIL import Image

# Set your RunPod endpoint details here
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/jxeyck5w71jw8o"
API_KEY = "YOUR_RUNPOD_API_KEY_HERE"  # Replace with your actual API key

def create_test_image_base64():
    """Create a simple test image and return as base64."""
    # Create a 512x512 test image
    image = Image.new('RGB', (512, 512), color='lightblue')
    
    # Add some visual content
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw a red circle (subject)
    draw.ellipse([156, 156, 356, 356], fill='red')
    
    # Add text
    draw.text((200, 250), "TEST", fill='white')
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    # Encode to base64
    image_data = buffer.getvalue()
    base64_string = base64.b64encode(image_data).decode('utf-8')
    
    return base64_string

def test_deployed_server():
    """Test the deployed server."""
    
    print(f"🧪 Testing Deployed Qwen Image Edit Server")
    print(f"📡 Endpoint: {RUNPOD_ENDPOINT}")
    print("=" * 60)
    
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image_base64 = create_test_image_base64()
    print(f"✅ Test image created (base64 length: {len(test_image_base64)})")
    
    # Test 1: Health check
    print("\n1️⃣  Testing health check...")
    try:
        payload = {"input": {"task": "health"}}
        response = requests.post(f"{RUNPOD_ENDPOINT}/run", headers=headers, json=payload, timeout=30)
        
        print(f"📊 Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received!")
            print(f"   Raw response: {json.dumps(result, indent=2)[:500]}...")
            
        else:
            print(f"❌ Health check failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Qwen Image Edit
    print("\n2️⃣  Testing Qwen image editing (background replacement)...")
    try:
        payload = {
            "input": {
                "task": "edit",
                "image": test_image_base64,
                "prompt": "replace background with beautiful tropical beach with palm trees",
                "num_inference_steps": 15,
                "guidance_scale": 7.5,
                "strength": 0.8
            }
        }
        response = requests.post(f"{RUNPOD_ENDPOINT}/run", headers=headers, json=payload, timeout=180)
        
        print(f"📊 Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received!")
            print(f"   Raw response: {json.dumps(result, indent=2)[:500]}...")
        else:
            print(f"❌ Edit request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Edit error: {e}")

    print("\n" + "=" * 60)
    print("🎯 Test completed! Check results above.")

if __name__ == "__main__":
    test_deployed_server()