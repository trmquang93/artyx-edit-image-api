#!/usr/bin/env python3
"""
Test the deployed Qwen Image Edit server with real API calls.
"""

import requests
import json
import base64
import io
from PIL import Image

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

def test_server_endpoint(base_url, api_key=None):
    """Test the deployed server endpoint."""
    
    print(f"🧪 Testing Deployed Qwen Image Edit Server")
    print(f"📡 Base URL: {base_url}")
    print("=" * 60)
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image_base64 = create_test_image_base64()
    print(f"✅ Test image created (base64 length: {len(test_image_base64)})")
    
    # Test 1: Health check
    print("\n1️⃣  Testing health check...")
    try:
        payload = {"input": {"task": "health"}}
        response = requests.post(f"{base_url}/run", headers=headers, json=payload, timeout=30)
        
        print(f"📊 Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health: {result.get('status', 'Unknown')}")
            if 'environment' in result:
                env = result['environment']
                print(f"   GPU Available: {env.get('gpu_available', 'Unknown')}")
                print(f"   Model Loaded: {env.get('model_loaded', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Image generation
    print("\n2️⃣  Testing Qwen image generation...")
    try:
        payload = {
            "input": {
                "task": "generate",
                "prompt": "a beautiful sunset over mountains, masterpiece, best quality",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        response = requests.post(f"{base_url}/run", headers=headers, json=payload, timeout=120)
        
        print(f"📊 Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Generation successful!")
                print(f"   Message: {result.get('message', 'N/A')}")
                if 'metadata' in result:
                    meta = result['metadata']
                    print(f"   Model: {meta.get('model', 'N/A')}")
                    print(f"   Processing time: {meta.get('processing_time', 'N/A'):.2f}s")
            else:
                print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Generation request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
    
    # Test 3: Qwen Image Edit
    print("\n3️⃣  Testing Qwen image editing (background replacement)...")
    try:
        payload = {
            "input": {
                "task": "edit",
                "image": test_image_base64,
                "prompt": "replace background with beautiful tropical beach with palm trees",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "strength": 0.8
            }
        }
        response = requests.post(f"{base_url}/run", headers=headers, json=payload, timeout=120)
        
        print(f"📊 Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Qwen Image Edit successful!")
                print(f"   Message: {result.get('message', 'N/A')}")
                if 'metadata' in result:
                    meta = result['metadata']
                    print(f"   Model: {meta.get('model', 'N/A')}")
                    print(f"   Processing time: {meta.get('processing_time', 'N/A'):.2f}s")
                    print(f"   Prompt: {meta.get('prompt', 'N/A')}")
            else:
                print(f"❌ Edit failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Edit request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Edit error: {e}")

    print("\n" + "=" * 60)
    print("🎯 Test completed! Check results above.")

def main():
    """Main test function."""
    
    print("Please provide your RunPod endpoint details:")
    
    # You can hardcode these or input them
    base_url = input("Enter RunPod endpoint URL (without /run): ").strip()
    if not base_url:
        print("❌ No endpoint URL provided")
        return
    
    api_key = input("Enter RunPod API key (optional, press Enter to skip): ").strip()
    if not api_key:
        api_key = None
        
    test_server_endpoint(base_url, api_key)

if __name__ == "__main__":
    # You can also directly call with your endpoint:
    # test_server_endpoint("https://api.runpod.ai/v2/your-endpoint-id", "your-api-key")
    main()