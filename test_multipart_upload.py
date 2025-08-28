#!/usr/bin/env python3
"""
Test script for multipart upload functionality
"""

import requests
import base64
import io
from PIL import Image

def create_test_image():
    """Create a simple test image."""
    # Create a 400x300 red image
    image = Image.new('RGB', (400, 300), color='red')
    
    # Add some simple content
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Draw a blue rectangle
    draw.rectangle([50, 50, 350, 250], fill='blue')
    
    # Add text (use default font)
    draw.text((100, 150), "TEST IMAGE", fill='white')
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    return buffer

def test_multipart_upload(base_url="http://localhost:8000"):
    """Test the multipart upload endpoints."""
    
    print("🧪 Testing Multipart Upload Functionality")
    print("=" * 50)
    
    # Create test image
    print("📸 Creating test image...")
    image_buffer = create_test_image()
    
    # Test health endpoint first
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to server: {e}")
        return False
    
    # Test multipart image editing
    print("\n2️⃣ Testing multipart image editing...")
    try:
        files = {
            'file': ('test_image.jpg', image_buffer, 'image/jpeg')
        }
        data = {
            'prompt': 'Add beautiful flowers and butterflies',
            'negative_prompt': 'dark, scary',
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'strength': 0.8
        }
        
        response = requests.post(
            f"{base_url}/api/v1/upload/edit",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image editing successful!")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Result image size: {len(result['image'])} chars (base64)")
            print(f"   Message: {result['message']}")
        else:
            print(f"❌ Image editing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Test multipart background replacement
    print("\n3️⃣ Testing multipart background replacement...")
    try:
        # Reset buffer position
        image_buffer.seek(0)
        
        files = {
            'file': ('test_image.jpg', image_buffer, 'image/jpeg')
        }
        data = {
            'prompt': 'Beautiful sunset over mountains',
            'negative_prompt': 'urban, city',
            'num_inference_steps': 25,
            'guidance_scale': 8.0,
            'strength': 0.9
        }
        
        response = requests.post(
            f"{base_url}/api/v1/upload/background-replacement",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Background replacement successful!")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Result image size: {len(result['image'])} chars (base64)")
            print(f"   Task type: {result['metadata'].get('task_type')}")
        else:
            print(f"❌ Background replacement failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    print("\n🎉 All multipart upload tests passed!")
    return True

def test_with_curl_examples(base_url="http://localhost:8000"):
    """Print curl examples for testing."""
    print("\n📝 CURL Examples for Manual Testing:")
    print("=" * 50)
    
    print("\n# Test image editing with file upload:")
    print(f"""curl -X POST "{base_url}/api/v1/upload/edit" \\
  -F "file=@/path/to/your/image.jpg" \\
  -F "prompt=Add beautiful flowers and make it vibrant" \\
  -F "negative_prompt=dark, blurry" \\
  -F "num_inference_steps=30" \\
  -F "guidance_scale=7.5" \\
  -F "strength=0.8" \\
  -F "seed=42"
""")
    
    print("\n# Test background replacement with file upload:")
    print(f"""curl -X POST "{base_url}/api/v1/upload/background-replacement" \\
  -F "file=@/path/to/your/image.jpg" \\
  -F "prompt=Beautiful sunset over ocean waves" \\
  -F "negative_prompt=city, urban" \\
  -F "num_inference_steps=25" \\
  -F "guidance_scale=8.0" \\
  -F "strength=0.9"
""")
    
    print(f"\n# Check API documentation:")
    print(f"# Open in browser: {base_url}/docs")

if __name__ == "__main__":
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"🎯 Testing server at: {base_url}")
    
    # Run tests
    success = test_multipart_upload(base_url)
    
    # Show curl examples
    test_with_curl_examples(base_url)
    
    if success:
        print(f"\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1)