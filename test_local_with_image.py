#!/usr/bin/env python3
"""
Local test for RunPod handler with real image data.
"""

import sys
import os
import base64
import io
from PIL import Image

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runpod_handler import handler

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

def test_handler_with_real_image():
    """Test the RunPod handler function with real image data."""
    print("🧪 Testing RunPod Handler with Real Image")
    print("=" * 50)
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image_base64 = create_test_image_base64()
    print(f"✅ Test image created (base64 length: {len(test_image_base64)})")
    
    # Test health check
    print("\n1. Testing health check...")
    health_job = {"input": {"task": "health"}}
    result = handler(health_job)
    print(f"Health: {result['success']} - {result['message']}")
    
    # Test image generation
    print("\n2. Testing image generation...")
    generate_job = {
        "input": {
            "task": "generate", 
            "prompt": "a beautiful sunset over mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20
        }
    }
    result = handler(generate_job)
    print(f"Generate: {result['success']} - {result['message']}")
    if result['success']:
        print(f"   Model: {result['metadata']['model']}")
        print(f"   Processing time: {result['metadata']['processing_time']:.2f}s")
    
    # Test image editing with real image
    print("\n3. Testing image editing with real image...")
    edit_job = {
        "input": {
            "task": "edit",
            "image": test_image_base64,
            "prompt": "replace background with beautiful forest",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "strength": 0.8
        }
    }
    result = handler(edit_job)
    print(f"Edit: {result['success']} - {result.get('message', result.get('error', 'Unknown'))}")
    if result['success']:
        print(f"   Model: {result['metadata']['model']}")
        print(f"   Processing time: {result['metadata']['processing_time']:.2f}s")
    
    # Test background replacement
    print("\n4. Testing background replacement...")
    bg_job = {
        "input": {
            "task": "background_replacement",
            "image": test_image_base64,
            "prompt": "tropical beach with palm trees",
            "num_inference_steps": 15
        }
    }
    result = handler(bg_job)
    print(f"Background replacement: {result['success']} - {result.get('message', result.get('error', 'Unknown'))}")
    if result['success']:
        print(f"   Model: {result['metadata']['model']}")
        print(f"   Processing time: {result['metadata']['processing_time']:.2f}s")
    
    print("\n✅ All handler tests completed!")
    return True

if __name__ == "__main__":
    test_handler_with_real_image()