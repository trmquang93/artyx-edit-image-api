#!/usr/bin/env python3
"""
Local test script to validate real Qwen-Image integration.
Tests both text-to-image and image editing functionality.
"""

import asyncio
import base64
import io
import logging
import time
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our QwenImageManager
from runpod_handler import QwenImageManager


def create_test_image():
    """Create a simple test image for editing tests."""
    # Create a 512x512 test image with a simple pattern
    image = Image.new('RGB', (512, 512), color='lightblue')
    
    # Add some simple shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw a red circle in the center
    center_x, center_y = 256, 256
    radius = 100
    draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], fill='red')
    
    # Draw some text
    draw.text((50, 50), "TEST IMAGE", fill="black")
    draw.text((50, 400), "Background should change", fill="black")
    
    return image


def pil_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_string):
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


async def test_text_to_image():
    """Test Qwen-Image text-to-image generation."""
    logger.info("🧪 Testing Qwen-Image text-to-image generation...")
    
    manager = QwenImageManager()
    await manager.initialize()
    
    try:
        # Test simple generation
        start_time = time.time()
        result_base64 = await manager.generate_image(
            prompt="A beautiful sunset over mountains",
            width=512,
            height=512,
            num_inference_steps=20,  # Faster for testing
            guidance_scale=4.0
        )
        processing_time = time.time() - start_time
        
        # Convert result to PIL and save
        result_image = base64_to_pil(result_base64)
        result_image.save("test_qwen_generated.png")
        
        logger.info(f"✅ Text-to-image completed in {processing_time:.2f}s")
        logger.info(f"📁 Result saved as 'test_qwen_generated.png'")
        logger.info(f"📐 Result size: {result_image.size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Text-to-image failed: {e}")
        return False


async def test_image_editing():
    """Test Qwen-Image-Edit image editing."""
    logger.info("🧪 Testing Qwen-Image-Edit image editing...")
    
    manager = QwenImageManager()
    await manager.initialize()
    
    try:
        # Create test image
        test_image = create_test_image()
        test_image.save("test_input.png")
        test_base64 = pil_to_base64(test_image)
        
        logger.info("📸 Created test input image")
        
        # Test image editing
        start_time = time.time()
        result_base64 = await manager.edit_image(
            image_base64=test_base64,
            prompt="Change the background to a beautiful forest",
            num_inference_steps=30,  # Optimized for quality
            guidance_scale=4.0,
            strength=0.8
        )
        processing_time = time.time() - start_time
        
        # Convert result to PIL and save
        result_image = base64_to_pil(result_base64)
        result_image.save("test_qwen_edited.png")
        
        logger.info(f"✅ Image editing completed in {processing_time:.2f}s")
        logger.info(f"📁 Result saved as 'test_qwen_edited.png'")
        logger.info(f"📐 Result size: {result_image.size}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Image editing failed: {e}")
        return False


async def test_health_check():
    """Test model health and system info."""
    logger.info("🧪 Testing system health...")
    
    manager = QwenImageManager()
    await manager.initialize()
    
    try:
        health_info = await manager.get_health_info()
        
        logger.info("💡 System Health Info:")
        logger.info(f"   Model loaded: {health_info.get('model_loaded', False)}")
        logger.info(f"   GPU available: {health_info.get('gpu_available', False)}")
        
        if 'memory_usage' in health_info:
            mem = health_info['memory_usage']
            logger.info(f"   GPU Memory: {mem.get('allocated_gb', 0):.1f}GB allocated / {mem.get('total_gb', 0):.1f}GB total")
            logger.info(f"   GPU Utilization: {mem.get('utilization', 0):.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Qwen-Image Integration Tests")
    
    results = {
        'health': False,
        'text_to_image': False,
        'image_editing': False
    }
    
    # Test 1: Health check
    logger.info("\n" + "="*50)
    results['health'] = await test_health_check()
    
    # Test 2: Text-to-image generation
    logger.info("\n" + "="*50)
    results['text_to_image'] = await test_text_to_image()
    
    # Test 3: Image editing
    logger.info("\n" + "="*50)
    results['image_editing'] = await test_image_editing()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("🏁 TEST RESULTS SUMMARY:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    logger.info(f"\n📊 Overall: {total_passed}/3 tests passed")
    
    if total_passed == 3:
        logger.info("🎉 All tests passed! Qwen-Image integration is working correctly.")
    elif total_passed > 0:
        logger.info("⚠️  Some tests passed. Qwen-Image may be partially working.")
    else:
        logger.info("💥 All tests failed. Qwen-Image integration needs debugging.")
    
    return total_passed == 3


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)