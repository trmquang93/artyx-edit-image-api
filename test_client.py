#!/usr/bin/env python3
"""
Test client for Qwen-Image AI Editing Server.
"""

import asyncio
import base64
import json
import time
from io import BytesIO
from typing import Dict, Any, Optional

import aiohttp
import requests
from PIL import Image, ImageDraw


class QwenImageClient:
    """Client for testing Qwen-Image server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def create_test_image(self, width: int = 512, height: int = 512) -> str:
        """Create a test image and return as base64."""
        # Create a simple test image
        image = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(image)
        
        # Draw some shapes
        draw.rectangle([50, 50, width-50, height-50], outline='red', width=3)
        draw.ellipse([100, 100, width-100, height-100], fill='yellow')
        draw.text((width//2-50, height//2), "TEST", fill='black')
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def health_check(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        url = f"{self.base_url}/api/v1/health"
        
        try:
            async with self.session.get(url) as response:
                return await response.json()
        except Exception as e:
            return {"error": f"Health check failed: {str(e)}"}
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint."""
        url = f"{self.base_url}/api/v1/models"
        
        try:
            async with self.session.get(url) as response:
                return await response.json()
        except Exception as e:
            return {"error": f"Model info failed: {str(e)}"}
    
    async def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,  # Reduced for testing
        guidance_scale: float = 4.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Test image generation."""
        url = f"{self.base_url}/api/v1/generate"
        
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        try:
            async with self.session.post(url, json=payload) as response:
                return await response.json()
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
    
    async def edit_image(
        self,
        image_base64: str,
        prompt: str,
        num_inference_steps: int = 20,  # Reduced for testing
        guidance_scale: float = 4.0,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Test image editing."""
        url = f"{self.base_url}/api/v1/edit"
        
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        try:
            async with self.session.post(url, json=payload) as response:
                return await response.json()
        except Exception as e:
            return {"error": f"Editing failed: {str(e)}"}
    
    def save_result_image(self, base64_data: str, filename: str):
        """Save base64 image to file."""
        try:
            image_data = base64.b64decode(base64_data)
            with open(filename, 'wb') as f:
                f.write(image_data)
            print(f"✅ Image saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to save image: {e}")


async def run_tests(base_url: str = "http://localhost:8000"):
    """Run comprehensive tests."""
    print(f"🧪 Testing Qwen-Image server at {base_url}")
    print("=" * 50)
    
    async with QwenImageClient(base_url) as client:
        
        # Test 1: Health Check
        print("1. Testing health check...")
        health = await client.health_check()
        if "error" in health:
            print(f"❌ Health check failed: {health['error']}")
            return
        else:
            print(f"✅ Health: {health.get('status', 'unknown')}")
            print(f"   Model loaded: {health.get('model_loaded', False)}")
            print(f"   GPU available: {health.get('gpu_available', False)}")
        
        print()
        
        # Test 2: Model Info
        print("2. Testing model info...")
        model_info = await client.get_model_info()
        if "error" in model_info:
            print(f"❌ Model info failed: {model_info['error']}")
        else:
            print(f"✅ Model: {model_info.get('model_name', 'unknown')}")
            print(f"   Capabilities: {', '.join(model_info.get('capabilities', []))}")
        
        print()
        
        # Test 3: Image Generation
        print("3. Testing image generation...")
        start_time = time.time()
        
        result = await client.generate_image(
            prompt="A beautiful sunset over mountains, masterpiece, high quality",
            width=512,  # Smaller for testing
            height=512,
            num_inference_steps=20,
            seed=42
        )
        
        generation_time = time.time() - start_time
        
        if result.get("success"):
            print(f"✅ Generation successful ({generation_time:.1f}s)")
            print(f"   Processing time: {result.get('processing_time', 0):.1f}s")
            
            # Save result
            if result.get("image"):
                client.save_result_image(result["image"], "test_generation.png")
        else:
            print(f"❌ Generation failed: {result.get('message', 'unknown error')}")
        
        print()
        
        # Test 4: Image Editing
        print("4. Testing image editing...")
        
        # Create test image
        test_image = client.create_test_image(512, 512)
        
        start_time = time.time()
        
        result = await client.edit_image(
            image_base64=test_image,
            prompt="Turn the background into a starry night sky",
            num_inference_steps=20,
            strength=0.7,
            seed=42
        )
        
        editing_time = time.time() - start_time
        
        if result.get("success"):
            print(f"✅ Editing successful ({editing_time:.1f}s)")
            print(f"   Processing time: {result.get('processing_time', 0):.1f}s")
            
            # Save result
            if result.get("image"):
                client.save_result_image(result["image"], "test_editing.png")
        else:
            print(f"❌ Editing failed: {result.get('message', 'unknown error')}")
        
        print()
        print("🎉 Testing completed!")


def test_runpod_endpoint(endpoint_id: str, runpod_api_key: str):
    """Test RunPod serverless endpoint."""
    try:
        import runpod
        runpod.api_key = runpod_api_key
        
        print(f"🧪 Testing RunPod endpoint: {endpoint_id}")
        print("=" * 50)
        
        # Test 1: Health Check
        print("1. Testing health check...")
        
        result = runpod.run_sync(endpoint_id, {
            "task": "health"
        }, timeout=60)
        
        if result.get("status") == "healthy":
            print("✅ Health check successful")
            print(f"   Model loaded: {result.get('model_loaded', False)}")
            print(f"   GPU available: {result.get('gpu_available', False)}")
        else:
            print(f"❌ Health check failed: {result}")
            return
        
        print()
        
        # Test 2: Image Generation
        print("2. Testing image generation...")
        
        result = runpod.run_sync(endpoint_id, {
            "task": "generate",
            "prompt": "A beautiful landscape with mountains and lakes",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "seed": 42
        }, timeout=120)
        
        if result.get("success"):
            print("✅ Generation successful")
            print(f"   Processing time: {result.get('metadata', {}).get('processing_time', 0):.1f}s")
            
            # Save result
            if result.get("image"):
                client = QwenImageClient()
                client.save_result_image(result["image"], "runpod_generation.png")
        else:
            print(f"❌ Generation failed: {result.get('error', 'unknown')}")
        
        print()
        print("🎉 RunPod testing completed!")
        
    except ImportError:
        print("❌ RunPod package not installed. Install with: pip install runpod")
    except Exception as e:
        print(f"❌ RunPod test failed: {e}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen-Image server")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--runpod-endpoint", help="RunPod endpoint ID to test")
    parser.add_argument("--runpod-api-key", help="RunPod API key")
    
    args = parser.parse_args()
    
    if args.runpod_endpoint:
        if not args.runpod_api_key:
            print("❌ RunPod API key required for endpoint testing")
            return
        test_runpod_endpoint(args.runpod_endpoint, args.runpod_api_key)
    else:
        asyncio.run(run_tests(args.url))


if __name__ == "__main__":
    main()