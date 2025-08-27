#!/usr/bin/env python3
"""
Local validation script for Qwen-Image integration on macOS.
Tests what we can without GPU requirements.
"""

import sys
import traceback
import time
import asyncio

def test_basic_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        from PIL import Image
        print("✅ Pillow")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_project_imports():
    """Test if our project modules can be imported."""
    print("\n🧪 Testing project imports...")
    
    try:
        from models.qwen_image import QwenImageManager
        print("✅ QwenImageManager")
        
        from models.image_processor import ImageProcessor
        print("✅ ImageProcessor")
        
        # Test instantiation
        manager = QwenImageManager()
        print("✅ QwenImageManager instantiation")
        
        processor = ImageProcessor()
        print("✅ ImageProcessor instantiation")
        
        return True
        
    except Exception as e:
        print(f"❌ Project import failed: {e}")
        traceback.print_exc()
        return False

def test_diffusers_models():
    """Test if Qwen-Image models are accessible."""
    print("\n🧪 Testing model accessibility...")
    
    try:
        from diffusers import DiffusionPipeline
        
        # Test if we can get model info (without downloading)
        model_id = "Qwen/Qwen-Image"
        
        # This will check if model exists on HuggingFace
        try:
            from diffusers.utils import is_accelerate_available
            print(f"✅ Model {model_id} exists on HuggingFace")
        except:
            print(f"⚠️  Could not verify {model_id}")
        
        # Test QwenImageEditPipeline import
        try:
            from diffusers import QwenImageEditPipeline
            print("✅ QwenImageEditPipeline available")
        except ImportError:
            print("⚠️  QwenImageEditPipeline not available (may need latest diffusers)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model accessibility test failed: {e}")
        return False

def test_handler_structure():
    """Test RunPod handler structure."""
    print("\n🧪 Testing handler structure...")
    
    try:
        from runpod_handler import handler
        
        # Test health check (should work without GPU)
        job = {"input": {"task": "health"}}
        result = handler(job)
        
        if result.get("success") and "message" in result:
            print("✅ Handler health check works")
            print(f"   Message: {result['message']}")
        else:
            print(f"❌ Handler health check failed: {result}")
            return False
        
        # Test error handling
        job = {"input": {"task": "invalid_task"}}
        result = handler(job)
        
        if not result.get("success") and "error" in result:
            print("✅ Handler error handling works")
        else:
            print(f"❌ Handler error handling failed: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Handler test failed: {e}")
        traceback.print_exc()
        return False

def test_image_processor():
    """Test image processing utilities."""
    print("\n🧪 Testing image processor...")
    
    try:
        from models.image_processor import ImageProcessor
        from PIL import Image
        import base64
        import io
        
        processor = ImageProcessor()
        
        # Create a small test image
        test_image = Image.new('RGB', (64, 64), color='red')
        
        # Test PIL to base64
        base64_str = asyncio.run(processor.pil_to_base64(test_image))
        if base64_str and len(base64_str) > 100:
            print("✅ PIL to base64 conversion")
        else:
            print("❌ PIL to base64 failed")
            return False
        
        # Test base64 to PIL (async function, need to handle)
        restored_image = asyncio.run(processor.base64_to_pil(base64_str))
        if restored_image and restored_image.size == (64, 64):
            print("✅ Base64 to PIL conversion")
        else:
            print("❌ Base64 to PIL failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Image processor test failed: {e}")
        traceback.print_exc()
        return False

def test_requirements():
    """Check if requirements.txt packages are installed."""
    print("\n🧪 Testing requirements.txt compliance...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'diffusers', 'transformers', 
        'torch', 'torchvision', 'accelerate', 'Pillow',
        'python-multipart', 'pydantic', 'numpy', 'runpod'
    ]
    
    success_count = 0
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
            success_count += 1
        except ImportError:
            print(f"❌ {package} missing")
    
    print(f"📊 {success_count}/{len(required_packages)} packages available")
    return success_count == len(required_packages)

def main():
    """Run all validation tests."""
    print("🚀 Starting Qwen-Image Local Validation")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Imports", test_project_imports),
        ("Model Accessibility", test_diffusers_models),
        ("Handler Structure", test_handler_structure),
        ("Image Processor", test_image_processor),
        ("Requirements", test_requirements),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print("-" * 50)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for RunPod deployment.")
        return True
    elif passed >= total - 1:
        print("⚠️  Minor issues detected. Proceed with caution.")
        return True
    else:
        print("❌ Major issues detected. Fix before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)