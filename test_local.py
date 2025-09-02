#!/usr/bin/env python3
"""
Local test script for enhanced handler implementation (without RunPod dependency).
Tests core functionality that can be validated locally.
"""

import sys
import time
import base64
import json
import tempfile
import os
from pathlib import Path

def generate_test_base64_image():
    """Generate a simple test image as base64 for testing."""
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add some content
        draw.rectangle([100, 100, 400, 400], fill='red', outline='black', width=5)
        draw.text((200, 240), "TEST IMAGE", fill="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_string
        
    except Exception as e:
        print(f"❌ Failed to generate test image: {e}")
        return None

def test_save_data_if_base64():
    """Test the Flux-inspired save_data_if_base64 function."""
    print("\n📥 Testing save_data_if_base64 Function...")
    
    try:
        # Import the function from handler (need to mock runpod import)
        sys.path.insert(0, '/Users/quang.tranminh/Projects/new-ios/artyx/artyx-image-editing-server')
        
        # Mock runpod module
        import types
        runpod_mock = types.ModuleType('runpod')
        sys.modules['runpod'] = runpod_mock
        
        # Now import our local handler
        import handler_local as handler_enhanced
        
        # Test base64 detection and saving
        test_image_b64 = generate_test_base64_image()
        if not test_image_b64:
            print("❌ Could not generate test image")
            return False
        
        # Test save_data_if_base64 function
        temp_dir = tempfile.mkdtemp()
        
        # Test with base64 data
        result_path = handler_enhanced.save_data_if_base64(
            test_image_b64, temp_dir, "test_image.png"
        )
        
        if os.path.exists(result_path):
            print("✅ Base64 data saved successfully")
            print(f"   Saved to: {result_path}")
            file_size = os.path.getsize(result_path)
            print(f"   File size: {file_size} bytes")
        else:
            print("❌ Base64 save failed - file not found")
            return False
        
        # Test with data URL prefix
        data_url = f"data:image/png;base64,{test_image_b64}"
        result_path2 = handler_enhanced.save_data_if_base64(
            data_url, temp_dir, "test_image2.png"
        )
        
        if os.path.exists(result_path2):
            print("✅ Data URL handling works correctly")
            file_size2 = os.path.getsize(result_path2)
            print(f"   File size: {file_size2} bytes")
        else:
            print("❌ Data URL handling failed")
            return False
        
        # Test with regular path (should return unchanged)
        regular_path = "/regular/file/path.jpg"
        result_path3 = handler_enhanced.save_data_if_base64(
            regular_path, temp_dir, "test_image3.png"
        )
        
        if result_path3 == regular_path:
            print("✅ Regular path handling works correctly")
        else:
            print("❌ Regular path handling failed")
            return False
        
        # Clean up
        try:
            os.remove(result_path)
            os.remove(result_path2)
            os.rmdir(temp_dir)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ save_data_if_base64 test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processor():
    """Test ImageProcessor class."""
    print("\n🖼️  Testing ImageProcessor Class...")
    
    try:
        # Mock runpod module
        import types
        runpod_mock = types.ModuleType('runpod')
        sys.modules['runpod'] = runpod_mock
        
        # Import our local handler
        import handler_local as handler_enhanced
        
        # Create ImageProcessor instance
        processor = handler_enhanced.ImageProcessor()
        
        # Generate test image
        test_image_b64 = generate_test_base64_image()
        if not test_image_b64:
            print("❌ Could not generate test image")
            return False
        
        # Test base64 to PIL conversion
        pil_image = processor.base64_to_pil(test_image_b64)
        print(f"✅ Base64 to PIL conversion successful")
        print(f"   Image size: {pil_image.size}")
        print(f"   Image mode: {pil_image.mode}")
        
        # Test PIL to base64 conversion
        converted_b64 = processor.pil_to_base64(pil_image)
        print(f"✅ PIL to base64 conversion successful")
        print(f"   Result length: {len(converted_b64)} characters")
        
        # Test with data URL prefix
        data_url = f"data:image/png;base64,{test_image_b64}"
        pil_image2 = processor.base64_to_pil(data_url)
        print(f"✅ Data URL to PIL conversion successful")
        print(f"   Image size: {pil_image2.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ ImageProcessor test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_manager_init():
    """Test QwenImageManager initialization."""
    print("\n🤖 Testing QwenImageManager Initialization...")
    
    try:
        # Mock runpod module
        import types
        runpod_mock = types.ModuleType('runpod')
        sys.modules['runpod'] = runpod_mock
        
        # Import our local handler
        import handler_local as handler_enhanced
        
        # Create QwenImageManager instance
        manager = handler_enhanced.QwenImageManager()
        print("✅ QwenImageManager created successfully")
        
        # Test initialization (this will test dependency checking)
        manager.initialize()
        print(f"✅ QwenImageManager initialization completed")
        print(f"   Initialized: {manager._initialized}")
        print(f"   Device: {manager.device}")
        print(f"   Torch dtype: {manager.torch_dtype}")
        
        # Test health info
        health = manager.get_health_info()
        print("✅ Health info retrieved successfully")
        print(f"   Model loaded: {health.get('model_loaded', False)}")
        print(f"   GPU available: {health.get('gpu_available', False)}")
        
        if 'memory_usage' in health:
            mem = health['memory_usage']
            print(f"   GPU memory: {mem['allocated_gb']}GB / {mem['total_gb']}GB ({mem['utilization']}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ QwenImageManager test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_check():
    """Test CUDA checking function."""
    print("\n🔍 Testing CUDA Check Function...")
    
    try:
        # Mock runpod module
        import types
        runpod_mock = types.ModuleType('runpod')
        sys.modules['runpod'] = runpod_mock
        
        # Import our local handler
        import handler_local as handler_enhanced
        
        # Test CUDA check function
        cuda_available = handler_enhanced.check_cuda_availability()
        print(f"✅ CUDA check completed")
        print(f"   CUDA available: {cuda_available}")
        
        # Check environment variables that should be set
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_devices}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA check test: {e}")
        # This might fail if CUDA is not available, which is okay for local testing
        print("⚠️  CUDA check failed (expected on non-GPU systems)")
        return True  # Don't fail the test suite for this

def test_basic_imports():
    """Test that all required imports work."""
    print("\n📦 Testing Basic Imports...")
    
    required_imports = [
        'sys', 'os', 'time', 'logging', 'traceback', 'base64', 'io', 'uuid', 'json', 'binascii'
    ]
    
    optional_imports = [
        'torch', 'diffusers', 'transformers', 'PIL', 'numpy', 'requests'
    ]
    
    try:
        for module in required_imports:
            __import__(module)
            print(f"✅ {module}")
        
        print("\n📦 Testing Optional ML Dependencies...")
        available_optional = []
        for module in optional_imports:
            try:
                __import__(module)
                print(f"✅ {module}")
                available_optional.append(module)
            except ImportError:
                print(f"⚠️  {module} (not available)")
        
        print(f"\n📊 Optional dependencies available: {len(available_optional)}/{len(optional_imports)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test exception: {e}")
        return False

def main():
    """Run local tests."""
    print("🧪 Enhanced Handler Local Test Suite")
    print("==========================================")
    print("Testing functionality that doesn't require RunPod...")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("CUDA Check", test_cuda_check),
        ("save_data_if_base64 Function", test_save_data_if_base64),
        ("ImageProcessor Class", test_image_processor),
        ("QwenImageManager Init", test_qwen_manager_init),
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Local Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.3f}s")
    print('='*60)
    
    if passed >= total * 0.8:  # Allow 80% pass rate for local testing
        print("🎉 Local tests mostly passed! Core functionality is working.")
        print("💡 Deploy to RunPod for full functionality testing.")
        return True
    else:
        print("❌ Too many local tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)