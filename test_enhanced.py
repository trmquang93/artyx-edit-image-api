#!/usr/bin/env python3
"""
Test script for enhanced handler implementation.
Tests all major functionality and compares with original implementation.
"""

import sys
import time
import base64
import json
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

def test_health_check():
    """Test health check endpoint."""
    print("\n🏥 Testing Health Check...")
    
    try:
        import handler_enhanced
        
        test_job = {
            "input": {
                "task": "health"
            }
        }
        
        start_time = time.time()
        result = handler_enhanced.handler(test_job)
        execution_time = time.time() - start_time
        
        print(f"⏱️  Health check took: {execution_time:.3f}s")
        
        if result.get("success"):
            print("✅ Health check passed")
            env = result.get("environment", {})
            print(f"   Server type: {env.get('server_type', 'unknown')}")
            print(f"   GPU available: {env.get('gpu_available', False)}")
            print(f"   Model loaded: {env.get('model_loaded', False)}")
            print(f"   Python version: {env.get('python_version', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Health check exception: {e}")
        return False

def test_image_generation():
    """Test text-to-image generation."""
    print("\n🎨 Testing Image Generation...")
    
    try:
        import handler_enhanced
        
        test_job = {
            "input": {
                "task": "generate",
                "prompt": "a beautiful landscape with mountains and a lake",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,  # Reduced for faster testing
                "guidance_scale": 4.0,
                "seed": 42
            }
        }
        
        start_time = time.time()
        result = handler_enhanced.handler(test_job)
        execution_time = time.time() - start_time
        
        print(f"⏱️  Generation took: {execution_time:.3f}s")
        
        if result.get("success"):
            print("✅ Image generation successful")
            metadata = result.get("metadata", {})
            print(f"   Model: {metadata.get('model', 'unknown')}")
            print(f"   Dimensions: {metadata.get('dimensions', 'unknown')}")
            print(f"   Processing time: {metadata.get('processing_time', 0):.3f}s")
            print(f"   Image size: {len(result.get('image', ''))} characters")
            return True
        else:
            print(f"❌ Image generation failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Image generation exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_editing():
    """Test image editing functionality."""
    print("\n🖼️  Testing Image Editing...")
    
    # Generate test image
    test_image_b64 = generate_test_base64_image()
    if not test_image_b64:
        print("❌ Could not generate test image for editing")
        return False
    
    try:
        import handler_enhanced
        
        test_job = {
            "input": {
                "task": "edit",
                "image": test_image_b64,
                "prompt": "change background to a sunset beach scene",
                "num_inference_steps": 20,  # Reduced for faster testing
                "guidance_scale": 4.0,
                "strength": 0.8,
                "seed": 42
            }
        }
        
        start_time = time.time()
        result = handler_enhanced.handler(test_job)
        execution_time = time.time() - start_time
        
        print(f"⏱️  Editing took: {execution_time:.3f}s")
        
        if result.get("success"):
            print("✅ Image editing successful")
            metadata = result.get("metadata", {})
            print(f"   Model: {metadata.get('model', 'unknown')}")
            print(f"   Processing time: {metadata.get('processing_time', 0):.3f}s")
            print(f"   Strength: {metadata.get('strength', 'unknown')}")
            print(f"   Result size: {len(result.get('image', ''))} characters")
            return True
        else:
            print(f"❌ Image editing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Image editing exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base64_handling():
    """Test Flux-inspired base64 handling functions."""
    print("\n📥 Testing Base64 Handling...")
    
    try:
        import handler_enhanced
        import tempfile
        import os
        
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
        
        # Test with regular path (should return unchanged)
        regular_path = "/regular/file/path.jpg"
        result_path2 = handler_enhanced.save_data_if_base64(
            regular_path, temp_dir, "test_image2.png"
        )
        
        if result_path2 == regular_path:
            print("✅ Regular path handling works correctly")
        else:
            print("❌ Regular path handling failed")
            return False
        
        # Clean up
        try:
            os.remove(result_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Base64 handling test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_endpoint():
    """Test debug endpoint."""
    print("\n🐛 Testing Debug Endpoint...")
    
    try:
        import handler_enhanced
        
        test_job = {
            "input": {
                "task": "debug"
            }
        }
        
        start_time = time.time()
        result = handler_enhanced.handler(test_job)
        execution_time = time.time() - start_time
        
        print(f"⏱️  Debug took: {execution_time:.3f}s")
        
        if result.get("success"):
            print("✅ Debug endpoint successful")
            print(f"   Handler version: {result.get('handler_version', 'unknown')}")
            print(f"   Python version: {result.get('python_version', 'unknown')[:50]}...")
            print(f"   Installed packages: {len(result.get('installed_packages', ''))} characters")
            
            env_vars = result.get("environment_vars", {})
            print(f"   CUDA_VISIBLE_DEVICES: {env_vars.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"   FORCE_CUDA: {env_vars.get('FORCE_CUDA', 'not set')}")
            
            return True
        else:
            print(f"❌ Debug endpoint failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Debug endpoint exception: {e}")
        return False

def run_performance_comparison():
    """Compare performance with original handler if available."""
    print("\n⚡ Performance Comparison...")
    
    try:
        # Try to import original handler for comparison
        original_available = False
        try:
            import runpod_handler
            original_available = True
            print("✅ Original handler available for comparison")
        except ImportError:
            print("⚠️  Original handler not available for comparison")
        
        import handler_enhanced
        
        # Test health check performance
        test_job = {"input": {"task": "health"}}
        
        # Test new handler
        start_time = time.time()
        new_result = handler_enhanced.handler(test_job)
        new_time = time.time() - start_time
        
        print(f"🆕 New handler health check: {new_time:.3f}s")
        
        if original_available:
            # Test original handler
            try:
                start_time = time.time()
                old_result = runpod_handler.handler(test_job)
                old_time = time.time() - start_time
                
                print(f"📊 Original handler health check: {old_time:.3f}s")
                print(f"🏁 Performance ratio: {old_time/new_time:.2f}x")
                
                if new_time < old_time:
                    print("✅ New handler is faster!")
                else:
                    print("⚠️  Original handler is faster")
                
            except Exception as e:
                print(f"❌ Original handler test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Enhanced Handler Test Suite")
    print("===================================")
    
    tests = [
        ("Health Check", test_health_check),
        ("Base64 Handling", test_base64_handling),
        ("Debug Endpoint", test_debug_endpoint),
        ("Image Generation", test_image_generation),
        ("Image Editing", test_image_editing),
        ("Performance Comparison", run_performance_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.3f}s")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Enhanced implementation is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()