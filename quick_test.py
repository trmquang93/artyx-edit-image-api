#!/usr/bin/env python3
"""
Quick validation script to test basic functionality.
"""

import sys

def test_basic_imports():
    """Test if basic imports work."""
    try:
        print("Testing basic imports...")
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_project_structure():
    """Test project structure imports."""
    try:
        print("\nTesting project structure...")
        from models.qwen_image import QwenImageManager
        print("✅ QwenImageManager import")
        
        from models.image_processor import ImageProcessor
        print("✅ ImageProcessor import")
        
        from runpod_handler import handler
        print("✅ RunPod handler import")
        
        return True
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False

def test_handler_basic():
    """Test basic handler functionality."""
    try:
        print("\nTesting handler basic functionality...")
        from runpod_handler import handler
        
        # Test with simple health check (should work without models)
        result = handler({"input": {"task": "health"}})
        
        if result.get("success"):
            print("✅ Handler health check works")
            return True
        else:
            print(f"❌ Handler failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Handler test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Qwen-Image Test")
    print("=" * 30)
    
    tests = [
        test_basic_imports,
        test_project_structure,
        test_handler_basic,
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📊 Result: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Basic tests passed! Ready for deployment.")
    else:
        print("❌ Some tests failed. Check errors above.")
    
    sys.exit(0 if passed == len(tests) else 1)