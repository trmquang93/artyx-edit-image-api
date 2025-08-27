#!/usr/bin/env python3
"""
Local test script to verify asyncio handling works correctly.
Tests the run_async_in_sync function before deploying.
"""

import asyncio
import time
import concurrent.futures


def run_async_in_sync(coro):
    """Run async function in sync context, handling existing event loops."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run in thread pool to avoid nested event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


async def mock_async_function(message="test", delay=0.1):
    """Mock async function to test with."""
    await asyncio.sleep(delay)
    return f"Async result: {message}"


def test_no_event_loop():
    """Test when no event loop is running."""
    print("🧪 Test 1: No existing event loop")
    try:
        result = run_async_in_sync(mock_async_function("no_loop"))
        print(f"✅ Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_within_event_loop():
    """Test when already within an event loop."""
    print("🧪 Test 2: Within existing event loop")
    try:
        result = run_async_in_sync(mock_async_function("within_loop"))
        print(f"✅ Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_multiple_calls():
    """Test multiple sequential calls."""
    print("🧪 Test 3: Multiple sequential calls")
    try:
        results = []
        for i in range(3):
            result = run_async_in_sync(mock_async_function(f"call_{i}"))
            results.append(result)
        
        print(f"✅ Results: {results}")
        return len(results) == 3
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_error_handling():
    """Test error handling in async functions."""
    print("🧪 Test 4: Error handling")
    
    async def failing_function():
        await asyncio.sleep(0.1)
        raise ValueError("Test error")
    
    try:
        result = run_async_in_sync(failing_function())
        print(f"❌ Should have failed but got: {result}")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
        return True
    except Exception as e:
        print(f"❌ Wrong error type: {e}")
        return False


def test_mock_health_check():
    """Mock the health check functionality."""
    print("🧪 Test 5: Mock health check")
    
    async def mock_get_health_info():
        # Simulate what the real health check does
        await asyncio.sleep(0.05)  # Small delay to simulate work
        return {
            "model_loaded": False,
            "gpu_available": False
        }
    
    try:
        start_time = time.time()
        health_info = run_async_in_sync(mock_get_health_info())
        elapsed = time.time() - start_time
        
        print(f"✅ Health info: {health_info}")
        print(f"✅ Elapsed time: {elapsed:.3f}s")
        
        return isinstance(health_info, dict) and "model_loaded" in health_info
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing asyncio fix for RunPod handler")
    print("=" * 50)
    
    tests = [
        ("No Event Loop", test_no_event_loop),
        ("Multiple Calls", test_multiple_calls),
        ("Error Handling", test_error_handling), 
        ("Mock Health Check", test_mock_health_check),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    
    # Test within event loop scenario
    print(f"\nWithin Event Loop:")
    try:
        result = asyncio.run(test_within_event_loop())
        if result:
            passed += 1
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/5 tests passed")
    
    if passed == 5:
        print("🎉 All tests passed! AsyncIO fix is working correctly.")
        print("✅ Ready to deploy to RunPod")
        return True
    else:
        print("❌ Some tests failed. Fix before deploying.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)