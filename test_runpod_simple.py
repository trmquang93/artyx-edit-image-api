#!/usr/bin/env python3
"""
Simple RunPod API test using the official SDK.
"""

import json
import time
import os

# Set API key from environment variable
if not os.getenv("RUNPOD_API_KEY"):
    os.environ["RUNPOD_API_KEY"] = "your-runpod-api-key"

try:
    import runpod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")
    print("✅ RunPod SDK imported successfully")
except ImportError as e:
    print(f"❌ Failed to import runpod: {e}")
    exit(1)

def test_basic_api():
    """Test basic API functionality."""
    try:
        # Try to get pods (basic API test)
        pods = runpod.get_pods()
        print(f"✅ API connection successful! Found {len(pods)} pods.")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def create_simple_endpoint():
    """Create a simple endpoint using RunPod SDK."""
    try:
        print("Creating endpoint with basic configuration...")
        
        # Simple endpoint configuration
        endpoint_config = {
            "name": f"qwen-test-{int(time.time())}",
            "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
            "gpu_count": 1,
            "container_disk_in_gb": 50,
            "env": {
                "LOG_LEVEL": "INFO",
                "TORCH_HOME": "/tmp/.torch"
            }
        }
        
        # Create endpoint using RunPod SDK
        endpoint = runpod.create_endpoint(endpoint_config)
        
        if endpoint:
            print(f"✅ Endpoint created successfully!")
            print(f"   ID: {endpoint.get('id', 'N/A')}")
            print(f"   Name: {endpoint_config['name']}")
            
            # Save info
            with open("simple_endpoint.json", "w") as f:
                json.dump({
                    "endpoint": endpoint,
                    "config": endpoint_config,
                    "created_at": time.time()
                }, f, indent=2)
            
            return endpoint
        else:
            print("❌ Endpoint creation returned None")
            return None
            
    except Exception as e:
        print(f"❌ Endpoint creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_serverless_function():
    """Test running a simple serverless function."""
    try:
        # Simple test function
        def test_handler(event):
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Hello from RunPod!",
                    "input": event.get("input", {}),
                    "timestamp": time.time()
                })
            }
        
        print("Testing serverless handler...")
        
        # Test locally first
        test_event = {
            "input": {
                "test": "data",
                "message": "hello world"
            }
        }
        
        result = test_handler(test_event)
        print(f"✅ Local handler test successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Handler test failed: {e}")
        return False

def main():
    print("🧪 RunPod Simple API Test")
    print("=" * 40)
    
    # Test 1: Basic API connection
    print("1. Testing basic API connection...")
    if not test_basic_api():
        print("⚠️  API test failed, but continuing...")
    print()
    
    # Test 2: Test handler function
    print("2. Testing serverless handler...")
    if test_serverless_function():
        print("✅ Handler test passed")
    print()
    
    # Test 3: Try to create endpoint
    print("3. Attempting to create endpoint...")
    endpoint = create_simple_endpoint()
    
    if endpoint:
        print("\n🎉 Success! Endpoint created.")
        print("Next steps:")
        print("1. Deploy your custom code to the endpoint")
        print("2. Test the endpoint with API calls")
        print("3. Monitor logs and performance")
    else:
        print("\n⚠️  Endpoint creation failed.")
        print("This might be due to:")
        print("- API key permissions")
        print("- Account limits")
        print("- Service availability")
    
    print(f"\nAPI Key (partial): rpa_***{runpod.api_key[-6:]}")

if __name__ == "__main__":
    main()