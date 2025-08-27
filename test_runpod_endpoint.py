#!/usr/bin/env python3
"""
Test script to verify RunPod endpoint is working.
"""

import requests
import json
import time

# RunPod endpoint from previous conversation
ENDPOINT_URL = "https://api.runpod.ai/v2/25v1qttrao3ikz/run"

# You'll need to set your RunPod API key
API_KEY = "YOUR_RUNPOD_API_KEY"  # Replace with your actual API key

def test_runpod_endpoint(task_type="health", **kwargs):
    """Test RunPod endpoint with a specific task."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "input": {
            "task": task_type,
            **kwargs
        }
    }
    
    print(f"🧪 Testing RunPod endpoint: {task_type}")
    print(f"📡 Endpoint: {ENDPOINT_URL}")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Error Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Run all endpoint tests."""
    print("🚀 Testing RunPod AI Image Editing Server Endpoint")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    health_result = test_runpod_endpoint("health")
    
    # Test 2: Generate task
    print("\n2. Testing generate task...")
    generate_result = test_runpod_endpoint("generate", prompt="a beautiful sunset over mountains")
    
    # Test 3: Edit task
    print("\n3. Testing edit task...")
    edit_result = test_runpod_endpoint("edit", image="base64_placeholder", prompt="make it more colorful")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"Health Check: {'✅ PASS' if health_result else '❌ FAIL'}")
    print(f"Generate Task: {'✅ PASS' if generate_result else '❌ FAIL'}")
    print(f"Edit Task: {'✅ PASS' if edit_result else '❌ FAIL'}")
    
    if all([health_result, generate_result, edit_result]):
        print("\n🎉 All tests passed! RunPod endpoint is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the deployment in RunPod console.")

if __name__ == "__main__":
    main()