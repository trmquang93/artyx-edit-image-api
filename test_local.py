#!/usr/bin/env python3
"""
Local test for RunPod handler functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runpod_handler import handler

def test_handler():
    """Test the RunPod handler function locally."""
    print("🧪 Testing RunPod Handler Locally")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    health_job = {"input": {"task": "health"}}
    result = handler(health_job)
    print(f"Health check result: {result}")
    
    # Test generate task  
    print("\n2. Testing generate task...")
    generate_job = {"input": {"task": "generate", "prompt": "a sunset over mountains"}}
    result = handler(generate_job)
    print(f"Generate result: {result}")
    
    # Test edit task
    print("\n3. Testing edit task...")
    edit_job = {"input": {"task": "edit", "image": "base64_placeholder"}}
    result = handler(edit_job)
    print(f"Edit result: {result}")
    
    # Test unknown task
    print("\n4. Testing unknown task...")
    unknown_job = {"input": {"task": "unknown"}}
    result = handler(unknown_job)
    print(f"Unknown task result: {result}")
    
    print("\n✅ All handler tests completed successfully!")

if __name__ == "__main__":
    test_handler()