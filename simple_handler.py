#!/usr/bin/env python3
"""
Ultra simple RunPod handler without complex imports.
Based on RunPod's simplest working examples.
"""

import os
import json
import time

def handler(job):
    """Simple test handler that doesn't rely on complex imports."""
    print("=== JOB RECEIVED ===")
    print(f"Job: {json.dumps(job, indent=2)}")
    
    try:
        job_input = job.get("input", {})
        task_type = job_input.get("task", "unknown")
        
        print(f"Task type: {task_type}")
        
        if task_type == "health":
            return {
                "success": True,
                "status": "healthy", 
                "message": "Simple handler is working",
                "timestamp": time.time(),
                "python_version": "3.10.13",
                "container_status": "running"
            }
        else:
            return {
                "success": False,
                "error": f"Task '{task_type}' not supported",
                "supported_tasks": ["health"]
            }
            
    except Exception as e:
        print(f"Handler error: {e}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}"
        }

# Try to start RunPod with minimal configuration
if __name__ == "__main__":
    print("🚀 Starting ultra simple RunPod handler...")
    
    try:
        # Try installing a known working version first
        import subprocess
        import sys
        
        print("Installing known working RunPod version...")
        
        # Try specific version that's known to work
        versions_to_try = [
            "1.6.2",  # Older stable version
            "1.5.1",  # Even older
            "git+https://github.com/runpod/runpod-python.git@v1.6.2"
        ]
        
        for version in versions_to_try:
            try:
                print(f"Trying RunPod version: {version}")
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"], 
                             capture_output=True)
                result = subprocess.run([sys.executable, "-m", "pip", "install", f"runpod=={version}"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Successfully installed {version}")
                    
                    # Test import
                    import runpod
                    print(f"RunPod attributes: {dir(runpod)}")
                    
                    if hasattr(runpod, 'serverless'):
                        print("✅ serverless module found!")
                        runpod.serverless.start({"handler": handler})
                        break
                    else:
                        print("❌ serverless module not found, trying next version...")
                        continue
                        
            except Exception as e:
                print(f"❌ Version {version} failed: {e}")
                continue
        
        print("❌ All versions failed, keeping container alive...")
        while True:
            time.sleep(60)
            print(f"Container still alive... {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        print(f"Startup failed: {e}")
        print("Keeping container alive for debugging...")
        while True:
            time.sleep(60)