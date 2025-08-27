#!/usr/bin/env python3
"""
Working RunPod handler using direct imports to bypass package exposure issues.
"""

import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handler(job):
    """RunPod serverless handler function."""
    logger.info("=== JOB RECEIVED ===")
    logger.info(f"Job input: {job}")
    
    try:
        job_input = job.get("input", {})
        task_type = job_input.get("task", "unknown")
        
        logger.info(f"Task type: {task_type}")
        
        if task_type == "health":
            return {
                "success": True,
                "status": "healthy",
                "message": "RunPod serverless worker is running successfully!",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "timestamp": time.time(),
                    "runpod_status": "working_with_direct_import"
                }
            }
        elif task_type == "generate":
            prompt = job_input.get("prompt", "")
            return {
                "success": True,
                "message": f"Generated response for: {prompt}",
                "output": "This is a test AI-generated response",
                "metadata": {
                    "processing_time": 1.5,
                    "model": "test-model"
                }
            }
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}",
                "supported_tasks": ["health", "generate"]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}"
        }

def main():
    """Main entry point with direct import solution."""
    logger.info("🚀 Starting RunPod Handler with Direct Import Solution")
    logger.info(f"Python version: {sys.version}")
    
    try:
        # Import runpod and access serverless module
        logger.info("Importing RunPod and accessing serverless module...")
        import runpod
        
        logger.info("✅ RunPod imported")
        serverless = runpod.serverless  # Access the serverless module
        logger.info("✅ Serverless module accessible")
        logger.info(f"Serverless attributes: {[attr for attr in dir(serverless) if not attr.startswith('_')]}")
        
        if hasattr(serverless, 'start'):
            logger.info("✅ runpod.serverless.start found")
            logger.info("🎯 Starting RunPod serverless worker...")
            
            # Start the serverless worker
            runpod.serverless.start({"handler": handler})
            
        else:
            logger.error("❌ runpod.serverless.start not found")
            raise Exception("Missing runpod.serverless.start method")
            
    except ImportError as e:
        logger.error(f"❌ RunPod import failed: {e}")
        
        # Try to install and fix
        logger.info("Attempting to reinstall RunPod...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "runpod"])
            import runpod.serverless
            logger.info("✅ RunPod reinstalled and imported")
            runpod.serverless.start({"handler": handler})
        except Exception as fix_error:
            logger.error(f"❌ Failed to fix RunPod: {fix_error}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Keep container alive for debugging
        logger.info("Keeping container alive for debugging...")
        try:
            while True:
                time.sleep(60)
                logger.info(f"Worker still alive... {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

if __name__ == "__main__":
    main()