#!/usr/bin/env python3
"""
Working RunPod serverless handler based on confirmed working patterns.
Uses Python 3.11 + runpod==1.7.2
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
                "message": "RunPod serverless worker is running",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "timestamp": time.time(),
                    "runpod_status": "working"
                }
            }
        elif task_type == "generate":
            prompt = job_input.get("prompt", "")
            return {
                "success": True,
                "message": f"Generated response for: {prompt}",
                "output": "This would be AI-generated content",
                "metadata": {
                    "processing_time": 1.0,
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
    """Main entry point with comprehensive RunPod testing."""
    logger.info("🚀 Starting Working RunPod Handler")
    logger.info(f"Python version: {sys.version}")
    
    try:
        # Verify RunPod installation
        logger.info("Testing RunPod installation...")
        import runpod
        
        logger.info(f"✅ RunPod imported successfully")
        logger.info(f"RunPod version: {getattr(runpod, '__version__', 'unknown')}")
        logger.info(f"RunPod attributes: {[attr for attr in dir(runpod) if not attr.startswith('_')]}")
        
        # Test serverless availability
        if hasattr(runpod, 'serverless'):
            logger.info("✅ runpod.serverless found")
            
            if hasattr(runpod.serverless, 'start'):
                logger.info("✅ runpod.serverless.start found")
                logger.info("🎯 Starting RunPod serverless worker...")
                
                # Start the serverless worker
                runpod.serverless.start({"handler": handler})
                
            else:
                logger.error("❌ runpod.serverless.start not found")
                raise Exception("Missing runpod.serverless.start method")
        else:
            logger.error("❌ runpod.serverless not found")
            raise Exception("Missing runpod.serverless module")
            
    except ImportError as e:
        logger.error(f"❌ RunPod import failed: {e}")
        logger.info("Attempting to fix installation...")
        
        import subprocess
        try:
            # Try to fix with stable version
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"])
            subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])  
            subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod==1.7.2"])
            
            # Try again
            import runpod
            logger.info("✅ RunPod fixed and imported")
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