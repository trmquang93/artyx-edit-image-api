#!/usr/bin/env python3
"""
Simple debug worker without ML dependencies for testing RunPod integration.
"""

import os
import sys
import time
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check the environment and log important info."""
    logger.info("=== SIMPLE DEBUG WORKER ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"RUNPOD_MODE: {os.getenv('RUNPOD_MODE', 'NOT SET')}")
    logger.info(f"Files in current directory: {os.listdir('.')}")

def simple_handler(job):
    """Simple test handler."""
    logger.info("=== JOB RECEIVED ===")
    logger.info(f"Job: {job}")
    
    try:
        job_input = job.get("input", {})
        task_type = job_input.get("task", "unknown")
        
        logger.info(f"Task type: {task_type}")
        
        if task_type == "health":
            return {
                "success": True,
                "status": "healthy",
                "message": "Simple debug worker is running",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "working_directory": os.getcwd(),
                    "runpod_mode": os.getenv('RUNPOD_MODE'),
                    "timestamp": time.time()
                }
            }
        else:
            return {
                "success": False,
                "error": f"Task '{task_type}' not supported in simple debug mode",
                "supported_tasks": ["health"],
                "message": "This is a minimal debug worker for testing RunPod integration"
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}"
        }

def main():
    """Main entry point."""
    try:
        # Check environment
        check_environment()
        
        # Try to import runpod
        logger.info("Importing RunPod...")
        try:
            import runpod
            logger.info("✅ RunPod imported successfully")
            logger.info(f"RunPod version: {getattr(runpod, '__version__', 'unknown')}")
            logger.info(f"RunPod attributes: {[attr for attr in dir(runpod) if not attr.startswith('_')]}")
            
            # Check for serverless functionality
            if hasattr(runpod, 'serverless'):
                logger.info("✅ runpod.serverless found")
            else:
                logger.warning("❌ runpod.serverless not found - checking alternatives")
                
        except ImportError as e:
            logger.error(f"❌ RunPod import failed: {e}, installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "runpod"])
            import runpod
            logger.info("✅ RunPod installed and imported")
        
        logger.info("Starting RunPod serverless worker...")
        
        # Try to access serverless functionality
        try:
            if hasattr(runpod, 'serverless'):
                logger.info("Using runpod.serverless.start() API")
                runpod.serverless.start({"handler": simple_handler})
            else:
                # Force reinstall if serverless module missing
                logger.warning("runpod.serverless not available, reinstalling...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod>=1.7.0"])
                
                # Re-import after reinstall
                import importlib
                importlib.reload(runpod) if 'runpod' in globals() else None
                import runpod
                
                logger.info("✅ RunPod reinstalled")
                logger.info(f"RunPod attributes after reinstall: {[attr for attr in dir(runpod) if not attr.startswith('_')]}")
                
                if hasattr(runpod, 'serverless'):
                    logger.info("✅ Using runpod.serverless.start() after reinstall")
                    runpod.serverless.start({"handler": simple_handler})
                else:
                    raise Exception("runpod.serverless still not available after reinstall")
                    
        except Exception as serverless_error:
            logger.error(f"Serverless start error: {serverless_error}")
            raise
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Keep the container alive for debugging
        logger.info("Keeping container alive for debugging...")
        try:
            while True:
                time.sleep(60)
                logger.info(f"Simple debug worker still alive... {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

if __name__ == "__main__":
    main()