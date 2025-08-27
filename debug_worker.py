#!/usr/bin/env python3
"""
Debug version of RunPod worker with extensive logging.
"""

import os
import sys
import logging
import traceback
import time

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check the environment and log important info."""
    logger.info("=== ENVIRONMENT CHECK ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python path: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"RUNPOD_MODE: {os.getenv('RUNPOD_MODE', 'NOT SET')}")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'NOT SET')}")
    logger.info(f"TORCH_HOME: {os.getenv('TORCH_HOME', 'NOT SET')}")
    
    # Check if we can import basic packages
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.error(f"Failed to import/check PyTorch: {e}")
    
    try:
        import runpod
        logger.info(f"RunPod package available: {runpod.__version__ if hasattr(runpod, '__version__') else 'version unknown'}")
    except Exception as e:
        logger.error(f"Failed to import RunPod: {e}")

def simple_handler(job):
    """Simple handler for testing."""
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
                "message": "Debug worker is running",
                "environment": {
                    "python_version": sys.version,
                    "working_directory": os.getcwd(),
                    "runpod_mode": os.getenv('RUNPOD_MODE'),
                    "torch_available": False  # Will update if torch works
                }
            }
        else:
            return {
                "success": False,
                "error": f"Task '{task_type}' not supported in debug mode",
                "supported_tasks": ["health"]
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
    logger.info("=== STARTING DEBUG WORKER ===")
    
    try:
        # Check environment
        check_environment()
        
        # Try to import runpod
        logger.info("Importing RunPod...")
        import runpod
        
        logger.info("Starting RunPod serverless worker...")
        runpod.serverless.start({
            "handler": simple_handler
        })
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Keep the container alive for debugging
        logger.info("Keeping container alive for debugging...")
        while True:
            time.sleep(60)
            logger.info("Debug worker still alive...")

if __name__ == "__main__":
    main()