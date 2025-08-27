#!/usr/bin/env python3
"""
Robust startup script with comprehensive error handling and logging.
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path

# Configure logging to both file and stdout
log_file = "/app/startup.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def log_system_info():
    """Log comprehensive system information."""
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Environment variables
    logger.info("=== ENVIRONMENT VARIABLES ===")
    for key, value in os.environ.items():
        if 'RUNPOD' in key or 'LOG' in key or 'TORCH' in key or 'HF' in key:
            logger.info(f"{key}: {value}")
    
    # Python path
    logger.info(f"Python path: {sys.path}")
    
    # Available memory (if possible)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB")
    except ImportError:
        logger.info("psutil not available - cannot check memory")

def test_imports():
    """Test importing required packages."""
    logger.info("=== TESTING IMPORTS ===")
    
    # Test basic imports
    packages = [
        'os', 'sys', 'json', 'time', 'logging',
        'requests', 'base64', 'io'
    ]
    
    for package in packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} - OK")
        except ImportError as e:
            logger.error(f"❌ {package} - FAILED: {e}")
    
    # Test torch
    try:
        import torch
        logger.info(f"✅ torch - Version: {torch.__version__}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   CUDA devices: {torch.cuda.device_count()}")
            logger.info(f"   Current device: {torch.cuda.current_device()}")
            logger.info(f"   Device name: {torch.cuda.get_device_name(0)}")
            
            # Test GPU memory
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_cached = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"   GPU memory allocated: {memory_allocated:.2f} GB")
                logger.info(f"   GPU memory cached: {memory_cached:.2f} GB")
            except Exception as e:
                logger.warning(f"   Could not get GPU memory info: {e}")
    except ImportError as e:
        logger.error(f"❌ torch - FAILED: {e}")
    
    # Test PIL
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow - OK")
    except ImportError as e:
        logger.error(f"❌ PIL/Pillow - FAILED: {e}")
    
    # Test RunPod
    try:
        import runpod
        logger.info("✅ runpod - OK")
        logger.info(f"   Version: {getattr(runpod, '__version__', 'unknown')}")
    except ImportError as e:
        logger.error(f"❌ runpod - FAILED: {e}")
        return False
    
    return True

def test_file_access():
    """Test file system access."""
    logger.info("=== TESTING FILE ACCESS ===")
    
    try:
        # Test write access
        test_file = "/app/test_write.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("✅ File write access - OK")
        
        # Test cache directories
        cache_dirs = [
            os.getenv('TORCH_HOME', '/tmp/.torch'),
            os.getenv('HF_HOME', '/tmp/.huggingface'),
            os.getenv('TRANSFORMERS_CACHE', '/tmp/.transformers')
        ]
        
        for cache_dir in cache_dirs:
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Cache directory {cache_dir} - OK")
            except Exception as e:
                logger.error(f"❌ Cache directory {cache_dir} - FAILED: {e}")
                
    except Exception as e:
        logger.error(f"❌ File access test failed: {e}")

def simple_handler(job):
    """Minimal test handler."""
    logger.info(f"=== JOB RECEIVED ===")
    logger.info(f"Job input: {job}")
    
    job_input = job.get("input", {})
    task = job_input.get("task", "unknown")
    
    return {
        "success": True,
        "task": task,
        "message": "Debug handler working",
        "timestamp": time.time(),
        "environment": {
            "python_version": sys.version.split()[0],
            "cwd": os.getcwd(),
            "runpod_mode": os.getenv('RUNPOD_MODE', 'not_set')
        }
    }

def main():
    """Main startup function."""
    logger.info("=== STARTING DEBUG WORKER ===")
    logger.info(f"Log file: {log_file}")
    
    try:
        # System checks
        log_system_info()
        
        # Import tests
        if not test_imports():
            logger.error("Import tests failed - cannot continue")
            logger.info("Keeping container alive for debugging...")
            while True:
                time.sleep(60)
                logger.info("Container still running...")
        
        # File access tests
        test_file_access()
        
        # Start RunPod worker
        logger.info("=== STARTING RUNPOD WORKER ===")
        import runpod
        
        logger.info("Calling runpod.serverless.start...")
        runpod.serverless.start({
            "handler": simple_handler
        })
        
    except Exception as e:
        logger.error(f"STARTUP FAILED: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Keep container alive for debugging
        logger.info("Keeping container alive for debugging...")
        try:
            while True:
                time.sleep(60)
                logger.info("Debug worker still alive...")
                logger.info(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in keep-alive loop: {e}")

if __name__ == "__main__":
    main()