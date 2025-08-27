#!/usr/bin/env python3
"""
Deep debugging script to identify RunPod import issues.
"""

import sys
import os
import subprocess
import importlib
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_python_environment():
    """Debug Python environment in detail."""
    logger.info("=== PYTHON ENVIRONMENT DEBUG ===")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    logger.info(f"User: {os.environ.get('USER', 'NOT SET')}")
    
def debug_package_installation():
    """Debug package installation location and structure."""
    logger.info("=== PACKAGE INSTALLATION DEBUG ===")
    
    # Check pip list
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        logger.info("Installed packages:")
        for line in result.stdout.split('\n')[:10]:  # First 10 lines
            logger.info(f"  {line}")
        
        # Look for runpod specifically
        runpod_lines = [line for line in result.stdout.split('\n') if 'runpod' in line.lower()]
        logger.info(f"RunPod packages: {runpod_lines}")
        
    except Exception as e:
        logger.error(f"Failed to get pip list: {e}")
    
    # Check pip show
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", "runpod"], 
                              capture_output=True, text=True)
        logger.info("RunPod package details:")
        logger.info(result.stdout)
        
    except Exception as e:
        logger.error(f"Failed to get package details: {e}")

def debug_import_mechanism():
    """Debug import mechanism in detail."""
    logger.info("=== IMPORT MECHANISM DEBUG ===")
    
    try:
        # Try to find the module file
        import runpod
        logger.info(f"RunPod module location: {runpod.__file__ if hasattr(runpod, '__file__') else 'NO __file__'}")
        logger.info(f"RunPod module: {runpod}")
        logger.info(f"RunPod dir(): {dir(runpod)}")
        logger.info(f"RunPod __dict__: {runpod.__dict__ if hasattr(runpod, '__dict__') else 'NO __dict__'}")
        logger.info(f"RunPod __all__: {getattr(runpod, '__all__', 'NO __all__')}")
        
        # Check if it's a namespace package
        logger.info(f"RunPod __path__: {getattr(runpod, '__path__', 'NO __path__')}")
        logger.info(f"RunPod __package__: {getattr(runpod, '__package__', 'NO __package__')}")
        
        # Try to manually import submodules
        logger.info("Attempting manual submodule imports...")
        try:
            from runpod import serverless
            logger.info(f"✅ Manual import runpod.serverless successful: {serverless}")
        except ImportError as e:
            logger.error(f"❌ Manual import runpod.serverless failed: {e}")
            
        # Check what's actually in the runpod directory
        if hasattr(runpod, '__file__'):
            runpod_dir = os.path.dirname(runpod.__file__)
            if os.path.exists(runpod_dir):
                logger.info(f"RunPod directory contents: {os.listdir(runpod_dir)}")
                
    except Exception as e:
        logger.error(f"Import debug failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def try_alternative_imports():
    """Try alternative import methods."""
    logger.info("=== ALTERNATIVE IMPORT METHODS ===")
    
    # Method 1: importlib
    try:
        import importlib
        runpod_module = importlib.import_module('runpod')
        logger.info(f"✅ importlib method: {dir(runpod_module)}")
    except Exception as e:
        logger.error(f"❌ importlib method failed: {e}")
    
    # Method 2: __import__
    try:
        runpod_module = __import__('runpod')
        logger.info(f"✅ __import__ method: {dir(runpod_module)}")
    except Exception as e:
        logger.error(f"❌ __import__ method failed: {e}")
    
    # Method 3: Direct sys.modules check
    try:
        if 'runpod' in sys.modules:
            logger.info(f"✅ runpod in sys.modules: {dir(sys.modules['runpod'])}")
        else:
            logger.info("❌ runpod not in sys.modules")
    except Exception as e:
        logger.error(f"❌ sys.modules check failed: {e}")

def clean_reinstall_runpod():
    """Completely clean reinstall of RunPod."""
    logger.info("=== CLEAN REINSTALL ===")
    
    try:
        # Remove from sys.modules if present
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith('runpod')]
        for module in modules_to_remove:
            logger.info(f"Removing {module} from sys.modules")
            del sys.modules[module]
        
        # Uninstall completely
        logger.info("Uninstalling RunPod...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"], 
                      capture_output=True)
        
        # Clear pip cache
        logger.info("Clearing pip cache...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True)
        
        # Install with verbose output
        logger.info("Installing RunPod from GitHub with verbose output...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-v",
            "git+https://github.com/runpod/runpod-python.git"
        ], capture_output=True, text=True)
        
        logger.info(f"Install stdout: {result.stdout[-1000:]}")  # Last 1000 chars
        if result.stderr:
            logger.info(f"Install stderr: {result.stderr[-1000:]}")
        
        logger.info("✅ Clean reinstall completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clean reinstall failed: {e}")
        return False

def main():
    """Main debugging function."""
    logger.info("🔍 DEEP RUNPOD DEBUGGING SESSION")
    
    debug_python_environment()
    debug_package_installation()
    debug_import_mechanism()
    try_alternative_imports()
    
    # Try clean reinstall
    if clean_reinstall_runpod():
        logger.info("Testing after clean reinstall...")
        debug_import_mechanism()
        try_alternative_imports()
    
    logger.info("🔍 DEBUGGING COMPLETE")

if __name__ == "__main__":
    main()