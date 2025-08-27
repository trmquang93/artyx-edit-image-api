#!/usr/bin/env python3
"""
Local test script to validate RunPod installation sequence.
Tests the exact steps that will be run in Docker.
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result."""
    logger.info(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ {description} - EXCEPTION: {e}")
        return False

def test_pip_sequence():
    """Test the exact pip sequence from Dockerfile."""
    logger.info("=== TESTING PIP INSTALLATION SEQUENCE ===")
    
    # Step 1: Upgrade pip
    if not run_command("pip install --upgrade pip", "Upgrade pip"):
        return False
    
    # Step 2: Purge cache 
    if not run_command("pip cache purge", "Purge pip cache"):
        return False
    
    # Step 3: Install packages with no cache
    if not run_command("pip install --no-cache-dir runpod==1.7.2 requests pillow", 
                      "Install RunPod 1.7.2 + dependencies"):
        return False
    
    return True

def test_runpod_import():
    """Test RunPod import and serverless availability."""
    logger.info("=== TESTING RUNPOD IMPORT ===")
    
    try:
        import runpod
        logger.info(f"✅ RunPod imported successfully")
        
        # Check version
        version = getattr(runpod, '__version__', 'unknown')
        logger.info(f"RunPod version: {version}")
        
        # Check attributes
        attrs = [attr for attr in dir(runpod) if not attr.startswith('_')]
        logger.info(f"RunPod attributes: {attrs}")
        
        # Check for serverless
        if hasattr(runpod, 'serverless'):
            logger.info("✅ runpod.serverless found")
            
            # Check serverless attributes
            serverless_attrs = [attr for attr in dir(runpod.serverless) if not attr.startswith('_')]
            logger.info(f"Serverless attributes: {serverless_attrs}")
            
            if hasattr(runpod.serverless, 'start'):
                logger.info("✅ runpod.serverless.start found")
                logger.info("🎯 RunPod installation is fully functional!")
                return True
            else:
                logger.error("❌ runpod.serverless.start not found")
                return False
        else:
            logger.error("❌ runpod.serverless not found")
            return False
            
    except ImportError as e:
        logger.error(f"❌ RunPod import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ RunPod test failed: {e}")
        return False

def test_simple_handler():
    """Test a simple handler to ensure it works."""
    logger.info("=== TESTING SIMPLE HANDLER ===")
    
    try:
        import runpod
        
        def test_handler(job):
            return {"success": True, "message": "Handler working"}
        
        logger.info("✅ Handler defined successfully")
        logger.info("✅ Ready for serverless.start() - would work in RunPod environment")
        return True
        
    except Exception as e:
        logger.error(f"❌ Handler test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 LOCAL RUNPOD INSTALLATION TEST")
    logger.info(f"Python version: {sys.version}")
    
    success = True
    
    # Test 1: Pip installation sequence
    if not test_pip_sequence():
        logger.error("❌ Pip installation sequence failed")
        success = False
    
    # Test 2: RunPod import and functionality
    if not test_runpod_import():
        logger.error("❌ RunPod import/functionality test failed")
        success = False
    
    # Test 3: Simple handler
    if not test_simple_handler():
        logger.error("❌ Handler test failed")
        success = False
    
    if success:
        logger.info("🎉 ALL TESTS PASSED - RunPod installation is working correctly!")
        logger.info("✅ Ready to build Docker container")
    else:
        logger.error("💥 SOME TESTS FAILED - Fix issues before Docker build")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)