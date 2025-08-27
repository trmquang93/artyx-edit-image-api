#!/usr/bin/env python3
"""
Simple test to validate RunPod import and serverless functionality.
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_runpod_import():
    """Test RunPod import and serverless access."""
    
    logger.info("=== RUNPOD IMPORT TEST ===")
    logger.info(f"Python version: {sys.version}")
    
    try:
        # Test 1: Basic import
        logger.info("Testing basic import...")
        import runpod
        logger.info("✅ import runpod - SUCCESS")
        
        # Test 2: Check version
        version = getattr(runpod, '__version__', 'unknown')
        logger.info(f"RunPod version: {version}")
        
        # Test 3: List attributes
        attrs = [attr for attr in dir(runpod) if not attr.startswith('_')]
        logger.info(f"RunPod public attributes: {attrs}")
        
        # Test 4: Check for serverless
        if hasattr(runpod, 'serverless'):
            logger.info("✅ runpod.serverless - FOUND")
            
            # Test 5: Check serverless attributes
            serverless_attrs = [attr for attr in dir(runpod.serverless) if not attr.startswith('_')]
            logger.info(f"Serverless attributes: {serverless_attrs}")
            
            # Test 6: Check for start method
            if hasattr(runpod.serverless, 'start'):
                logger.info("✅ runpod.serverless.start - FOUND")
                return True
            else:
                logger.error("❌ runpod.serverless.start - NOT FOUND")
                return False
        else:
            logger.error("❌ runpod.serverless - NOT FOUND")
            return False
            
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def reinstall_runpod():
    """Reinstall RunPod package."""
    logger.info("=== REINSTALLING RUNPOD ===")
    
    try:
        # Uninstall
        logger.info("Uninstalling RunPod...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"], 
                            capture_output=True, text=True)
        
        # Install
        logger.info("Installing RunPod...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod>=1.7.0"], 
                            capture_output=True, text=True)
        
        logger.info("✅ RunPod reinstalled successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reinstall failed: {e}")
        return False

def main():
    """Main test function."""
    
    # Test current installation
    if test_runpod_import():
        logger.info("🎉 RunPod is working correctly!")
        return
    
    # Try reinstall
    logger.info("Attempting to fix by reinstalling...")
    if reinstall_runpod():
        if test_runpod_import():
            logger.info("🎉 RunPod fixed and working!")
        else:
            logger.error("💥 RunPod still not working after reinstall")
    else:
        logger.error("💥 Could not reinstall RunPod")

if __name__ == "__main__":
    main()