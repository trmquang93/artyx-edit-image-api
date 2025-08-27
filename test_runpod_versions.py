#!/usr/bin/env python3
"""
Test multiple RunPod versions to find one that actually works.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_runpod_version(version):
    """Test a specific RunPod version."""
    logger.info(f"🧪 Testing RunPod version: {version}")
    
    try:
        # Uninstall current version
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "runpod", "-y"], 
                      capture_output=True)
        
        # Install specific version
        if version == "latest":
            install_cmd = [sys.executable, "-m", "pip", "install", "runpod"]
        elif version.startswith("git+"):
            install_cmd = [sys.executable, "-m", "pip", "install", version]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", f"runpod=={version}"]
        
        result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"❌ Installation failed for {version}")
            logger.error(f"Error: {result.stderr}")
            return False
        
        # Test import
        try:
            import importlib
            # Force reload if already imported
            if 'runpod' in sys.modules:
                importlib.reload(sys.modules['runpod'])
            
            import runpod
            
            # Get version info
            version_str = getattr(runpod, '__version__', 'unknown')
            attrs = [attr for attr in dir(runpod) if not attr.startswith('_')]
            has_serverless = hasattr(runpod, 'serverless')
            
            logger.info(f"Version: {version_str}")
            logger.info(f"Attributes: {attrs}")
            logger.info(f"Has serverless: {has_serverless}")
            
            if has_serverless and hasattr(runpod.serverless, 'start'):
                logger.info(f"✅ {version} - WORKING! Has serverless.start()")
                return True
            else:
                logger.warning(f"⚠️ {version} - Missing serverless functionality")
                return False
                
        except Exception as e:
            logger.error(f"❌ Import failed for {version}: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed for {version}: {e}")
        return False

def main():
    """Test multiple versions to find working ones."""
    logger.info("🔍 TESTING MULTIPLE RUNPOD VERSIONS")
    
    # Versions to test (from newest to oldest)
    versions_to_test = [
        "1.7.13",  # Latest
        "1.7.6",   # Mid-range
        "1.7.2",   # What we tried
        "1.6.2",   # Older stable
        "1.6.1",   # Even older
        "1.5.1",   # Much older
        "git+https://github.com/runpod/runpod-python.git",  # Latest GitHub
    ]
    
    working_versions = []
    
    for version in versions_to_test:
        logger.info(f"\n{'='*50}")
        if test_runpod_version(version):
            working_versions.append(version)
            logger.info(f"🎉 FOUND WORKING VERSION: {version}")
        
        # Break early if we find a working version
        if working_versions:
            logger.info(f"\n✅ SUCCESS! Working version found: {version}")
            logger.info("Stopping search - we have a working version")
            break
    
    if working_versions:
        logger.info(f"\n🎯 RECOMMENDATION: Use RunPod version {working_versions[0]}")
        return working_versions[0]
    else:
        logger.error("\n💥 NO WORKING VERSIONS FOUND")
        logger.error("This suggests a fundamental compatibility issue")
        return None

if __name__ == "__main__":
    working_version = main()
    if working_version:
        print(f"\nUse this version in Dockerfile: runpod=={working_version}")
        sys.exit(0)
    else:
        print("\nNo working version found!")
        sys.exit(1)