#!/usr/bin/env python3
"""
Deployment script for updated Qwen-Image server to RunPod.
This deploys the real AI implementation with proper Qwen models.
"""

import argparse
import subprocess
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, check=check,
            capture_output=True, text=True
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr.strip()}")
        raise


def build_docker_image(image_name, tag="latest"):
    """Build the Docker image."""
    logger.info("🔨 Building Docker image with real Qwen-Image support...")
    
    try:
        # Build the image
        cmd = f"docker build -t {image_name}:{tag} ."
        run_command(cmd)
        
        logger.info(f"✅ Docker image built successfully: {image_name}:{tag}")
        return True
        
    except subprocess.CalledProcessError:
        logger.error("❌ Docker build failed")
        return False


def push_docker_image(image_name, tag="latest", registry="ghcr.io"):
    """Push the Docker image to a registry."""
    logger.info(f"📤 Pushing Docker image to {registry}...")
    
    try:
        full_name = f"{registry}/{image_name}:{tag}"
        
        # Tag for registry
        run_command(f"docker tag {image_name}:{tag} {full_name}")
        
        # Push to registry
        run_command(f"docker push {full_name}")
        
        logger.info(f"✅ Docker image pushed successfully: {full_name}")
        return full_name
        
    except subprocess.CalledProcessError:
        logger.error("❌ Docker push failed")
        return None


def deploy_to_runpod(image_name, endpoint_id=None):
    """Deploy to RunPod (placeholder - would need RunPod API integration)."""
    logger.info("🚀 Deploying to RunPod...")
    
    logger.info("📋 Manual RunPod Deployment Steps:")
    logger.info("1. Go to RunPod console: https://www.runpod.io/console/serverless")
    logger.info("2. Update your existing endpoint or create a new one")
    logger.info(f"3. Use Docker image: {image_name}")
    logger.info("4. Set environment variables:")
    logger.info("   - SERVER_MODE=runpod")
    logger.info("   - HF_HOME=/tmp/.huggingface")
    logger.info("   - TORCH_HOME=/tmp/.torch")
    logger.info("5. Configure GPU requirements:")
    logger.info("   - Minimum: 24GB VRAM (for Qwen-Image 20B)")
    logger.info("   - Recommended: RTX 4090, A100, or similar")
    logger.info("6. Set timeout: 300s (models need time to load)")
    logger.info("7. Test with a health check request")
    
    if endpoint_id:
        logger.info(f"🎯 Your existing endpoint ID: {endpoint_id}")
        logger.info("   Update this endpoint with the new image")
    else:
        logger.info("💡 If this is a new deployment, create a new serverless endpoint")
    
    logger.info("\n⚠️  Important Notes:")
    logger.info("- First request will be slow (model loading)")
    logger.info("- Real AI processing takes 15-30s vs previous 3-8s")
    logger.info("- Monitor GPU memory usage")
    logger.info("- Test both text-to-image and image editing")


def validate_deployment():
    """Validate the deployment is ready."""
    logger.info("✅ Deployment preparation complete!")
    logger.info("\n📋 Next Steps:")
    logger.info("1. Push Docker image to your registry")
    logger.info("2. Update RunPod endpoint configuration") 
    logger.info("3. Test with the provided test script")
    logger.info("4. Monitor performance and error rates")
    logger.info("\n🧪 To test locally first:")
    logger.info("   python test_qwen_integration.py")


def main():
    """Main deployment workflow."""
    parser = argparse.ArgumentParser(description="Deploy Qwen-Image server to RunPod")
    parser.add_argument("--image-name", default="artyx-qwen-server", 
                        help="Docker image name")
    parser.add_argument("--tag", default="latest", 
                        help="Docker image tag")
    parser.add_argument("--registry", default="ghcr.io/your-username",
                        help="Docker registry (update with your username)")
    parser.add_argument("--endpoint-id", 
                        help="Existing RunPod endpoint ID to update")
    parser.add_argument("--build-only", action="store_true",
                        help="Only build, don't push or deploy")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build, only deploy")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Qwen-Image Server Deployment")
    logger.info(f"Image: {args.image_name}:{args.tag}")
    
    success = True
    
    # Build Docker image
    if not args.skip_build:
        if not build_docker_image(args.image_name, args.tag):
            success = False
    
    # Push to registry (if not build-only)
    pushed_image = None
    if success and not args.build_only:
        pushed_image = push_docker_image(args.image_name, args.tag, args.registry)
        if not pushed_image:
            success = False
    
    # Deploy to RunPod
    if success and not args.build_only:
        deploy_to_runpod(pushed_image or f"{args.image_name}:{args.tag}", args.endpoint_id)
    
    # Final validation
    validate_deployment()
    
    if success:
        logger.info("🎉 Deployment preparation completed successfully!")
        return 0
    else:
        logger.error("💥 Deployment preparation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())