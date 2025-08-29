#!/usr/bin/env python3
"""
Quick deployment script to update RunPod endpoint with network volume optimization.
This script builds and deploys the disk space optimized version.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False

def main():
    """Main deployment function."""
    print("🚀 RunPod Network Volume Optimization Deployment")
    print("=" * 60)
    
    # Build timestamp for unique tagging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = "ghcr.io/trmquang93/artyx-edit-image-api"
    
    # Step 1: Build optimized Docker image
    print(f"\n📦 Building optimized Docker image...")
    build_cmd = f"docker build -f Dockerfile.optimized -t {image_name}:network-volume-{timestamp} -t {image_name}:latest ."
    
    if not run_command(build_cmd, "Docker build"):
        print("❌ Build failed. Check Docker installation and permissions.")
        sys.exit(1)
    
    # Step 2: Push to registry
    print(f"\n📤 Pushing image to registry...")
    
    # Push timestamped version
    push_cmd_timestamp = f"docker push {image_name}:network-volume-{timestamp}"
    if not run_command(push_cmd_timestamp, f"Push timestamped version"):
        print("❌ Push failed. Check registry authentication.")
        sys.exit(1)
    
    # Push latest
    push_cmd_latest = f"docker push {image_name}:latest"
    if not run_command(push_cmd_latest, "Push latest tag"):
        print("❌ Push failed. Check registry authentication.")
        sys.exit(1)
    
    # Step 3: Display completion message
    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"""
🐳 Docker Images Built and Pushed:
   • {image_name}:network-volume-{timestamp}
   • {image_name}:latest

📝 Next Steps in RunPod Console:

1. Update Environment Variables:
   HF_HOME=/runpod-volume/.huggingface
   TRANSFORMERS_CACHE=/runpod-volume/.transformers
   TORCH_HOME=/runpod-volume/.torch
   PORT=80
   PORT_HEALTH=80

2. Ensure Network Volume is attached (200GB recommended)

3. Redeploy your serverless endpoint with the updated image

4. Monitor logs for disk usage reports:
   "Container disk: X.XGB free / 50.0GB total"
   "Network volume: X.XGB free / 200.0GB total"

🎯 Expected Results:
   • Models cache to network volume (not container disk)
   • No more "No space left on device" errors
   • Faster subsequent startups (models persist)
   
📚 Full documentation: RUNPOD_NETWORK_VOLUME_SETUP.md
""")

if __name__ == "__main__":
    main()