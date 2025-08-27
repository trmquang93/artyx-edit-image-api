#!/usr/bin/env python3
"""
Deployment script for Qwen-Image AI Editing Server to RunPod.
"""

import json
import os
import sys
import subprocess
import time
from typing import Dict, Any, Optional

try:
    import runpod
except ImportError:
    print("Error: runpod package not found. Install with: pip install runpod")
    sys.exit(1)


class RunPodDeployer:
    """Handles deployment to RunPod serverless."""
    
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            print("Error: RUNPOD_API_KEY environment variable not set")
            sys.exit(1)
        
        runpod.api_key = self.api_key
    
    def build_and_push_image(self, image_name: str = "qwen-image-server") -> str:
        """Build Docker image and push to registry."""
        print("Building Docker image...")
        
        # Build image
        build_cmd = [
            "docker", "build",
            "-t", f"{image_name}:latest",
            "."
        ]
        
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Docker build failed: {result.stderr}")
            sys.exit(1)
        
        print("Docker image built successfully")
        
        # For RunPod, you might need to push to a registry
        # This example assumes you're using a local registry or RunPod's registry
        return f"{image_name}:latest"
    
    def create_template(self, image_name: str) -> str:
        """Create RunPod template."""
        print("Creating RunPod template...")
        
        # Load template configuration
        with open("runpod/template.json", "r") as f:
            template_config = json.load(f)
        
        template_config["imageName"] = image_name
        
        try:
            template = runpod.create_template(template_config)
            template_id = template["id"]
            print(f"Template created: {template_id}")
            return template_id
            
        except Exception as e:
            print(f"Failed to create template: {e}")
            sys.exit(1)
    
    def create_endpoint(self, template_id: str, endpoint_name: str = None) -> Dict[str, Any]:
        """Create RunPod serverless endpoint."""
        if not endpoint_name:
            endpoint_name = f"qwen-image-{int(time.time())}"
        
        print(f"Creating endpoint: {endpoint_name}")
        
        endpoint_config = {
            "name": endpoint_name,
            "template_id": template_id,
            "locations": {
                "US": {
                    "workers_min": 0,
                    "workers_max": 3,
                    "idle_timeout": 5,
                    "scaler_type": "QUEUE_DELAY",
                    "scaler_value": 4
                }
            },
            "network_volume_id": None,
            "flashboot": True
        }
        
        try:
            endpoint = runpod.create_endpoint(endpoint_config)
            print(f"Endpoint created successfully!")
            print(f"Endpoint ID: {endpoint['id']}")
            print(f"Endpoint URL: {endpoint.get('url', 'N/A')}")
            return endpoint
            
        except Exception as e:
            print(f"Failed to create endpoint: {e}")
            sys.exit(1)
    
    def test_endpoint(self, endpoint_id: str):
        """Test the deployed endpoint."""
        print("Testing endpoint...")
        
        # Simple health check test
        test_input = {
            "task": "health"
        }
        
        try:
            result = runpod.run_sync(endpoint_id, test_input, timeout=60)
            print("Test result:", result)
            
            if result.get("status") == "healthy":
                print("✅ Endpoint is healthy and ready!")
            else:
                print("⚠️  Endpoint responded but may have issues")
                
        except Exception as e:
            print(f"Endpoint test failed: {e}")
            print("The endpoint may still be initializing. Try testing again in a few minutes.")
    
    def deploy(
        self,
        image_name: str = "qwen-image-server",
        endpoint_name: str = None,
        skip_build: bool = False
    ) -> Dict[str, Any]:
        """Full deployment pipeline."""
        print("🚀 Starting deployment to RunPod...")
        
        # Build and push image (if needed)
        if not skip_build:
            final_image_name = self.build_and_push_image(image_name)
        else:
            final_image_name = f"{image_name}:latest"
            print(f"Skipping build, using existing image: {final_image_name}")
        
        # Create template
        template_id = self.create_template(final_image_name)
        
        # Create endpoint
        endpoint = self.create_endpoint(template_id, endpoint_name)
        
        # Test endpoint
        print("\nWaiting for endpoint to initialize...")
        time.sleep(30)  # Give it time to start
        self.test_endpoint(endpoint["id"])
        
        print("\n✅ Deployment completed successfully!")
        print(f"Endpoint ID: {endpoint['id']}")
        print(f"Template ID: {template_id}")
        
        # Save deployment info
        deployment_info = {
            "endpoint_id": endpoint["id"],
            "template_id": template_id,
            "image_name": final_image_name,
            "deployed_at": time.time()
        }
        
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        print("Deployment info saved to deployment_info.json")
        
        return endpoint


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Qwen-Image server to RunPod")
    parser.add_argument("--image-name", default="qwen-image-server",
                       help="Docker image name")
    parser.add_argument("--endpoint-name", help="Custom endpoint name")
    parser.add_argument("--skip-build", action="store_true",
                       help="Skip Docker build step")
    parser.add_argument("--test-only", help="Test existing endpoint ID")
    
    args = parser.parse_args()
    
    deployer = RunPodDeployer()
    
    if args.test_only:
        deployer.test_endpoint(args.test_only)
    else:
        deployer.deploy(
            image_name=args.image_name,
            endpoint_name=args.endpoint_name,
            skip_build=args.skip_build
        )


if __name__ == "__main__":
    main()