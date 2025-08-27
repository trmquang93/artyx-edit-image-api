#!/usr/bin/env python3
"""
Clean RunPod serverless handler for AI image editing using Qwen-Image models.
"""

import sys
import time
import logging
import traceback
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager - will be initialized once
model_manager = None

def handler(job):
    """RunPod serverless handler function."""
    logger.info("=== JOB RECEIVED ===")
    logger.info(f"Job input: {job}")
    
    try:
        global model_manager
        
        # Initialize model manager if not already done
        if model_manager is None:
            logger.info("Initializing Qwen-Image model manager...")
            from models.qwen_image import QwenImageManager
            model_manager = QwenImageManager()
            # Run async initialization in sync context
            asyncio.run(model_manager.initialize())
            logger.info("Model manager initialized successfully")
        
        job_input = job.get("input", {})
        task_type = job_input.get("task", "health")
        
        logger.info(f"Processing task: {task_type}")
        
        if task_type == "health":
            # Get health info from model manager
            health_info = asyncio.run(model_manager.get_health_info())
            return {
                "success": True,
                "status": "healthy",
                "message": "AI Image Editing Server is running successfully!",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "timestamp": time.time(),
                    "server_type": "ai_image_editing",
                    "model_loaded": health_info.get("model_loaded", False),
                    "gpu_available": health_info.get("gpu_available", False)
                }
            }
        elif task_type == "generate":
            prompt = job_input.get("prompt", "a beautiful landscape")
            negative_prompt = job_input.get("negative_prompt", "")
            width = job_input.get("width", 1024)
            height = job_input.get("height", 1024)
            num_inference_steps = job_input.get("num_inference_steps", 30)
            guidance_scale = job_input.get("guidance_scale", 4.0)
            seed = job_input.get("seed")
            
            # Generate image using Qwen-Image
            start_time = time.time()
            result_image = asyncio.run(model_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            ))
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": f"Generated image for: {prompt}",
                "image": result_image,  # Base64 encoded image
                "metadata": {
                    "processing_time": processing_time,
                    "model": "Qwen/Qwen-Image",
                    "prompt": prompt,
                    "dimensions": f"{width}x{height}",
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                }
            }
        elif task_type == "edit":
            image_base64 = job_input.get("image")
            prompt = job_input.get("prompt", "edit the image")
            negative_prompt = job_input.get("negative_prompt", "")
            num_inference_steps = job_input.get("num_inference_steps", 30)
            guidance_scale = job_input.get("guidance_scale", 4.0)
            strength = job_input.get("strength", 0.8)
            seed = job_input.get("seed")
            
            if not image_base64:
                return {
                    "success": False,
                    "error": "No input image provided for editing"
                }
            
            # Edit image using Qwen-Image-Edit
            start_time = time.time()
            result_image = asyncio.run(model_manager.edit_image(
                image_base64=image_base64,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed
            ))
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "Image editing completed successfully",
                "image": result_image,  # Base64 encoded image
                "metadata": {
                    "processing_time": processing_time,
                    "model": "Qwen/Qwen-Image-Edit",
                    "prompt": prompt,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed
                }
            }
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}",
                "supported_tasks": ["health", "generate", "edit"]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}"
        }

def main():
    """Main entry point for RunPod serverless worker."""
    logger.info("🚀 Starting AI Image Editing Server")
    logger.info(f"Python version: {sys.version}")
    
    try:
        # Import and start RunPod worker
        import runpod
        logger.info("✅ RunPod imported successfully")
        
        # Start the serverless worker
        logger.info("🎯 Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()