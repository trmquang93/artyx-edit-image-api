"""
RunPod serverless handler for Qwen-Image AI editing server.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runpod
from models.qwen_image import QwenImageManager
from utils.logging import setup_logging


# Global model manager
model_manager: Optional[QwenImageManager] = None


async def initialize_model():
    """Initialize the model manager."""
    global model_manager
    
    if model_manager is None:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
        logger = logging.getLogger(__name__)
        
        logger.info("Initializing Qwen-Image model manager...")
        model_manager = QwenImageManager()
        await model_manager.initialize()
        logger.info("Model manager initialized successfully")
    
    return model_manager


async def generate_image_handler(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle text-to-image generation requests."""
    try:
        # Initialize model if needed
        manager = await initialize_model()
        
        # Extract parameters
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        num_inference_steps = job_input.get("num_inference_steps", 50)
        guidance_scale = job_input.get("guidance_scale", 4.0)
        seed = job_input.get("seed")
        
        # Validate inputs
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Generate image
        start_time = time.time()
        result_image = await manager.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "image": result_image,
            "metadata": {
                "prompt": prompt,
                "dimensions": f"{width}x{height}",
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "processing_time": processing_time
            }
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def edit_image_handler(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle image editing requests."""
    try:
        # Initialize model if needed
        manager = await initialize_model()
        
        # Extract parameters
        image_base64 = job_input.get("image", "")
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt")
        num_inference_steps = job_input.get("num_inference_steps", 50)
        guidance_scale = job_input.get("guidance_scale", 4.0)
        strength = job_input.get("strength", 0.8)
        seed = job_input.get("seed")
        
        # Validate inputs
        if not image_base64:
            return {"error": "Input image is required"}
        if not prompt:
            return {"error": "Prompt is required"}
        
        # Edit image
        start_time = time.time()
        result_image = await manager.edit_image(
            image_base64=image_base64,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "image": result_image,
            "metadata": {
                "prompt": prompt,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "seed": seed,
                "processing_time": processing_time
            }
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Editing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def health_check_handler(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle health check requests."""
    try:
        manager = await initialize_model()
        health = await manager.get_health_info()
        
        return {
            "status": "healthy",
            "model_loaded": health["model_loaded"],
            "gpu_available": health["gpu_available"],
            "memory_usage": health.get("memory_usage", {})
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def sync_wrapper(async_handler):
    """Wrapper to run async handlers in sync context."""
    def wrapper(job_input):
        try:
            # Create new event loop for each job
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(async_handler(job_input))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Handler wrapper error: {e}")
            return {
                "success": False,
                "error": f"Handler error: {str(e)}"
            }
    
    return wrapper


def main_handler(job):
    """Main RunPod handler that routes requests to appropriate handlers."""
    job_input = job.get("input", {})
    task_type = job_input.get("task", "generate")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing task: {task_type}")
    
    # Route to appropriate handler
    if task_type == "generate":
        return sync_wrapper(generate_image_handler)(job_input)
    elif task_type == "edit":
        return sync_wrapper(edit_image_handler)(job_input)
    elif task_type == "health":
        return sync_wrapper(health_check_handler)(job_input)
    else:
        return {
            "success": False,
            "error": f"Unknown task type: {task_type}",
            "supported_tasks": ["generate", "edit", "health"]
        }


if __name__ == "__main__":
    # Start RunPod serverless worker
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RunPod serverless worker...")
    
    runpod.serverless.start({
        "handler": main_handler
    })