#!/usr/bin/env python3
"""
Standalone RunPod serverless handler for AI image editing using Qwen-Image models.
All dependencies embedded to avoid import issues.
"""

import sys
import os
import time
import logging
import traceback
import asyncio
import base64
import io
import concurrent.futures
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_async_in_sync(coro):
    """Run async function in sync context, handling existing event loops."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run in thread pool to avoid nested event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


class ImageProcessor:
    """Simple image processing utilities."""
    
    def __init__(self):
        self.max_size = 2048
    
    async def base64_to_pil(self, base64_string: str):
        """Convert base64 string to PIL Image."""
        try:
            from PIL import Image
            
            # Remove data URL prefix if present
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Open as PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert base64 to PIL image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    async def pil_to_base64(self, image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format=format, quality=95, optimize=True)
            buffer.seek(0)
            
            # Encode to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert PIL image to base64: {e}")
            raise ValueError(f"Image conversion failed: {str(e)}")


class QwenImageManager:
    """Manages Qwen-Image models for text-to-image and image editing."""
    
    def __init__(self):
        self.text_to_image_pipeline = None
        self.image_edit_pipeline = None
        self.image_processor = ImageProcessor()
        self.device = None
        self.torch_dtype = None
        self._initialized = False
        
        # Model configurations
        self.text_to_image_model = "Qwen/Qwen-Image"
        self.image_edit_model = "Qwen/Qwen-Image-Edit"
    
    async def initialize(self):
        """Initialize the model pipelines."""
        if self._initialized:
            return
            
        logger.info("Initializing Qwen-Image models...")
        
        try:
            # Check if torch is available, install if needed
            try:
                import torch
            except ImportError:
                logger.info("PyTorch not found, attempting runtime installation...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "diffusers", "transformers", "accelerate", "Pillow"])
                import torch
            
            from diffusers import DiffusionPipeline
            from transformers import set_seed
            
            # Set device and dtype
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
            
            # Initialize text-to-image pipeline
            logger.info(f"Loading text-to-image model: {self.text_to_image_model}")
            self.text_to_image_pipeline = DiffusionPipeline.from_pretrained(
                self.text_to_image_model,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            )
            self.text_to_image_pipeline = self.text_to_image_pipeline.to(self.device)
            
            # Initialize image editing pipeline
            logger.info(f"Loading image editing model: {self.image_edit_model}")
            try:
                from diffusers import QwenImageEditPipeline
                self.image_edit_pipeline = QwenImageEditPipeline.from_pretrained(
                    self.image_edit_model,
                    torch_dtype=self.torch_dtype
                )
            except ImportError:
                logger.warning("QwenImageEditPipeline not available, using DiffusionPipeline")
                self.image_edit_pipeline = DiffusionPipeline.from_pretrained(
                    self.image_edit_model,
                    torch_dtype=self.torch_dtype
                )
            
            self.image_edit_pipeline = self.image_edit_pipeline.to(self.device)
            
            self._initialized = True
            logger.info("Qwen-Image models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None
    ) -> str:
        """Generate image from text prompt."""
        if not self._initialized:
            await self.initialize()
        
        if not self.text_to_image_pipeline:
            raise RuntimeError("Text-to-image pipeline not available")
        
        try:
            import torch
            from transformers import set_seed
            
            # Set seed for reproducibility
            if seed is not None:
                set_seed(seed)
            
            logger.info(f"Generating {width}x{height} image with {num_inference_steps} steps")
            
            # Generate image
            with torch.inference_mode():
                result = self.text_to_image_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=guidance_scale
                )
            
            # Convert to base64
            return await self.image_processor.pil_to_base64(result.images[0])
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def edit_image(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> str:
        """Edit image using text prompt."""
        if not self._initialized:
            await self.initialize()
        
        if not self.image_edit_pipeline:
            raise RuntimeError("Image editing pipeline not available")
        
        try:
            import torch
            from transformers import set_seed
            
            # Set seed for reproducibility
            if seed is not None:
                set_seed(seed)
            
            # Decode input image
            input_image = await self.image_processor.base64_to_pil(image_base64)
            
            logger.info(f"Editing image with {num_inference_steps} steps, strength {strength}")
            
            # Edit image
            with torch.inference_mode():
                result = self.image_edit_pipeline(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=guidance_scale,
                    strength=strength
                )
            
            # Convert to base64
            return await self.image_processor.pil_to_base64(result.images[0])
            
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
    
    async def get_health_info(self):
        """Get health information about the model manager."""
        try:
            import torch
        except ImportError:
            return {"model_loaded": False, "gpu_available": False}
        
        health = {
            "model_loaded": self._initialized,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                health["memory_usage"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "utilization": round(memory_allocated / memory_total * 100, 1)
                }
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return health


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
            model_manager = QwenImageManager()
            # Run async initialization in sync context with proper event loop handling
            run_async_in_sync(model_manager.initialize())
            logger.info("Model manager initialized successfully")
        
        job_input = job.get("input", {})
        task_type = job_input.get("task", "health")
        
        logger.info(f"Processing task: {task_type}")
        
        if task_type == "health":
            # Get health info from model manager
            health_info = run_async_in_sync(model_manager.get_health_info())
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
            result_image = run_async_in_sync(model_manager.generate_image(
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
        elif task_type == "debug":
            # Debug endpoint to check available packages
            import subprocess
            
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                      capture_output=True, text=True, timeout=30)
                pip_list = result.stdout
            except:
                pip_list = "Could not get pip list"
            
            return {
                "success": True,
                "python_version": sys.version,
                "python_path": sys.executable,
                "installed_packages": pip_list[:2000],  # Truncate to avoid huge response
                "environment_vars": {
                    "PATH": os.environ.get("PATH", "")[:500],
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "HOME": os.environ.get("HOME", ""),
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
            result_image = run_async_in_sync(model_manager.edit_image(
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