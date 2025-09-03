#!/usr/bin/env python3
"""
Enhanced RunPod serverless handler for AI image editing using Qwen-Image models.
Based on proven production patterns for reliability and performance.
All dependencies embedded to avoid import issues.
"""

import runpod
import sys
import os
import time
import logging
import traceback
import base64
import io
import uuid
import json
import binascii
import concurrent.futures
from typing import Optional

# Set up logging with comprehensive format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정 (Enhanced pattern)
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)

def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    Enhanced pattern: 입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다.
    만약 일반 경로 문자열이라면 그대로 반환합니다.
    """
    # 입력값이 문자열이 아니면 그대로 반환
    if not isinstance(data_input, str):
        return data_input

    try:
        # Remove data URL prefix if present
        if data_input.startswith('data:image/'):
            data_input = data_input.split(',', 1)[1]
        
        # Base64 문자열은 디코딩을 시도하면 성공합니다.
        decoded_data = base64.b64decode(data_input)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 디코딩에 성공하면, 임시 파일로 저장합니다.
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:  # 바이너리 쓰기 모드('wb')로 저장
            f.write(decoded_data)
        
        # 저장된 파일의 경로를 반환합니다.
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path

    except (binascii.Error, ValueError):
        # 디코딩에 실패하면, 일반 경로로 간주하고 원래 값을 그대로 반환합니다.
        logger.info(f"➡️ '{data_input[:50]}...'은(는) 파일 경로 또는 URL로 처리합니다.")
        return data_input


class ImageProcessor:
    """Simple image processing utilities - embedded in handler for reliability."""
    
    def __init__(self):
        self.max_size = 2048
    
    def base64_to_pil(self, base64_string: str):
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
    
    def pil_to_base64(self, image, format: str = 'PNG') -> str:
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
    """Manages Qwen-Image models with enhanced production patterns."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.device = None
        self.torch_dtype = None
        self._initialized = False
        
        # Model configurations - Pure Qwen models only
        self.text_to_image_model = "Qwen/Qwen-Image"
        self.image_edit_model = "Qwen/Qwen-Image-Edit"
    
    def initialize(self):
        """Initialize the model pipelines - synchronous for reliability."""
        if self._initialized:
            return
            
        logger.info("🚀 Initializing Qwen-Image models...")
        
        try:
            # Check if torch is available
            try:
                import torch
                logger.info("✅ PyTorch found successfully")
            except ImportError as e:
                logger.error("❌ PyTorch not found. Please ensure torch and related ML dependencies are installed.")
                logger.error("Required packages: torch, torchvision, diffusers, transformers, accelerate")
                self._initialized = False
                return
            
            try:
                from diffusers import DiffusionPipeline
                from transformers import set_seed
                logger.info("✅ Diffusers and transformers imported successfully")
            except ImportError as e:
                logger.error(f"❌ Failed to import diffusers or transformers: {e}")
                self._initialized = False
                return
            
            # Set device and dtype
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
            
            # Initialize models for real AI processing
            logger.info("🤖 Loading AI models for image processing...")
            
            self._initialized = True
            logger.info("✅ Model manager initialized (models will load on demand)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialized = False
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None
    ) -> str:
        """Generate image from text prompt using Qwen-Image model only."""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.info(f"🎨 Generating {width}x{height} image with Qwen-Image: '{prompt}'")
            
            from diffusers import DiffusionPipeline
            import torch
            
            # Load Qwen-Image model if not already loaded
            if not hasattr(self, 'qwen_text_pipeline'):
                logger.info("🔄 Loading Qwen-Image text-to-image model...")
                try:
                    self.qwen_text_pipeline = DiffusionPipeline.from_pretrained(
                        self.text_to_image_model,  # "Qwen/Qwen-Image"
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True
                    )
                    
                    if torch.cuda.is_available():
                        self.qwen_text_pipeline = self.qwen_text_pipeline.to("cuda")
                        logger.info("✅ Qwen-Image model loaded on GPU")
                    else:
                        logger.info("⚠️ Qwen-Image model loaded on CPU (slower)")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load Qwen-Image model: {e}")
                    raise RuntimeError(f"Qwen-Image model '{self.text_to_image_model}' could not be loaded: {str(e)}")
            
            # Set seed for reproducibility
            generator = torch.manual_seed(seed) if seed is not None else None
            
            # Generate with Qwen model
            logger.info(f"🎯 Processing with Qwen-Image: {num_inference_steps} steps")
            logger.info(f"📝 Prompt: '{prompt}' ({len(prompt)} chars)")
            
            with torch.inference_mode():
                result = self.qwen_text_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
            
            logger.info("✅ Qwen-Image generation completed")
            
            # Convert to base64
            return self.image_processor.pil_to_base64(result)
            
        except Exception as e:
            logger.error(f"Qwen-Image generation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Qwen-Image generation failed: {str(e)}")
    
    def edit_image(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> str:
        """Edit image using Qwen-Image-Edit model only."""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.info(f"🎨 Editing image with Qwen-Image-Edit: '{prompt}'")
            
            # Decode input image to verify it's valid
            input_image = self.image_processor.base64_to_pil(image_base64)
            logger.info(f"✅ Input image decoded successfully: {input_image.size}")
            
            from diffusers import DiffusionPipeline
            import torch
            
            # Load Qwen-Image-Edit model if not already loaded
            if not hasattr(self, 'edit_pipeline'):
                logger.info("🔄 Loading Qwen-Image-Edit model...")
                try:
                    self.edit_pipeline = DiffusionPipeline.from_pretrained(
                        self.image_edit_model,  # "Qwen/Qwen-Image-Edit"
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True
                    )
                    
                    if torch.cuda.is_available():
                        self.edit_pipeline = self.edit_pipeline.to("cuda")
                        logger.info("✅ Qwen-Image-Edit model loaded on GPU")
                    else:
                        logger.info("⚠️ Qwen-Image-Edit model loaded on CPU (slower)")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load Qwen-Image-Edit model: {e}")
                    raise RuntimeError(f"Qwen-Image-Edit model '{self.image_edit_model}' could not be loaded: {str(e)}")
            
            # Set seed for reproducibility
            generator = torch.manual_seed(seed) if seed is not None else None
            
            logger.info(f"🎯 Processing with Qwen-Image-Edit: {num_inference_steps} steps")
            logger.info(f"📝 Edit instruction: '{prompt}' ({len(prompt)} chars)")
            
            with torch.inference_mode():
                result = self.edit_pipeline(
                    prompt=prompt,
                    image=input_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator
                ).images[0]
            
            logger.info("✅ Qwen-Image-Edit processing completed")
            
            # Convert back to base64
            return self.image_processor.pil_to_base64(result)
            
        except Exception as e:
            logger.error(f"Qwen-Image-Edit processing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Qwen-Image-Edit processing failed: {str(e)}")
    
    def get_health_info(self):
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


# Global model manager - will be initialized once (Enhanced pattern)
model_manager = None


def handler(job):
    """RunPod serverless handler function - Enhanced production structure."""
    logger.info("=== JOB RECEIVED ===")
    logger.info(f"Job input: {job}")
    
    try:
        global model_manager
        
        # Initialize model manager if not already done
        if model_manager is None:
            logger.info("🚀 Initializing Qwen-Image model manager...")
            model_manager = QwenImageManager()
            # Synchronous initialization for reliability
            model_manager.initialize()
            logger.info("✅ Model manager initialized successfully")
        
        job_input = job.get("input", {})
        task_type = job_input.get("task", "health")
        
        logger.info(f"Processing task: {task_type}")
        
        if task_type == "health":
            # Get health info from model manager
            health_info = model_manager.get_health_info()
            return {
                "success": True,
                "status": "healthy",
                "message": "AI Image Editing Server is running successfully!",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "timestamp": time.time(),
                    "server_type": "ai_image_editing_enhanced",
                    "model_loaded": health_info.get("model_loaded", False),
                    "gpu_available": health_info.get("gpu_available", False)
                }
            }
        elif task_type == "generate":
            prompt = job_input.get("prompt", "a beautiful landscape")
            negative_prompt = job_input.get("negative_prompt", "")
            width = job_input.get("width", 1024)
            height = job_input.get("height", 1024)
            num_inference_steps = job_input.get("num_inference_steps", 50)
            guidance_scale = job_input.get("guidance_scale", 4.0)
            seed = job_input.get("seed")
            
            # Generate image using Qwen-Image
            start_time = time.time()
            result_image = model_manager.generate_image(
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
        elif task_type == "edit" or task_type == "background_replacement":
            # Handle image input with Flux's flexible pattern
            image_base64 = job_input.get("image")
            image_url = job_input.get("image_url")
            image_path = job_input.get("image_path")
            prompt = job_input.get("prompt", "edit the image")
            negative_prompt = job_input.get("negative_prompt", "")
            num_inference_steps = job_input.get("num_inference_steps", 50)
            guidance_scale = job_input.get("guidance_scale", 4.0)
            strength = job_input.get("strength", 0.8)
            seed = job_input.get("seed")
            
            # Generate unique task ID for tracking
            task_id = f"task_{uuid.uuid4()}"
            
            # Handle image input - support multiple formats for flexibility
            image_input = image_base64 or image_url or image_path
            if not image_input:
                return {
                    "success": False,
                    "error": "No input image provided. Please provide 'image' (base64), 'image_url', or 'image_path'."
                }
            
            # Use enhanced save_data_if_base64 pattern
            if image_url and not image_base64:
                try:
                    import requests
                    from PIL import Image
                    
                    logger.info(f"📥 Downloading image from URL: {image_url}")
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    
                    # Convert to PIL and then to base64
                    image = Image.open(io.BytesIO(response.content))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Convert to base64
                    image_base64 = model_manager.image_processor.pil_to_base64(image)
                    logger.info("✅ Successfully converted URL image to base64")
                    
                except Exception as e:
                    logger.error(f"Failed to download/convert image from URL: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to process image URL: {str(e)}"
                    }
            elif image_path and not image_base64:
                # Use enhanced helper function for base64 detection
                processed_path = save_data_if_base64(image_path, task_id, "input_image.jpg")
                
                if processed_path != image_path:
                    # It was base64, now we have a file path
                    logger.info(f"✅ Base64 data saved to: {processed_path}")
                    # Read the file back as base64
                    with open(processed_path, 'rb') as f:
                        image_base64 = base64.b64encode(f.read()).decode('utf-8')
                else:
                    # It's a file path, read it
                    try:
                        from PIL import Image
                        image = Image.open(processed_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image_base64 = model_manager.image_processor.pil_to_base64(image)
                        logger.info(f"✅ Successfully loaded image from path: {processed_path}")
                    except Exception as e:
                        logger.error(f"Failed to load image from path: {e}")
                        return {
                            "success": False,
                            "error": f"Failed to load image from path: {str(e)}"
                        }
            
            if not image_base64:
                return {
                    "success": False,
                    "error": "Failed to process input image data."
                }
            
            # Edit image using Qwen-Image-Edit
            start_time = time.time()
            result_image = model_manager.edit_image(
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
                "message": "Image editing completed successfully",
                "image": result_image,  # Base64 encoded image (Enhanced format)
                "metadata": {
                    "processing_time": processing_time,
                    "model": "Qwen/Qwen-Image-Edit",
                    "prompt": prompt,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed,
                    "task_id": task_id
                }
            }
        elif task_type == "debug":
            # Debug endpoint like current implementation but with enhanced logging
            import subprocess
            
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                      capture_output=True, text=True, timeout=30)
                pip_list = result.stdout
            except:
                pip_list = "Could not get pip list"
            
            return {
                "success": True,
                "handler_version": "enhanced_v1.0",
                "python_version": sys.version,
                "python_path": sys.executable,
                "installed_packages": pip_list[:2000],  # Truncate to avoid huge response
                "environment_vars": {
                    "PATH": os.environ.get("PATH", "")[:500],
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "FORCE_CUDA": os.environ.get("FORCE_CUDA", ""),
                }
            }
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}",
                "supported_tasks": ["health", "generate", "edit", "background_replacement", "debug"]
            }
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}",
            "handler_version": "enhanced_v1.0"
        }


def main():
    """Main entry point for RunPod serverless worker - Enhanced pattern."""
    logger.info("🚀 Starting AI Image Editing Server (Enhanced)")
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