"""
FastAPI routes for AI image editing endpoints.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from api.schemas import (
    GenerateImageRequest,
    EditImageRequest,
    ImageResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from models.qwen_image import QwenImageManager


router = APIRouter()
logger = logging.getLogger(__name__)


def get_model_manager(request: Request) -> QwenImageManager:
    """Dependency to get model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized"
        )
    return model_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Health check endpoint."""
    try:
        health_info = await model_manager.get_health_info()
        
        return HealthResponse(
            status="healthy",
            message="Service is running normally",
            model_loaded=health_info["model_loaded"],
            gpu_available=health_info["gpu_available"],
            memory_usage=health_info.get("memory_usage")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            model_loaded=False,
            gpu_available=False
        )


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Get information about loaded models."""
    try:
        info = await model_manager.get_model_info()
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info["model_type"],
            capabilities=info["capabilities"],
            supported_formats=info["supported_formats"],
            max_resolution=info["max_resolution"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.post("/generate", response_model=ImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Generate image from text prompt using Qwen-Image."""
    start_time = time.time()
    
    try:
        logger.info(f"Generating image with prompt: {request.prompt[:100]}...")
        
        # Generate image
        result_image = await model_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Image generated successfully in {processing_time:.2f}s")
        
        return ImageResponse(
            success=True,
            image=result_image,
            message="Image generated successfully",
            metadata={
                "prompt": request.prompt,
                "dimensions": f"{request.width}x{request.height}",
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Image generation failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        return ImageResponse(
            success=False,
            message=error_msg,
            processing_time=processing_time
        )


@router.post("/edit", response_model=ImageResponse)
async def edit_image(
    request: EditImageRequest,
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Edit image using text prompt with Qwen-Image editing model."""
    start_time = time.time()
    
    try:
        logger.info(f"Editing image with prompt: {request.prompt[:100]}...")
        
        # Edit image
        result_image = await model_manager.edit_image(
            image_base64=request.image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Image edited successfully in {processing_time:.2f}s")
        
        return ImageResponse(
            success=True,
            image=result_image,
            message="Image edited successfully",
            metadata={
                "prompt": request.prompt,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "strength": request.strength,
                "seed": request.seed
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Image editing failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        return ImageResponse(
            success=False,
            message=error_msg,
            processing_time=processing_time
        )


@router.get("/status")
async def get_status(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Get detailed server status information."""
    try:
        status = await model_manager.get_detailed_status()
        return {
            "server_status": "running",
            "timestamp": time.time(),
            **status
        }
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get server status: {str(e)}"
        )