"""
Qwen-Image AI Editing Server - FastAPI Application
Main entry point for the AI image editing server using Qwen-Image model.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from models.qwen_image import QwenImageManager
from utils.logging import setup_logging


# Global model manager instance
model_manager: Optional[QwenImageManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global model_manager
    
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing Qwen-Image model manager...")
        model_manager = QwenImageManager()
        await model_manager.initialize()
        logger.info("Model manager initialized successfully")
        
        # Store in app state for access in routes
        app.state.model_manager = model_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        raise
    
    yield
    
    # Shutdown
    if model_manager:
        logger.info("Cleaning up model manager...")
        await model_manager.cleanup()


# Create FastAPI application
app = FastAPI(
    title="Qwen-Image AI Editing Server",
    description="Advanced AI image generation and editing using Qwen-Image model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "Qwen-Image AI Editing Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )