# Qwen-Image AI Editing Server

A high-performance FastAPI server for AI image generation and editing using the Qwen-Image 20B model, optimized for RunPod serverless deployment.

## Features

- **Text-to-Image Generation**: Create high-quality images from text prompts
- **Image Editing**: Modify existing images using natural language instructions
- **High Resolution**: Support for images up to 2048x2048 pixels
- **Multi-language Support**: Generate images with text in multiple languages
- **GPU Acceleration**: Optimized for CUDA-enabled environments
- **FastAPI Integration**: RESTful API with automatic documentation
- **RunPod Ready**: Serverless deployment configuration included

## Architecture

```
artyx-image-editing-server/
├── main.py                 # FastAPI application entry point
├── api/                    # API endpoints and schemas
│   ├── routes.py          # REST API endpoints
│   └── schemas.py         # Pydantic request/response models
├── models/                 # AI model management
│   ├── qwen_image.py      # Qwen-Image pipeline wrapper
│   └── image_processor.py # Image processing utilities
├── runpod/                # RunPod serverless integration
│   ├── handler.py         # Serverless handler
│   └── template.json      # Deployment template
├── utils/                 # Utility functions
│   ├── logging.py         # Logging configuration
│   └── validators.py      # Input validation
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── deploy.py             # Deployment automation
```

## Quick Start

### Local Development

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Server**
```bash
python main.py
```

3. **Access API Documentation**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build Container**
```bash
docker build -t qwen-image-server .
```

2. **Run Container**
```bash
docker run -p 8000:8000 --gpus all qwen-image-server
```

### RunPod Serverless Deployment

1. **Set API Key**
```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

2. **Deploy to RunPod**
```bash
python deploy.py
```

3. **Test Deployment**
```bash
python deploy.py --test-only <endpoint-id>
```

## API Usage

### Text-to-Image Generation

```python
import requests

response = requests.post("http://localhost:8000/api/v1/generate", json={
    "prompt": "A beautiful mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
})

result = response.json()
if result["success"]:
    # result["image"] contains base64 encoded image
    print(f"Generated in {result['processing_time']:.2f}s")
```

### Image Editing

```python
import base64
import requests
from PIL import Image

# Load and encode image
with open("input.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/api/v1/edit", json={
    "image": image_data,
    "prompt": "Change the sky to a starry night",
    "strength": 0.8,
    "num_inference_steps": 50,
    "guidance_scale": 4.0
})

result = response.json()
if result["success"]:
    # Save edited image
    edited_data = base64.b64decode(result["image"])
    with open("output.jpg", "wb") as f:
        f.write(edited_data)
```

### Health Check

```python
import requests

response = requests.get("http://localhost:8000/api/v1/health")
health = response.json()

print(f"Status: {health['status']}")
print(f"Model loaded: {health['model_loaded']}")
print(f"GPU available: {health['gpu_available']}")
```

## RunPod Serverless Usage

### Text-to-Image Generation

```python
import runpod

runpod.api_key = "your-runpod-api-key"

result = runpod.run_sync("your-endpoint-id", {
    "task": "generate",
    "prompt": "A futuristic city skyline",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
})

if result["success"]:
    image_base64 = result["image"]
```

### Image Editing

```python
result = runpod.run_sync("your-endpoint-id", {
    "task": "edit",
    "image": "base64-encoded-image",
    "prompt": "Add rainbow colors to the sky",
    "strength": 0.7
})
```

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `TORCH_HOME`: PyTorch cache directory
- `HF_HOME`: HuggingFace cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory

### Model Configuration

The server uses the following Qwen-Image models:
- **Text-to-Image**: `Qwen/Qwen-Image`
- **Image Editing**: `Qwen/Qwen-Image-Edit`

Models are automatically downloaded on first use.

## Performance Optimization

### GPU Requirements

- **Minimum**: 16GB VRAM for 1024x1024 images
- **Recommended**: 24GB+ VRAM for 2048x2048 images
- **CUDA**: Version 12.1 or later

### Memory Optimization

The server includes several optimizations:
- Attention slicing for reduced VRAM usage
- XFormers memory efficient attention
- Model caching with lazy loading
- Automatic GPU memory cleanup

### Cold Start Times

- **RunPod Serverless**: ~2-5 seconds with FlashBoot
- **Local Docker**: ~10-30 seconds depending on hardware
- **Model Preloading**: Optional pre-download for faster starts

## Testing

### Run Test Suite

```bash
# Test local server
python test_client.py --url http://localhost:8000

# Test RunPod endpoint
python test_client.py --runpod-endpoint <endpoint-id> --runpod-api-key <api-key>
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Model info
curl http://localhost:8000/api/v1/models
```

## Monitoring and Logging

### Log Files

Logs are written to console and optionally to files:
- Application logs include request/response details
- Error logs capture failures with stack traces
- Performance logs track processing times

### Health Monitoring

The `/health` endpoint provides:
- Service status
- Model loading status
- GPU availability
- Memory usage statistics

### Metrics

Key performance metrics:
- Processing time per request
- GPU memory utilization
- Queue depth (for serverless)
- Error rates

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce image resolution
   - Enable attention slicing
   - Use smaller batch sizes

2. **Model Loading Failures**
   - Check internet connection
   - Verify HuggingFace cache permissions
   - Clear model cache if corrupted

3. **Slow Performance**
   - Ensure GPU acceleration is enabled
   - Check CUDA version compatibility
   - Monitor system resource usage

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## Development

### Project Structure

- `main.py`: FastAPI application setup
- `api/`: REST API implementation
- `models/`: AI model integration
- `runpod/`: Serverless deployment
- `utils/`: Helper utilities

### Adding New Features

1. Add API endpoints in `api/routes.py`
2. Define schemas in `api/schemas.py`
3. Implement model logic in `models/`
4. Add tests in `test_client.py`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs
3. Open an issue on GitHub
4. Contact support with deployment details

## Changelog

### v1.0.0
- Initial release
- Qwen-Image integration
- FastAPI server implementation
- RunPod serverless support
- Comprehensive testing suite