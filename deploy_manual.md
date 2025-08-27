# Manual Deployment Guide for RunPod

Since automatic deployment requires Docker setup, here's how to deploy the Qwen-Image AI editing server to RunPod manually:

## Step 1: Prepare Files

All necessary files have been created in the `artyx-image-editing-server/` directory:

- ✅ **runpod_worker.py** - Main RunPod serverless handler
- ✅ **Dockerfile** - Container configuration
- ✅ **requirements.txt** - Python dependencies
- ✅ **models/** - Qwen-Image integration
- ✅ **api/** - FastAPI endpoints
- ✅ **utils/** - Logging and validation

## Step 2: RunPod Web Console Deployment

### Option A: Upload Files to RunPod

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless

2. **Create New Endpoint**:
   - Click "New Endpoint"
   - Choose "Custom" template

3. **Container Configuration**:
   ```
   Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
   Container Disk: 50GB
   Ports: 8000/http (optional, for FastAPI mode)
   ```

4. **Environment Variables**:
   ```
   RUNPOD_MODE=true
   LOG_LEVEL=INFO
   TORCH_HOME=/tmp/.torch
   HF_HOME=/tmp/.huggingface
   TRANSFORMERS_CACHE=/tmp/.transformers
   ```

5. **Upload Code**:
   - Use RunPod's file upload feature
   - Upload all files from `artyx-image-editing-server/`
   - Set entry point: `python runpod_worker.py`

### Option B: GitHub Integration

1. **Create GitHub Repository** (optional):
   - Upload all files to a GitHub repo
   - Make it public or provide access token

2. **Use GitHub in RunPod**:
   - In template creation, use GitHub URL
   - Set Dockerfile path if needed

## Step 3: Test Deployment

### Using your RunPod API key:

Once deployed, test the endpoint:

```python
import requests

# Replace with your endpoint ID
ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-runpod-api-key"

# Test health check
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "task": "health"
        }
    }
)

print(response.json())
```

### Text-to-Image Generation Test:

```python
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "task": "generate",
            "prompt": "A beautiful mountain landscape at sunset",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50
        }
    },
    timeout=300  # 5 minutes
)

result = response.json()
if result.get("status") == "COMPLETED":
    image_base64 = result["output"]["image"]
    # Save or process the image
```

## Step 4: Monitoring

1. **Check Logs**: Monitor RunPod console for initialization and errors
2. **Performance**: Watch GPU memory usage and processing times
3. **Scaling**: Adjust min/max workers based on load

## Expected Initialization Time

- **Cold Start**: 30-60 seconds (model download + loading)
- **Warm Start**: 2-5 seconds
- **Processing**: 10-30 seconds per image (depending on settings)

## Troubleshooting

### Common Issues:

1. **Model Download Timeout**:
   - Increase container timeout
   - Use larger disk space

2. **GPU Memory Errors**:
   - Reduce image resolution
   - Lower inference steps
   - Use GPU with more VRAM

3. **Dependencies Missing**:
   - Check requirements.txt installation
   - Verify PyTorch CUDA version

### Debug Commands:

Add to your test input for debugging:
```json
{
  "task": "health",
  "debug": true
}
```

## Cost Estimation

- **GPU Type**: RTX 4090 (24GB) recommended
- **Processing**: ~$0.50-2.00 per image (depending on resolution/steps)
- **Idle**: $0 (serverless scaling)

## Production Configuration

For production use:
- Enable FlashBoot for faster cold starts
- Set appropriate min/max workers
- Configure proper error handling
- Add request queuing for high load

The server is now ready for deployment to RunPod serverless infrastructure!