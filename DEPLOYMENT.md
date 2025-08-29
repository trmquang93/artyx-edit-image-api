# RunPod Deployment Instructions

This repository contains a production-ready Qwen-Image AI editing server optimized for RunPod serverless deployment.

## Quick Start

### 1. RunPod Console Deployment (Recommended)

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless

2. **Create New Template**:
   - Click "New Template"
   - Choose "Source Code" deployment
   - **GitHub URL**: `https://github.com/trmquang93/artyx-edit-image-api`
   - **Container Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
   - **Container Disk**: 50GB
   - **Start Command**: `python runpod_worker.py`

3. **Environment Variables** (CRITICAL FOR DISK SPACE):
   ```
   RUNPOD_MODE=true
   LOG_LEVEL=INFO
   HF_HOME=/runpod-volume/.huggingface
   TRANSFORMERS_CACHE=/runpod-volume/.transformers
   TORCH_HOME=/runpod-volume/.torch
   PORT=80
   PORT_HEALTH=80
   ```
   
   **⚠️ IMPORTANT**: Use `/runpod-volume/` paths to store models on network volume instead of container disk. This prevents "No space left on device" errors.

4. **Network Volume Setup** (Required):
   - Create a Network Volume with 200GB capacity
   - Attach the volume to your endpoint
   - The volume will mount at `/runpod-volume/` automatically

5. **Create Serverless Endpoint**:
   - Use the template you just created
   - **Attach network volume** for model persistence
   - Configure scaling (0-3 workers recommended)
   - Enable FlashBoot for faster cold starts

### 2. Local Testing

```bash
# Clone repository
git clone https://github.com/trmquang93/artyx-edit-image-api.git
cd artyx-edit-image-api

# Set up environment
export RUNPOD_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Test locally
python runpod_worker.py  # RunPod mode
# OR
python main.py          # FastAPI server mode
```

### 3. API Usage

Once deployed, test your endpoint:

```python
import requests

# Replace with your actual endpoint ID and API key
ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-runpod-api-key"

# Text-to-image generation
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "task": "generate",
            "prompt": "A beautiful mountain landscape at sunset, masterpiece",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50
        }
    },
    timeout=300
)

result = response.json()
if result.get("status") == "COMPLETED":
    image_base64 = result["output"]["image"]
    print("Image generated successfully!")
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required
RUNPOD_API_KEY=your-runpod-api-key

# Optional
LOG_LEVEL=INFO
RUNPOD_MODE=true
TORCH_HOME=/tmp/.torch
HF_HOME=/tmp/.huggingface
```

## API Endpoints

The server supports three main tasks:

1. **Health Check**: `{"task": "health"}`
2. **Generate**: `{"task": "generate", "prompt": "...", "width": 1024, "height": 1024}`
3. **Edit**: `{"task": "edit", "image": "<base64>", "prompt": "change the sky"}`

## Performance

- **Cold Start**: 30-60 seconds (first request)
- **Warm Start**: 2-5 seconds (subsequent requests)
- **Processing**: 10-30 seconds per image
- **GPU Requirements**: 16GB+ VRAM recommended

## Troubleshooting

### Common Issues

1. **Model Loading Timeout**: Increase container timeout in RunPod settings
2. **GPU Memory Error**: Use smaller image sizes or reduce inference steps
3. **Dependencies Error**: Ensure all packages in requirements.txt are installed

### Debug Mode

Enable debug logging:
```json
{
  "input": {
    "task": "health",
    "debug": true
  }
}
```

## Cost Estimation

- **GPU**: ~$0.50-2.00 per image (depending on size/steps)
- **Storage**: Included in serverless pricing
- **Idle**: $0 (auto-scaling to zero)

## Support

For issues:
1. Check RunPod console logs
2. Verify environment variables
3. Test with health check endpoint first
4. Review the troubleshooting section above

## Security

- Never commit API keys to version control
- Use environment variables for all sensitive configuration
- The `.env.example` file shows the required variables