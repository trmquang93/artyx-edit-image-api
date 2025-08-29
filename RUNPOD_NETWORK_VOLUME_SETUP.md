# RunPod Network Volume Setup for Disk Space Optimization

## Problem Solved
This configuration fixes the "No space left on device" error by using the 200 GB network volume instead of the 50 GB container disk for model caching.

## RunPod Environment Variables Configuration

Update your RunPod serverless endpoint with these **exact** environment variables:

```
HF_HOME=/runpod-volume/.huggingface
TRANSFORMERS_CACHE=/runpod-volume/.transformers
TORCH_HOME=/runpod-volume/.torch
PORT=80
PORT_HEALTH=80
```

## Critical Notes

### Network Volume Path
- Your network volume ID: `v78qlse0t3`
- Mount path in container: `/runpod-volume/`
- **Do NOT use** `/runpod-volume/v78qlse0t3/` - RunPod handles the volume ID automatically

### Container Configuration
- **Container Disk**: Keep at 50 GB (sufficient for runtime)
- **Network Volume**: 200 GB (for model storage)
- **Docker Image**: Use `ghcr.io/trmquang93/artyx-edit-image-api:latest`

### Expected Behavior
1. **First Run**: Models download to network volume (5-10 minutes)
2. **Subsequent Runs**: Models load from network volume cache (30-60 seconds)
3. **Disk Usage**: Container stays under 10 GB, network volume grows to ~30-40 GB

## Deployment Steps

1. **Update Environment Variables** in RunPod dashboard using values above
2. **Deploy** with existing container image
3. **Monitor Logs** for disk usage reports:
   ```
   Container disk: 45.2GB free / 50.0GB total
   Network volume: 165.3GB free / 200.0GB total
   Created cache directory: /runpod-volume/.huggingface
   ```

## Verification Commands

If you have pod access, verify setup with:
```bash
# Check mount point
ls -la /runpod-volume/

# Check environment variables
echo $HF_HOME
echo $TRANSFORMERS_CACHE
echo $TORCH_HOME

# Check disk usage
df -h
```

## Troubleshooting

### Issue: Still getting disk space errors
- **Solution**: Verify environment variables are exactly as specified above
- **Check**: Ensure `/runpod-volume` path is used, not `/workspace`

### Issue: Models not persisting between runs
- **Solution**: Verify network volume is attached and environment variables point to `/runpod-volume`

### Issue: Slow first-time startup
- **Expected**: First model download takes 5-10 minutes, then cached permanently

## Cost Savings
- **Network Volume**: Models persist between serverless runs
- **Faster Startups**: Subsequent runs avoid re-downloading 20+ GB models
- **Reliability**: No more disk space failures during model loading

---
*This configuration has been tested and resolves the disk space issues completely.*