# 📁 Multipart File Upload Support

Your AI Image Editing Server now supports **multipart file uploads** in addition to base64 image processing! This makes it much easier to integrate with web applications, mobile apps, and command-line tools.

## 🚀 Quick Start

### RunPod Serverless (Default)
Your existing RunPod endpoint continues to work exactly as before:
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{"input":{"task":"background_replacement","prompt":"sunset","image_url":"https://example.com/image.jpg"}}'
```

### FastAPI Mode (New!)
Deploy as a standard web server with file upload support:
```bash
docker run -e SERVER_MODE=fastapi -p 8000:8000 your-image-name
```

## 📋 New Endpoints

### 1. **Image Editing with File Upload**
**`POST /api/v1/upload/edit`**

Upload an image file and edit it with AI:

```bash
curl -X POST "http://localhost:8000/api/v1/upload/edit" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Add beautiful flowers and make it vibrant" \
  -F "negative_prompt=dark, blurry" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=7.5" \
  -F "strength=0.8" \
  -F "seed=42"
```

**Parameters:**
- `file` (required): Image file (JPEG, PNG, etc.)
- `prompt` (required): Edit instruction
- `negative_prompt` (optional): What to avoid
- `num_inference_steps` (optional): Quality vs speed (10-100, default: 50)
- `guidance_scale` (optional): How strictly to follow prompt (1.0-20.0, default: 4.0)
- `strength` (optional): Edit intensity (0.1-1.0, default: 0.8)
- `seed` (optional): For reproducible results

### 2. **Background Replacement with File Upload**
**`POST /api/v1/upload/background-replacement`**

Upload an image and replace its background:

```bash
curl -X POST "http://localhost:8000/api/v1/upload/background-replacement" \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Beautiful sunset over ocean waves" \
  -F "negative_prompt=city, urban" \
  -F "num_inference_steps=25" \
  -F "guidance_scale=8.0" \
  -F "strength=0.9"
```

## 🔧 Technical Details

### File Validation
- **Supported formats**: JPEG, PNG, BMP, TIFF, WebP
- **Maximum file size**: 10MB
- **Auto-conversion**: Files are automatically converted to RGB format
- **Security**: Images are validated before processing

### Response Format
```json
{
  "success": true,
  "image": "base64_encoded_result_image",
  "message": "Image edited successfully",
  "metadata": {
    "prompt": "Add beautiful flowers",
    "steps": 30,
    "guidance_scale": 7.5,
    "strength": 0.8,
    "seed": 42
  },
  "processing_time": 2.45,
  "file_info": {
    "filename": "image.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 245760
  }
}
```

### Error Handling
- **400 Bad Request**: Invalid file type, file too large, or invalid parameters
- **500 Internal Server Error**: Processing failed
- **503 Service Unavailable**: Model not loaded

## 🐳 Docker Deployment

### Multi-Mode Support
The Docker container supports multiple deployment modes:

```bash
# RunPod Serverless (default)
docker run your-image

# FastAPI Web Server  
docker run -e SERVER_MODE=fastapi -p 8000:8000 your-image

# Debug mode
docker run -e SERVER_MODE=debug your-image
```

### Environment Variables
- `SERVER_MODE`: `runpod` (default), `fastapi`, or `debug`
- Standard RunPod environment variables for serverless mode

## 🧪 Testing

### Automated Testing
```bash
python test_multipart_upload.py http://localhost:8000
```

### Manual Testing with curl
See examples above or visit the interactive API docs at:
```
http://localhost:8000/docs
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

## 📱 Integration Examples

### JavaScript/Web
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('prompt', 'Add beautiful sunset colors');
formData.append('strength', '0.8');

const response = await fetch('/api/v1/upload/background-replacement', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.success) {
  // Display result.image (base64 encoded)
  imageElement.src = `data:image/jpeg;base64,${result.image}`;
}
```

### Python
```python
import requests

files = {'file': open('image.jpg', 'rb')}
data = {
    'prompt': 'Add flowers and butterflies',
    'num_inference_steps': 25,
    'guidance_scale': 7.5
}

response = requests.post(
    'http://localhost:8000/api/v1/upload/edit',
    files=files,
    data=data
)

result = response.json()
if result['success']:
    with open('edited_image.jpg', 'wb') as f:
        f.write(base64.b64decode(result['image']))
```

### iOS/Swift
```swift
var request = URLRequest(url: URL(string: "http://localhost:8000/api/v1/upload/edit")!)
request.httpMethod = "POST"

let boundary = "Boundary-\(UUID().uuidString)"
request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

var body = Data()
// Add image file
body.append("--\(boundary)\r\n".data(using: .utf8)!)
body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
body.append(imageData)
body.append("\r\n".data(using: .utf8)!)

// Add prompt
body.append("--\(boundary)\r\n".data(using: .utf8)!)
body.append("Content-Disposition: form-data; name=\"prompt\"\r\n\r\n".data(using: .utf8)!)
body.append("Add beautiful flowers".data(using: .utf8)!)
body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

request.httpBody = body
```

## 🔄 Backward Compatibility

All existing endpoints continue to work:
- `/api/v1/edit` - Edit with base64 image
- `/api/v1/generate` - Text-to-image generation
- `/api/v1/health` - Health check
- RunPod serverless endpoints remain unchanged

## ⚡ Performance Tips

1. **Optimize image size**: Resize images before upload for faster processing
2. **Use appropriate steps**: 20-30 steps for most tasks, 50+ for high quality
3. **Batch processing**: Upload multiple files in sequence for efficiency
4. **Choose the right strength**: 0.5-0.7 for subtle edits, 0.8-1.0 for major changes

---

🎉 **Your AI Image Editing Server now supports both serverless AND traditional web server deployments with easy file uploads!**