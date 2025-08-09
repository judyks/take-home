# API Documentation

## Overview

The Video Generation API provides RESTful endpoints for generating videos from text prompts using the Lightricks/LTX-Video-0.9.7-distilled model.

**Base URL**: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication. Optional API key authentication can be enabled via environment variables.

## Endpoints

### Core Endpoints

#### GET `/`
**Description**: API homepage with overview and links.

**Response**: HTML page with API information.

#### GET `/health`
**Description**: Health check endpoint with system status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-09T00:47:34.123456",
  "gpu_available": true,
  "gpu_info": {
    "device_name": "NVIDIA H100 80GB HBM3",
    "memory_total_gb": 80.0,
    "memory_allocated_gb": 0.0,
    "memory_reserved_gb": 0.0
  }
}
```

#### GET `/docs`
**Description**: Interactive API documentation (Swagger UI).

#### GET `/model-status`
**Description**: Check video generation model loading status.

**Response**:
```json
{
  "status": "loaded",
  "model_name": "Lightricks/LTX-Video-0.9.7-distilled",
  "device": "cuda",
  "memory_usage_gb": 12.5
}
```

### Video Generation

#### POST `/generate`
**Description**: Generate a video from a text prompt.

**Parameters**:
- `prompt` (string, required): Text description of the video to generate
- `duration` (integer, optional): Video duration in seconds (default: 3, max: 10)
- `resolution` (integer, optional): Video resolution (default: 512, max: 512)
- `fps` (integer, optional): Frames per second (default: 8)

**Example Request**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -d "prompt=A cat walking in a garden" \
  -d "duration=3"
```

**Response**:
```json
{
  "status": "success",
  "message": "Video generated and saved successfully with LTX-Video-0.9.7-distilled!",
  "job_id": "2c49d3d2",
  "filename": "video_20250809_004734_2c49d3d2.mp4",
  "prompt": "A cat walking in a garden",
  "duration": 3,
  "frames_generated": 24,
  "resolution": "512x512",
  "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
  "file_size_mb": 2.2,
  "fps": 8.0,
  "generation_time_seconds": 9.99,
  "device_used": "cpu",
  "download_url": "/download/2c49d3d2",
  "preview_url": "/preview/2c49d3d2"
}
```

### Video Management

#### GET `/status/{job_id}`
**Description**: Get status and metadata for a generation job.

**Parameters**:
- `job_id` (string): The job ID returned from `/generate`

**Response**:
```json
{
  "job_id": "2c49d3d2",
  "status": "completed",
  "prompt": "A cat walking in a garden",
  "duration": 3,
  "created_at": "2025-08-09T00:47:35",
  "completed_at": "2025-08-09T00:47:45",
  "file_size_mb": 2.2,
  "generation_time_seconds": 9.99
}
```

#### GET `/download/{job_id}`
**Description**: Download the generated video file.

**Parameters**:
- `job_id` (string): The job ID returned from `/generate`

**Response**: MP4 video file download.

#### GET `/preview/{job_id}`
**Description**: Preview video in browser with metadata.

**Parameters**:
- `job_id` (string): The job ID returned from `/generate`

**Response**: HTML page with video player and metadata.

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters (empty prompt, duration too long, etc.)
- `404 Not Found`: Job ID not found or video file missing
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error during generation

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "validation_error",
  "job_id": "abc123def"
}
```

### Common Errors

#### Validation Errors
- Empty prompt: `{"detail": "Prompt cannot be empty"}`
- Duration too long: `{"detail": "Duration cannot exceed 10 seconds"}`
- Invalid duration: `{"detail": "Duration must be a positive number"}`

#### Generation Errors
- Out of memory: `{"detail": "GPU out of memory, try shorter duration"}`
- Model not loaded: `{"detail": "Model not loaded, please wait"}`

## Rate Limiting

Rate limiting is optional and disabled by default. When enabled:
- 10 requests per minute per IP address
- Exceeded limit returns `429 Too Many Requests`

## Examples

### Basic Video Generation

```bash
# Generate a 3-second video
curl -X POST "http://localhost:8000/generate?prompt=A%20beautiful%20sunset&duration=3"

# Check generation status
curl "http://localhost:8000/status/abc123def"

# Download the video
curl -o sunset.mp4 "http://localhost:8000/download/abc123def"
```

### Using with Python

```python
import requests
import time

# Generate video
response = requests.post(
    "http://localhost:8000/generate",
    data={
        "prompt": "A cat walking in a garden",
        "duration": 3
    }
)
result = response.json()
job_id = result["job_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/status/{job_id}")
print(status_response.json())

# Download video
video_response = requests.get(f"http://localhost:8000/download/{job_id}")
with open(f"video_{job_id}.mp4", "wb") as f:
    f.write(video_response.content)
```

### Using with JavaScript

```javascript
// Generate video
const generateVideo = async () => {
  const formData = new FormData();
  formData.append('prompt', 'A cat walking in a garden');
  formData.append('duration', '3');
  
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result.job_id;
};

// Check status
const checkStatus = async (jobId) => {
  const response = await fetch(`http://localhost:8000/status/${jobId}`);
  return response.json();
};

// Download video
const downloadVideo = (jobId) => {
  window.open(`http://localhost:8000/download/${jobId}`, '_blank');
};
```

## Configuration Parameters

The API behavior can be configured via environment variables:

### Model Settings
- `MODEL_NAME`: HuggingFace model name (default: "Lightricks/LTX-Video-0.9.7-distilled")
- `MODEL_DEVICE`: Device to use (auto/cuda/cpu, default: "auto")
- `MODEL_CACHE_DIR`: Directory for model caching

### Video Generation Limits
- `MAX_VIDEO_DURATION`: Maximum video duration in seconds (default: 10)
- `DEFAULT_VIDEO_DURATION`: Default duration when not specified (default: 3)
- `MAX_RESOLUTION`: Maximum video resolution (default: 512)
- `DEFAULT_FPS`: Default frames per second (default: 8)

### API Settings
- `API_HOST`: Host to bind to (default: "0.0.0.0")
- `API_PORT`: Port to listen on (default: 8000)
- `API_WORKERS`: Number of worker processes (default: 1)

### Security Settings
- `ENABLE_RATE_LIMIT`: Enable rate limiting (default: false)
- `RATE_LIMIT_PER_MINUTE`: Requests per minute when rate limiting enabled (default: 10)
- `API_KEY_REQUIRED`: Require API key for requests (default: false)
- `API_KEY`: API key value when required

### Storage Settings
- `OUTPUT_DIR`: Directory for storing generated videos (default: "./outputs")
- `MAX_STORAGE_GB`: Maximum storage usage in GB (default: 100)
- `CLEANUP_AFTER_DAYS`: Auto-cleanup videos after N days (default: 7)

## Monitoring

### Health Monitoring
- Use `/health` endpoint for health checks
- Monitor GPU memory usage via `/health` response
- Check model status with `/model-status`

### Logging
- Logs are available via `docker-compose logs video-api`
- Log level configurable via `LOG_LEVEL` environment variable
- Structured JSON logging for production environments

### Metrics
The API provides basic metrics in responses:
- Generation time per video
- GPU memory usage
- File sizes
- Timestamp information
