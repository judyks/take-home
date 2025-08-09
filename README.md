# Video Generation API

API for generating videos from text prompts using the Lightricks/LTX-Video-0.9.7-distilled model. Built with FastAPI and Docker for easy deployment and production use.

## Features

- Create videos from descriptive text prompts
- GPU acceleration with CPU fallback
- Docker containerization with health checks and monitoring
- Built-in video preview and download endpoints

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, CPU fallback available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd take-home
   ```

2. **Start the service**
   ```bash
   docker-compose up -d
   ```

3. **Verify the installation**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Generate your first video**
   ```bash
   curl -X POST "http://localhost:8000/generate" \
     -d "prompt=A cat walking in a garden" \
     -d "duration=3"
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Project Structure

```
take-home/
├── .vscode/
│   └── settings.json        # VS Code project settings (linting, formatting)
├── src/
│   └── main.py              # Main FastAPI application
├── configs/
│   └── settings.py          # Configuration management
├── outputs/
│   ├── videos/              # Generated video files
│   └── metadata/            # Video metadata storage
├── tests/
│   ├── test_video_generation.py
│   └── quick_test.py
├── .env.example           # Environment configuration template
├── .gitignore             # Git ignore file (excludes venv, .env, etc.)
├── docker-compose.yml     # Production deployment
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── deploy.sh              # Deployment script
├── demo.py                # Usage examples
├── API_DOCUMENTATION.md   
└── README.md             
```

## Development

### Local Development Setup

1. **Create a virtual environment** (outside the project directory)
   ```bash
   cd .. 
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Navigate back to the project and install dependencies**
   ```bash
   cd take-home
   pip install -r requirements.txt
   ```

3. **Run the development server**
   ```bash
   python src/main.py
   ```

### VS Code Setup

The project includes VS Code settings for consistent development:
- **Python interpreter**: Automatically points to `../venv/bin/python`
- **Linting**: Flake8 enabled for code quality
- **Formatting**: Black formatter for consistent code style
- **Environment**: Auto-activates virtual environment in terminal

### Testing

Run the test suite:
```bash
python tests/test_video_generation.py
python tests/quick_test.py
```

### Environment Configuration

Copy the environment template and customize as needed:
```bash
cp .env.example .env
```

### Configuration

The application can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |
| `MODEL_DEVICE` | `auto` | Device to use (auto/cuda/cpu) |
| `MAX_VIDEO_DURATION` | `10` | Maximum video duration (seconds) |
| `OUTPUT_DIR` | `./outputs` | Output directory for videos |

## Model Information

- **Model**: Lightricks/LTX-Video-0.9.7-distilled
- **Input**: Text prompts
- **Output**: MP4 videos (512x512 resolution)
- **Duration**: 1-10 seconds
- **Frame Rate**: 8 FPS (configurable)

## Deployment

### Production Deployment

1. **Deploy with Docker Compose**
   ```bash
   ./deploy.sh
   ```

2. **Scale the service**
   ```bash
   docker-compose up -d --scale video-api=3
   ```

3. **Monitor logs**
   ```bash
   docker-compose logs -f video-api
   ```


## Monitoring 

- **Health Endpoint**: `/health` - System status and GPU info
- **Model Status**: `/model-status` - Model loading and memory usage
- **Metrics**: Built-in generation time and resource usage tracking
- **Logs**: Structured JSON logging


## API Documentation
For detailed API usage, see [API_DOCUMENTATION.md](./API_DOCUMENTATION.md).
