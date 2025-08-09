#!/bin/bash

echo ""
echo "Service Information:"
echo "  • API URL: http://localhost:8000"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/health"
echo "  • Video Gallery: http://localhost:8000/gallery"
echo ""
echo "Management Commands:"
echo "  • View logs: docker-compose logs -f"
echo "  • Stop service: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Shell access: docker-compose exec video-api bash"
echo ""
echo "Generate your first video:"
echo "  curl -X POST 'http://localhost:8000/generate?prompt=A%20cat%20walking&duration=3'"ript builds and deploys the video generation API with all dependencies

set -e  # Exit on any error

echo "Video Generation API Deployment Script"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Docker and Docker Compose are available"

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down || true

# Build and start the service
echo "Building Docker image..."
docker-compose build --no-cache

echo "Starting the service..."
docker-compose up -d

# Wait for the service to start
echo "Waiting for service to start..."
sleep 30

# Test the API
echo "Testing the API..."
python3 tests/quick_test.py --wait || {
    echo "API tests failed. Checking logs..."
    docker-compose logs video-api
    exit 1
}

echo "Deployment completed successfully!"
echo ""
echo "Service Information:"
echo "  • API URL: http://localhost:8000"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/health"
echo "  • Video Gallery: http://localhost:8000/gallery"
echo ""
echo "Management Commands:"
echo "  • View logs: docker-compose logs -f"
echo "  • Stop service: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Shell access: docker-compose exec video-api bash"
echo ""
echo "Generate your first video:"
echo "  curl -X POST 'http://localhost:8000/generate?prompt=A%20cat%20walking&duration=3'"
