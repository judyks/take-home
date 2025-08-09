#!/usr/bin/env python3
"""
Video Generation API Demo Script
Demonstrates end-to-end usage of the video generation API
"""

import requests
import time
import json
import os
import sys
from typing import Dict, Any

class VideoGenerationDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_api_health(self) -> bool:
        print("Checking API Health...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   API Status: {data.get('status')}")
                print(f"   GPU Available: {data.get('gpu_available')}")
                if data.get('gpu_info'):
                    gpu_info = data['gpu_info']
                    print(f"   GPU: {gpu_info.get('device_name')}")
                    print(f"   GPU Memory: {gpu_info.get('memory_allocated_gb', 0):.1f}GB allocated")
                return True
            else:
                print(f"   Health check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   Cannot connect to API: {e}")
            print("   Make sure the API is running: docker-compose up -d")
            return False
    
    def check_model_status(self) -> bool:
        """Check model loading status"""
        print("\nChecking Model Status...")
        try:
            response = self.session.get(f"{self.base_url}/model-status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                print(f"   Model Status: {status}")
                
                if status == "not_loaded":
                    print("   Loading model (this may take a few minutes)...")
                    load_response = self.session.get(f"{self.base_url}/test-model-loading", timeout=300)
                    if load_response.status_code == 200:
                        load_data = load_response.json()
                        if load_data.get('status') == 'success':
                            print(f"   Model loaded: {load_data.get('model_id')}")
                            return True
                        else:
                            print(f"   Model loading status: {load_data.get('status')}")
                            print(f"   Message: {load_data.get('message')}")
                            return load_data.get('status') in ['success', 'pytorch_unavailable']
                    else:
                        print(f"   Model loading failed: HTTP {load_response.status_code}")
                        return False
                elif status == "loaded":
                    print(f"   Model ready: {data.get('model_id')}")
                    return True
                else:
                    print(f"   Unknown status: {status}")
                    return False
            else:
                print(f"   Status check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   Model status check failed: {e}")
            return False
    
    def generate_demo_video(self, prompt: str, duration: int = 3) -> Dict[str, Any]:
        print(f"\nGenerating Video...")
        print(f"   Prompt: '{prompt}'")
        print(f"   Duration: {duration} seconds")
        
        try:
            # Start generation
            print("   Starting generation...")
            response = self.session.post(
                f"{self.base_url}/generate",
                params={"prompt": prompt, "duration": duration},
                timeout=600  # 10 minutes
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                
                if status == "success":
                    job_id = data.get('job_id')
                    print(f"   Generation successful!")
                    print(f"   Job ID: {job_id}")
                    print(f"   Filename: {data.get('filename')}")
                    print(f"   Frames: {data.get('frames_generated')}")
                    print(f"   File Size: {data.get('file_size_mb')} MB")
                    print(f"   Generation Time: {data.get('generation_time_seconds')} seconds")
                    print(f"   Device: {data.get('device_used')}")
                    
                    return {
                        'success': True,
                        'job_id': job_id,
                        'data': data
                    }
                    
                elif status == "partial_success":
                    print(f"   Partial success: {data.get('message')}")
                    return {
                        'success': False,
                        'error': data.get('message')
                    }
                    
                else:
                    print(f"   Generation failed: {data.get('message')}")
                    return {
                        'success': False,
                        'error': data.get('message')
                    }
                    
            elif response.status_code == 503:
                error_detail = response.json().get('detail', 'Service unavailable')
                print(f"   Service unavailable: {error_detail}")
                return {
                    'success': False,
                    'error': error_detail
                }
                
            else:
                print(f"   Generation failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    pass
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            print("   Generation timed out (>10 minutes)")
            return {
                'success': False,
                'error': "Generation timeout"
            }
        except requests.exceptions.RequestException as e:
            print(f"   Generation request failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a generation job"""
        print(f"\nChecking Job Status...")
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                print(f"   Status: {status}")
                print(f"   Video Ready: {data.get('video_ready')}")
                
                if data.get('metadata'):
                    metadata = data['metadata']
                    print(f"   Prompt: {metadata.get('prompt')}")
                    print(f"   Duration: {metadata.get('duration')}s")
                    print(f"   Created: {metadata.get('created_at', '')[:19].replace('T', ' ')}")
                
                return {
                    'success': True,
                    'data': data
                }
            else:
                print(f"   Status check failed: HTTP {response.status_code}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                }
        except requests.exceptions.RequestException as e:
            print(f"   Status check failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def download_video(self, job_id: str, filename: str = None) -> bool:
        """Download a generated video"""
        print(f"\nDownloading Video...")
        try:
            response = self.session.get(f"{self.base_url}/download/{job_id}", timeout=60)
            if response.status_code == 200:
                # Determine filename
                if not filename:
                    content_disposition = response.headers.get('content-disposition', '')
                    if 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[1].strip('"')
                    else:
                        filename = f"video_{job_id}.mp4"
                
                # save file
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                print(f"   Downloaded: {filename}")
                print(f"   Size: {file_size / (1024*1024):.1f} MB")
                print(f"   Location: {os.path.abspath(filename)}")
                return True
                
            elif response.status_code == 404:
                print(f"   Video not found for job {job_id}")
                return False
            else:
                print(f"   Download failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   Download failed: {e}")
            return False
    
    def show_api_info(self):
        """Show API information and available endpoints"""
        print(f"\nAPI Information...")
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   Service: {data.get('message', 'Video Generation API')}")
                print(f"   Model: {data.get('model')}")
                print(f"   Documentation: {self.base_url}/docs")
                
                if data.get('recent_videos'):
                    print(f"   Recent Videos: {len(data['recent_videos'])} found")
                
                return True
            else:
                print(f"   API info failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   API info failed: {e}")
            return False
    
    def run_demo(self, prompt: str = "A cat walking in a garden", duration: int = 3):
        print("Video Generation API Demo")
        print("=" * 50)
        
        # api health check
        if not self.check_api_health():
            print("\nAPI is not available. Demo cannot continue.")
            return False
        
        # api info
        self.show_api_info()
        
        # ai model status
        if not self.check_model_status():
            print("\nModel is not ready, but continuing with demo...")
        
        # generate video
        result = self.generate_demo_video(prompt, duration)
        
        if not result['success']:
            print(f"\nVideo generation failed: {result['error']}")
            print("\nThis might be expected if:")
            print("   • Model is not loaded (PyTorch/CUDA issues)")
            print("   • GPU memory is insufficient")
            print("   • Model download failed")
            print(f"\nYou can still explore the API:")
            print(f"   • Documentation: {self.base_url}/docs")
            print(f"   • Health: {self.base_url}/health")
            return False
        
        job_id = result['job_id']
        
        self.check_job_status(job_id)
        
        # download video
        download_success = self.download_video(job_id)
        if download_success:
            print("\nDemo Complete!")
            print("=" * 50)
            print(f"Successfully generated and downloaded video")
            print(f"Job ID: {job_id}")
            print(f"Prompt: '{prompt}'")
            print(f"\nNext Steps:")
            print(f"   • Preview online: {self.base_url}/preview/{job_id}")
            print(f"   • Generate more: {self.base_url}/docs")
            return True
        else:
            print("\nDemo completed with issues")
            print("Video was generated but download failed")
            return False

def main():
    """Main demo function"""
    print("Video Generation API Demo")
    print("Initializing...")
    
    # get configuration
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    prompt = os.getenv("DEMO_PROMPT", "A cat walking in a garden")
    duration = int(os.getenv("DEMO_DURATION", "3"))
    
    # allow command line overrides
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("http"):
            base_url = sys.argv[1]
        else:
            prompt = " ".join(sys.argv[1:])
    
    demo = VideoGenerationDemo(base_url)
    success = demo.run_demo(prompt, duration)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
