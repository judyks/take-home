#!/usr/bin/env python3
"""
Prompt Test - Test a specific prompt with the Video Generation API
"""

import requests
import time
import sys

def test_prompt(prompt, duration=4, style="cinematic", fps=12):
    """Test a specific prompt and monitor until completion"""
    base_url = "http://localhost:8000"
    session = requests.Session()
    
    print(f"Testing Prompt: '{prompt}'")
    print("=" * 50)
    
    # Check API health first
    print("1. Checking API status...")
    try:
        response = session.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            print(f"   API not healthy: HTTP {response.status_code}")
            return False
        
        data = response.json()
        if data.get("status") != "healthy":
            print(f"   API not healthy: {data.get('status')}")
            return False
            
        print(f"   API is healthy, GPU available: {data.get('gpu_available')}")
    except Exception as e:
        print(f"   Error checking API: {e}")
        return False
    
    # Check model status
    print("2. Checking model status...")
    try:
        response = session.get(f"{base_url}/model-status", timeout=10)
        if response.status_code != 200:
            print(f"   Model check failed: HTTP {response.status_code}")
            return False
            
        data = response.json()
        if data.get("status") != "loaded":
            print(f"   Model not loaded: {data.get('status')}")
            return False
            
        print(f"   Model loaded: {data.get('model_id')}")
    except Exception as e:
        print(f"   Error checking model: {e}")
        return False
    
    # Analyze the prompt
    print("3. Analyzing prompt...")
    try:
        response = session.post(f"{base_url}/analyze-prompt", 
                               params={"prompt": prompt}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   Complexity: {data.get('complexity', 'unknown')}")
            suggestions = data.get('suggestions', [])
            if suggestions:
                print(f"   Suggestions: {', '.join(suggestions)}")
        else:
            print(f"   Prompt analysis failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   Error analyzing prompt: {e}")
    
    # Generate video
    print("4. Starting video generation...")
    try:
        params = {
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "fps": fps
        }
        
        print(f"   Prompt: {prompt}")
        print(f"   Duration: {duration} seconds")
        print(f"   Style: {style}")
        print(f"   FPS: {fps}")
        
        response = session.post(f"{base_url}/generate", params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"   Generation failed: HTTP {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    pass
            return False
            
        data = response.json()
        if data.get("status") != "success":
            print(f"   Generation failed: {data.get('message', 'Unknown error')}")
            return False
            
        job_id = data.get("job_id")
        print(f"   Generation started successfully!")
        print(f"   Job ID: {job_id}")
        
    except Exception as e:
        print(f"   Error during generation: {e}")
        return False
    
    # Monitor progress
    print("5. Monitoring generation progress...")
    start_time = time.time()
    max_wait = 180  # 3 minutes max
    
    while time.time() - start_time < max_wait:
        try:
            response = session.get(f"{base_url}/status/{job_id}", timeout=10)
            if response.status_code != 200:
                print(f"   Status check failed: HTTP {response.status_code}")
                break
                
            data = response.json()
            status = data.get("status")
            elapsed = time.time() - start_time
            
            if status == "completed":
                print(f"   Generation completed in {elapsed:.1f} seconds!")
                
                # Get video info
                metadata = data.get("metadata", {})
                video_info = metadata.get("video_info", {})
                
                print(f"   Filename: {video_info.get('filename', 'Unknown')}")
                print(f"   File size: {video_info.get('file_size_mb', 0):.2f} MB")
                print(f"   Total frames: {video_info.get('total_frames', 0)}")
                print(f"   Generation time: {metadata.get('generation_time_seconds', 0):.1f}s")
                
                # Test download
                print("6. Testing download...")
                try:
                    download_response = session.get(f"{base_url}/download/{job_id}", timeout=30)
                    if download_response.status_code == 200:
                        content_type = download_response.headers.get('content-type', '')
                        file_size = len(download_response.content)
                        print(f"   Download successful: {file_size:,} bytes ({content_type})")
                        print(f"   Download URL: {base_url}/download/{job_id}")
                        print(f"   Preview URL: {base_url}/preview/{job_id}")
                    else:
                        print(f"   Download failed: HTTP {download_response.status_code}")
                except Exception as e:
                    print(f"   Download error: {e}")
                
                return True
                
            elif status == "failed":
                error_msg = data.get("error", "Unknown error")
                print(f"   Generation failed: {error_msg}")
                return False
                
            elif status == "processing":
                print(f"   Still processing... ({elapsed:.0f}s elapsed)")
                time.sleep(10)  # Wait 10 seconds before checking again
                
            else:
                print(f"   Unknown status: {status}")
                time.sleep(5)
                
        except Exception as e:
            print(f"   Error checking status: {e}")
            time.sleep(5)
    
    print(f"   Generation timed out after {max_wait} seconds")
    print(f"   Job may still be processing. Check status: {base_url}/status/{job_id}")
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_prompt.py 'Your prompt here' [duration] [style] [fps]")
        print("\nExamples:")
        print("  python test_prompt.py 'A cat walking in a garden'")
        print("  python test_prompt.py 'A red car driving down a road' 5")
        print("  python test_prompt.py 'A bird flying in the sky' 4 realistic 16")
        print("\nAvailable styles: cinematic, realistic, artistic")
        return 1
    
    prompt = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    style = sys.argv[3] if len(sys.argv) > 3 else "cinematic"
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    
    try:
        success = test_prompt(prompt, duration, style, fps)
        
        print("\n" + "=" * 50)
        if success:
            print("PROMPT TEST: SUCCESS!")
            print("Video generated and downloaded successfully.")
        else:
            print("PROMPT TEST: FAILED")
            print("Video generation did not complete successfully.")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
