#!/usr/bin/env python3

import requests
import time
import json
import sys
import os
from pathlib import Path

def test_video_generation_comprehensive():
    """
    Comprehensive test of video generation functionality.
    Tests both demo generation and validates that videos are created based on prompts.
    """
    base_url = "http://localhost:8000"
    
    print("Comprehensive Video Generation Testing")
    print("=" * 60)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Health check passed")
            print(f"   GPU Available: {health_data.get('gpu_available', False)}")
            print(f"   GPU Info: {health_data.get('gpu_info', {}).get('device_name', 'Unknown')}")
        else:
            print(f"   Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Health check error: {e}")
        return False
    
    # Test 2: Multiple Prompt Testing
    test_prompts = [
        "A cat walking in a garden",
        "A beautiful sunset over the ocean with waves",
        "A robot dancing in a futuristic city",
        "A butterfly flying over colorful flowers",
        "A train moving through mountain scenery"
    ]
    
    print(f"\n2. Testing video generation with {len(test_prompts)} different prompts...")
    
    generated_videos = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Prompt {i}: '{prompt}'")
        
        try:
            # Generate video using demo endpoint (since it's working)
            response = requests.post(f"{base_url}/demo-generate", params={"prompt": prompt})
            
            if response.status_code == 200:
                demo_data = response.json()
                job_id = demo_data.get('job_id')
                filename = demo_data.get('filename')
                
                print(f"      Generation initiated successfully")
                print(f"      Job ID: {job_id}")
                print(f"      Filename: {filename}")
                
                # Check job status
                status_response = requests.get(f"{base_url}/status/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"      Status: {status_data.get('status')}")
                    print(f"      Video Ready: {status_data.get('video_ready')}")
                    
                    if status_data.get('video_ready'):
                        # Try to download and analyze the video
                        download_response = requests.get(f"{base_url}/download/{job_id}")
                        if download_response.status_code == 200:
                            video_size = len(download_response.content)
                            print(f"      Video Size: {video_size:,} bytes ({video_size/1024:.1f} KB)")
                            
                            # Validate video content
                            content_type = download_response.headers.get('content-type', '')
                            if 'video' in content_type or 'mp4' in content_type:
                                print(f"      Valid video content type: {content_type}")
                            else:
                                print(f"      Unexpected content type: {content_type}")
                            
                            generated_videos.append({
                                'prompt': prompt,
                                'job_id': job_id,
                                'filename': filename,
                                'size_bytes': video_size,
                                'content_type': content_type
                            })
                        else:
                            print(f"      Failed to download video: {download_response.status_code}")
                    else:
                        print(f"      Video not ready yet")
                else:
                    print(f"      Failed to check status: {status_response.status_code}")
            else:
                print(f"      Generation failed: {response.status_code}")
                if response.text:
                    print(f"      Response: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"      Error generating video: {e}")
    
    # Test 3: Video Quality Analysis
    print(f"\n3. Video Generation Analysis")
    print("-" * 40)
    
    if generated_videos:
        print(f"   Successfully generated: {len(generated_videos)}/{len(test_prompts)} videos")
        
        sizes = [v['size_bytes'] for v in generated_videos]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        print(f"   Video sizes:")
        print(f"      • Average: {avg_size:,.0f} bytes ({avg_size/1024:.1f} KB)")
        print(f"      • Range: {min_size:,} - {max_size:,} bytes")
        
        # Check for reasonable video sizes (should be more than just a few KB)
        reasonable_videos = [v for v in generated_videos if v['size_bytes'] > 10000]  # > 10KB
        print(f"   Videos with reasonable size (>10KB): {len(reasonable_videos)}/{len(generated_videos)}")
        
        if len(reasonable_videos) == len(generated_videos):
            print(f"   All videos appear to be properly generated!")
        else:
            print(f"   Some videos may be too small - possible generation issues")
            
    else:
        print(f"   No videos were successfully generated")
    
    # Test 4: Check if videos vary by prompt
    print(f"\n4. Prompt Variation Analysis")
    print("-" * 40)
    
    if len(generated_videos) >= 2:
        # Check if different prompts produce different sized videos
        size_variation = max(sizes) - min(sizes)
        if size_variation > 1000:  # More than 1KB difference
            print(f"   Good variation between videos ({size_variation:,} bytes difference)")
            print("   This suggests the API is responding to different prompts")
        else:
            print(f"   Low variation between videos ({size_variation:,} bytes difference)")
            print("   Videos may be very similar - check if prompts are being processed")
            
        # Show individual results
        print(f"\n   Individual Results:")
        for video in generated_videos:
            print(f"      • '{video['prompt'][:30]}...' → {video['size_bytes']:,} bytes")
    else:
        print("   Not enough videos to compare variation")
    
    # Test 5: Try the advanced model endpoint (if available)
    print(f"\n5. Testing Advanced Model Loading")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/test-model-loading")
        if response.status_code == 200:
            data = response.json()
            print(f"   Model Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            
            if data.get('status') == 'success':
                print(f"   Advanced model is available!")
                
                # Try generating with the advanced model
                print(f"\n   Testing advanced generation...")
                adv_response = requests.post(f"{base_url}/generate", params={"prompt": "advanced test"})
                if adv_response.status_code == 200:
                    print(f"   Advanced generation works!")
                    adv_data = adv_response.json()
                    print(f"   Job ID: {adv_data.get('job_id')}")
                    print(f"   Model Used: {adv_data.get('model_used')}")
                else:
                    print(f"   Advanced generation failed: {adv_response.status_code}")
            else:
                print(f"   Advanced model not available")
        else:
            print(f"   Failed to check advanced model: {response.status_code}")
    except Exception as e:
        print(f"   Error testing advanced model: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"TESTING SUMMARY")
    print(f"=" * 60)
    
    if generated_videos:
        success_rate = (len(generated_videos) / len(test_prompts)) * 100
        print(f"Success Rate: {success_rate:.1f}% ({len(generated_videos)}/{len(test_prompts)} videos)")
        
        if success_rate > 80:
            print(f"EXCELLENT: Video generation is working well!")
        else:
            print(f"NEEDS IMPROVEMENT: Some videos failed to generate")
        
        print(f"\nGenerated Videos:")
        for v in generated_videos:
            print(f"   • {v['filename']} ({v['size_bytes']:,} bytes)")
            
    else:
        print("CRITICAL FAILURE: No videos were generated.")
        print("Please check the server logs for errors.")

    print(f"\nManual Testing:")
    print(f"   • Browser: {base_url}")
    print(f"   • API docs: {base_url}/docs")
    print(f"   • Health: {base_url}/health")
    print(f"   • Videos are saved in: outputs/videos/")

def check_output_directory():
    """Check what videos are in the output directory"""
    print(f"\nChecking output directory...")
    
    videos_dir = Path("/home/user/video-generation-api/outputs/videos")
    metadata_dir = Path("/home/user/video-generation-api/outputs/metadata")
    
    if videos_dir.exists():
        video_files = sorted(videos_dir.glob('*.mp4'), key=os.path.getmtime, reverse=True)
        print(f"   Videos in directory: {len(list(videos_dir.glob('*.mp4')))}")
        if video_files:
            print(f"   Recent videos:")
            for f in video_files[:5]:
                stat = f.stat()
                print(f"      • {f.name} ({stat.st_size:,} bytes, {time.ctime(stat.st_mtime)})")
    else:
        print(f"   Videos directory not found")
    
    if metadata_dir.exists():
        print(f"   Metadata files: {len(list(metadata_dir.glob('*.json')))}")
    else:
        print(f"   Metadata directory not found")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        print("Waiting 30 seconds for server to start...")
        time.sleep(30)
    
    # Check existing videos first
    check_output_directory()
    
    # Run comprehensive tests
    test_video_generation_comprehensive()
