#!/usr/bin/env python3
"""
Test script for the improved video generation API
Tests the refined prompt optimization and parameter settings
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test if the API is running"""
    print("Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"API is healthy - GPU available: {data.get('gpu_available', False)}")
            return True
        else:
            print(f"API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Could not connect to API: {e}")
        return False

def test_model_status():
    """Check if the model is loaded"""
    print("\nChecking model status...")
    try:
        response = requests.get(f"{BASE_URL}/model-status")
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'loaded':
                print(f"Model is loaded: {data.get('model_id', 'Unknown')}")
                return True
            else:
                print(f"Model not loaded: {data.get('message', 'Unknown')}")
                return False
        else:
            print(f"Model status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Could not check model status: {e}")
        return False

def test_simple_generation():
    """Test video generation with a simple prompt"""
    print("\nTesting simple video generation...")
    
    test_params = {
        "prompt": "a cat walking in a garden",
        "duration": 3,
        "fps": 8
    }
    
    print(f"Prompt: '{test_params['prompt']}'")
    print("Starting generation...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            params=test_params,  
            timeout=300 
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"Video generated successfully!")
            print(f"Job ID: {data.get('job_id', 'Unknown')}")
            print(f"Filename: {data.get('filename', 'Unknown')}")
            print(f"Total time: {generation_time:.1f}s")
            print(f"File size: {data.get('file_size_mb', 'Unknown')} MB")
            print(f"Enhanced prompt: '{data.get('enhanced_prompt', 'Unknown')}'")
            print(f"Parameters used:")
            params = data.get('generation_parameters', {})
            print(f"   - Guidance scale: {params.get('guidance_scale', 'Unknown')}")
            print(f"   - Inference steps: {params.get('num_inference_steps', 'Unknown')}")
            print(f"   - Negative prompt: {params.get('negative_prompt', 'Unknown')}")
            
            print(f"\nAccess links:")
            print(f"   Download: {BASE_URL}/download/{data.get('job_id', '')}")
            print(f"   Preview: {BASE_URL}/preview/{data.get('job_id', '')}")
            
            return data.get('job_id')
        else:
            print(f"Generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Error response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out (generation took too long)")
        return None
    except Exception as e:
        print(f"Generation request failed: {e}")
        return None

def test_style_generation():
    """Test video generation with a style preset"""
    print("\nTesting style preset generation...")
    
    test_params = {
        "prompt": "a bird flying over mountains",
        "duration": 3,
        "fps": 8,
        "style": "cinematic"
    }
    
    print(f"Prompt: '{test_params['prompt']}'")
    print(f"Style: {test_params['style']}")
    print("Starting generation...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            params=test_params,
            timeout=300
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"Styled video generated successfully!")
            print(f"Job ID: {data.get('job_id', 'Unknown')}")
            print(f"Generation time: {generation_time:.1f}s")
            print(f"Enhanced prompt: '{data.get('enhanced_prompt', 'Unknown')}'")
            
            return data.get('job_id')
        else:
            print(f"Styled generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Styled generation request failed: {e}")
        return None

def test_comparison_prompt():
    """Test with a prompt that should show improvement over the old system"""
    print("\nTesting improved prompt handling...")
    
    test_params = {
        "prompt": "dog running on beach",  # simple prompt
        "duration": 3,
        "fps": 8
    }
    
    print(f"Testing simple prompt: '{test_params['prompt']}'")
    print("This should work better with our refined optimization...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            params=test_params,
            timeout=300
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"Simple prompt video generated!")
            print(f"Job ID: {data.get('job_id', 'Unknown')}")
            print(f"Generation time: {generation_time:.1f}s")
            print(f"Original prompt: '{data.get('prompt', 'Unknown')}'")
            print(f"Enhanced prompt: '{data.get('enhanced_prompt', 'Unknown')}'")
            
            params = data.get('generation_parameters', {})
            print(f"Conservative parameters used:")
            print(f"   - Guidance scale: {params.get('guidance_scale', 'Unknown')} (lower = more creative)")
            print(f"   - Steps: {params.get('num_inference_steps', 'Unknown')} (balanced quality/speed)")
            
            return data.get('job_id')
        else:
            print(f"Simple prompt generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Simple prompt generation failed: {e}")
        return None

def main():
    print("Testing Improved Video Generation API")
    print("=" * 50)
    
    if not test_health():
        print("Cannot proceed - API is not accessible")
        return
    
    if not test_model_status():
        print("Cannot proceed - Model is not loaded")
        return
    
    print("\nRunning video generation tests...")
    
    job_id_1 = test_simple_generation()
    job_id_2 = test_style_generation()
    job_id_3 = test_comparison_prompt()
    
    print("\nTest Summary:")
    print("=" * 50)
    successful_tests = sum([1 for job_id in [job_id_1, job_id_2, job_id_3] if job_id])
    total_tests = 3
    
    print(f"Successful generations: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        print(f"\nGenerated video links:")
        
        if job_id_1:
            print(f"\nVideo 1 (Simple generation):")
            print(f"   Preview: {BASE_URL}/preview/{job_id_1}")
            print(f"   Download: {BASE_URL}/download/{job_id_1}")
            
        if job_id_2:
            print(f"\nVideo 2 (Style preset):")
            print(f"   Preview: {BASE_URL}/preview/{job_id_2}")
            print(f"   Download: {BASE_URL}/download/{job_id_2}")
            
        if job_id_3:
            print(f"\nVideo 3 (Improved prompt):")
            print(f"   Preview: {BASE_URL}/preview/{job_id_3}")
            print(f"   Download: {BASE_URL}/download/{job_id_3}")
    else:
        print(f"no successful generations.")

if __name__ == "__main__":
    main()
