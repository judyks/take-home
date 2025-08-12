import unittest
import requests
import os


class TestImprovedGeneration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Test parameters for higher-quality generation (optimized settings)
        self.test_cases = [
            ("A red ball bouncing slowly", 16, 512),
            ("A cat sitting in sunlight", 16, 512),
            ("Rain drops on glass", 16, 512),
        ]
    
    def _test_high_quality_generation_case(self, prompt, fps, resolution):
        """Helper method to test a single case of high quality generation."""
        print(f"\nTesting high quality generation with prompt: '{prompt}'")
        
        # quality parameters for better output
        payload = {
            "prompt": prompt,
            "duration": 3, 
            "fps": fps,
            "resolution": resolution,
            "num_inference_steps": 40, 
            "guidance_scale": 9.0,      
            "seed": None,               # Let model choose for variety
        }
        
        response = requests.post(f"{self.base_url}/generate", params=payload, timeout=90)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        print(f"High-quality generation successful!")
        print(f"Generation time: {data.get('generation_time_seconds', 'N/A')} seconds")
        print(f"Model used: {data.get('model_used', 'N/A')}")
        print(f"Device used: {data.get('device_used', 'N/A')}")
        print(f"File size: {data.get('file_size_mb', 'N/A')} MB")
        
        # Get job ID and video URL
        job_id = data.get("job_id")
        if job_id:
            video_url = f"{self.base_url}/preview/{job_id}"
            print(f"High quality video preview URL: {video_url}")
            
            # verify video metadata
            try:
                meta_response = requests.get(f"{self.base_url}/metadata/{job_id}", timeout=10)
                if meta_response.status_code == 200:
                    metadata = meta_response.json()
                    
                    self.assertEqual(metadata.get("fps", 0), fps, "FPS setting was not applied")
                    self.assertEqual(metadata.get("resolution", 0), resolution, "Resolution setting was not applied")
                    print(f"Quality parameters verified: FPS={fps}, Resolution={resolution}")
                else:
                    print(f"Metadata endpoint returned {meta_response.status_code}, skipping quality verification")
            except requests.exceptions.RequestException as e:
                print(f"Could not verify metadata: {e}")
    
    def test_high_quality_generation_red_ball(self):
        """Test high quality generation with red ball bouncing prompt."""
        prompt, fps, resolution = self.test_cases[0]
        self._test_high_quality_generation_case(prompt, fps, resolution)
    
    def test_high_quality_generation_cat_stretching(self):
        """Test high quality generation with cat stretching prompt."""
        prompt, fps, resolution = self.test_cases[1]
        self._test_high_quality_generation_case(prompt, fps, resolution)
    
    def test_high_quality_generation_raindrops(self):
        """Test high quality generation with raindrops prompt."""
        prompt, fps, resolution = self.test_cases[2]
        self._test_high_quality_generation_case(prompt, fps, resolution)


if __name__ == '__main__':
    unittest.main()
