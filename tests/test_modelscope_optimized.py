import unittest
import requests
import os


class TestModelScopeOptimized(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for ModelScope-optimized generation."""
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Test cases optimized for ModelScope model characteristics
        self.test_cases = [
            ("A dog running in a park", 8, 256),
            ("A flower blooming", 8, 256),
            ("Clouds moving in sky", 8, 256),
        ]
    
    def _test_modelscope_generation_case(self, prompt, fps, resolution):
        """Helper method optimized for ModelScope model."""
        print(f"\nTesting ModelScope-optimized generation with prompt: '{prompt}'")
        
        # Parameters optimized for ModelScope model
        payload = {
            "prompt": prompt,
            "duration": 2,  # Shorter duration often works better
            "fps": fps,
            "resolution": resolution,
            "num_inference_steps": 25,  # Optimal for ModelScope
            "guidance_scale": 15.0,     # Higher guidance often works better for ModelScope
            "negative_prompt": "blurry, low quality, distorted",  # Negative prompt for quality
        }
        
        # Make request
        response = requests.post(f"{self.base_url}/generate", params=payload, timeout=90)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        print(f"ModelScope-optimized generation successful!")
        print(f"Generation time: {data.get('generation_time_seconds', 'N/A')} seconds")
        print(f"Model used: {data.get('model_used', 'N/A')}")
        print(f"File size: {data.get('file_size_mb', 'N/A')} MB")
        
        # Get job ID and video URL
        job_id = data.get("job_id")
        if job_id:
            video_url = f"{self.base_url}/preview/{job_id}"
            print(f"ModelScope video preview URL: {video_url}")
    
    def test_modelscope_dog_running(self):
        """Test ModelScope-optimized generation with dog running."""
        prompt, fps, resolution = self.test_cases[0]
        self._test_modelscope_generation_case(prompt, fps, resolution)
    
    def test_modelscope_flower_blooming(self):
        """Test ModelScope-optimized generation with flower blooming."""
        prompt, fps, resolution = self.test_cases[1]
        self._test_modelscope_generation_case(prompt, fps, resolution)
    
    def test_modelscope_clouds_moving(self):
        """Test ModelScope-optimized generation with clouds moving."""
        prompt, fps, resolution = self.test_cases[2]
        self._test_modelscope_generation_case(prompt, fps, resolution)


if __name__ == '__main__':
    unittest.main()
