import unittest
import requests
import os


class TestAlternativeModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Test cases with very simple prompts for better model performance
        self.test_cases = [
            ("A bird flying", 12, 256),
            ("Ocean waves", 12, 256), 
            ("Fire burning", 12, 256),
        ]
    
    def _test_simple_generation_case(self, prompt, fps, resolution):
        """Helper method to test simple, high-quality generation."""
        print(f"\nTesting simple generation with prompt: '{prompt}'")
        
        # Use minimal parameters for maximum quality
        payload = {
            "prompt": prompt,
            "duration": 2,  # Shorter for better quality
            "fps": fps,
            "resolution": resolution,
            "num_inference_steps": 20,  # Lower for better stability
            "guidance_scale": 7.0,      # Moderate guidance
        }
        
        # Make request
        response = requests.post(f"{self.base_url}/generate", params=payload, timeout=60)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        print(f"Simple generation successful!")
        print(f"Generation time: {data.get('generation_time_seconds', 'N/A')} seconds")
        print(f"Model used: {data.get('model_used', 'N/A')}")
        print(f"File size: {data.get('file_size_mb', 'N/A')} MB")
        
        # Get job ID and video URL
        job_id = data.get("job_id")
        if job_id:
            video_url = f"{self.base_url}/preview/{job_id}"
            print(f"Simple video preview URL: {video_url}")
    
    def test_simple_generation_bird(self):
        """Test simple generation with bird flying prompt."""
        prompt, fps, resolution = self.test_cases[0]
        self._test_simple_generation_case(prompt, fps, resolution)
    
    def test_simple_generation_ocean(self):
        """Test simple generation with ocean waves prompt."""
        prompt, fps, resolution = self.test_cases[1]
        self._test_simple_generation_case(prompt, fps, resolution)
    
    def test_simple_generation_fire(self):
        """Test simple generation with fire burning prompt."""
        prompt, fps, resolution = self.test_cases[2]
        self._test_simple_generation_case(prompt, fps, resolution)


if __name__ == '__main__':
    unittest.main()
