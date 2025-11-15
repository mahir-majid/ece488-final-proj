"""
This script generates images using Stable Diffusion with a Hugging Face
inference endpoint. Need to provide a HUGGINGFACE_API_KEY environment variable
and also a HUGGINGFACE_ENDPOINT environment variable.
Required: NEW_PERSON_OUTPUT_IMAGE_PATH environment variable (full path to output image file).
Optional: NEW_PERSON_PROMPT environment variable (defaults to "A crowd of people with three happy people and one sad person ").
"""

import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def generate_image(prompt: str, api_key: str, endpoint: str) -> bytes:
    """Generate an image using Hugging Face Inference API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(endpoint, headers=headers, json={"inputs": prompt})
    response.raise_for_status()
    
    # Handle response - could be base64 string or dict with image data
    result = response.json()
    if isinstance(result, dict) and 'image' in result:
        image_data = result['image']
    elif isinstance(result, str):
        image_data = result
    else:
        image_data = result
    
    # Decode if base64 string
    if isinstance(image_data, str):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        return base64.b64decode(image_data)
    
    return image_data


def main():
    """Generate and save Stable Diffusion images."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    endpoint = os.getenv("HUGGINGFACE_ENDPOINT")
    
    if not api_key or not endpoint:
        raise ValueError("HUGGINGFACE_API_KEY and HUGGINGFACE_ENDPOINT must be set")
    
    output_image_path = os.getenv("NEW_PERSON_OUTPUT_IMAGE_PATH")
    if not output_image_path:
        raise ValueError("NEW_PERSON_OUTPUT_IMAGE_PATH must be set")
    
    filepath = Path(output_image_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    prompt = os.getenv("NEW_PERSON_PROMPT", "A crowd of people with three happy people and one sad person ")
    prompts = [prompt]
    
    print(f"Generating {len(prompts)} images...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt}")
        try:
            image_bytes = generate_image(prompt, api_key, endpoint)
            
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nDone! Image saved to: {filepath}")


if __name__ == "__main__":
    main()
 