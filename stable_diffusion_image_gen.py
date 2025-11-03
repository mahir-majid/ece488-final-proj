"""
This script generates images using Stable Diffusion with a Hugging Face
inference endpoint. Need to provide a HUGGINGFACE_API_KEY environment variable
and also a HUGGINGFACE_ENDPOINT environment variable.
"""

import os
import base64
import requests
from pathlib import Path
from datetime import datetime


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
    
    output_dir = Path(__file__).parent / "output_HF_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        "A street sign that says 'STOP' in bold letters",
        "A billboard displaying 'Welcome to the City' with text",
        "A book cover with title 'Machine Learning' written on it",
        "A blackboard with '2 + 2 = 4' written in chalk",
        "A menu board showing 'Coffee $3.50' in large text",
        "A license plate with 'ABC123' on it",
        "A neon sign glowing 'OPEN' in red letters",
        "A newspaper headline 'BREAKING NEWS' in bold",
        "A price tag showing '$99.99' clearly visible",
        "A street name sign 'MAIN STREET' on a pole"
    ]
    
    print(f"Generating {len(prompts)} images...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt}")
        try:
            image_bytes = generate_image(prompt, api_key, endpoint)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{i}.png"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            print(f"  Saved: {filepath}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nDone! Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
