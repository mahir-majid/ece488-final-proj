"""
Simple test script for InfiniteYou RunPod API
Uses environment variables for configuration
"""

import os
import time
import base64
import requests
import random
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{image_data}"

def base64_to_image(base64_string, output_path):
    """Convert base64 string to image and save"""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")

def test_infiniteyou_api():
    """
    Test the InfiniteYou RunPod API
    
    Environment Variables:
    - RUNPOD_INFU_URL: RunPod endpoint URL
    - RUNPOD_API_KEY: RunPod API key
    - INFINITE_INPUT_FACE_IMAGE_PATH: Path to identity/face image (required)
    - INFINITE_OUTPUT_IMAGE_PATH: Full path to save output image (required)
    - INFINITE_PERSON_PROMPT: Prompt for image generation (required)
    """
    
    # Get environment variables
    runpod_url = os.getenv('RUNPOD_INFU_URL')
    runpod_api_key = os.getenv('RUNPOD_API_KEY')
    
    # Extract endpoint ID for status polling
    if runpod_url and '/v2/' in runpod_url:
        # Extract endpoint ID from URL like https://api.runpod.ai/v2/k0nmfiw7iqu5t0/run
        endpoint_id = runpod_url.split('/v2/')[1].split('/')[0]
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status"
    else:
        status_url = runpod_url
    
    if not runpod_url:
        print("❌ Error: RUNPOD_URL environment variable not set")
        return
    
    if not runpod_api_key:
        print("❌ Error: RUNPOD_API_KEY environment variable not set")
        return
    
    # Get image paths from environment variables
    input_image_path = os.getenv('INFINITE_INPUT_FACE_IMAGE_PATH')
    if not input_image_path:
        raise ValueError("INFINITE_INPUT_FACE_IMAGE_PATH must be set")
    print(f"input_image_path: {input_image_path}")
    
    # Get output image path from environment variable
    output_image_path = os.getenv('INFINITE_OUTPUT_IMAGE_PATH')
    if not output_image_path:
        raise ValueError("INFINITE_OUTPUT_IMAGE_PATH must be set")
    
    # Get prompt from environment variable
    prompt = os.getenv('INFINITE_PERSON_PROMPT')
    if not prompt:
        raise ValueError("INFINITE_PERSON_PROMPT must be set")
    
    # Ensure output directory exists
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use fixed dimensions
    width = 1024
    height = 1024
    
    if not os.path.exists(input_image_path):
        print(f"❌ Error: Input image not found at {input_image_path}")
        print("Please set INFINITE_INPUT_FACE_IMAGE_PATH in your .env file")
        return
    
    # Convert identity image to base64
    print(f"🔄 Converting identity image to base64: {input_image_path}")
    id_image_base64 = image_to_base64(input_image_path)
    
    # Generate a random seed for reproducible but varied results
    random_seed = random.randint(1, 999999)
    
    # Prepare request payload in RunPod format
    payload = {
        "id": "test-job",
        "input": {
            "id_image": id_image_base64,
            "prompt": prompt,
            "infu_source_img_token": "man",
            "model_version": "aes_stage2",
            "enable_realism": True,
            "enable_anti_blur": True,
            "enable_face_realism": True,
            "enable_realism_one": False,
            "enable_realism_two": False,
            "realism_weight": 1.0,
            "anti_blur_weight": 1.0,
            "face_realism_weight": 1.0,
            "realism_one_weight": 1.0,
            "realism_two_weight": 1.0,
            "flux_model": "flux-schnell",
            "seed": random_seed,
            "guidance_scale": 5,  
            "num_steps": 5,        # Increased for better quality
            "width": width,         # Dynamic width based on filename ending
            "height": height,       # Dynamic height based on filename ending
        }
    }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {runpod_api_key}"
    }
    
    print(f"🚀 Sending request to: {runpod_url}")
    print(f"📝 Prompt: {payload['input']['prompt']}")
    
    # Track request time
    start_time = time.time()
    
    try:
        # Send request
        response = requests.post(
            runpod_url,
            json=payload,
            headers=headers,
            timeout=150 
        )
        
        request_time = time.time() - start_time
        
        print(f"⏱️  Request completed in: {request_time:.2f} seconds")
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            # Check if job is queued or processing
            if result.get("status") in ["IN_QUEUE", "IN_PROGRESS"]:
                job_id = result.get("id")
                print(f"⏳ Job {job_id} is {result.get('status').lower()}. Polling for completion...")
                
                # Track generation time only when IN_PROGRESS
                generation_start_time = None
                in_progress_detected = False
                
                # Poll for job completion
                max_polls = 2000  # ~16.7 minutes with 5-second intervals (matches initial timeout)
                poll_count = 0
                
                while poll_count < max_polls:
                    time.sleep(0.5)  # Wait 0.5 seconds between polls
                    poll_count += 1
                    
                    # Check job status
                    status_response = requests.get(
                        f"{status_url}/{job_id}",
                        headers=headers,
                        timeout=30
                    )
                    
                    if status_response.status_code == 200:
                        status_result = status_response.json()
                        current_status = status_result.get("status")
                        
                        # Start tracking generation time when IN_PROGRESS is first detected
                        if current_status == "IN_PROGRESS" and not in_progress_detected:
                            generation_start_time = time.time()
                            in_progress_detected = True
                            print(f"🔄 Generation started - tracking time...")
                        
                        if current_status == "COMPLETED":
                            if generation_start_time:
                                generation_time = time.time() - generation_start_time
                                print(f"✅ Job completed - Generation time: {generation_time:.1f} seconds")
                            else:
                                elapsed = poll_count * 0.5
                                print(f"✅ Job completed after {elapsed:.1f} seconds including queue time")
                            
                            # Get the output
                            if "output" in status_result:
                                output = status_result["output"]
                                
                                if output.get("success"):
                                    print("✅ API call successful!")
                                    
                                    # Save the generated image
                                    if "image" in output:
                                        base64_to_image(output["image"], output_image_path)
                                        
                                        # Print metadata
                                        metadata = output.get("metadata", {})
                                        print(f"📊 Metadata:")
                                        print(f"   - Seed: {metadata.get('seed', 'N/A')}")
                                        print(f"   - Model: {metadata.get('model_version', 'N/A')}")
                                        print(f"   - Prompt: {metadata.get('prompt', 'N/A')}")
                                        print(f"   - Dimensions: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
                                    else:
                                        print("❌ No image in response")
                                        
                                else:
                                    print(f"❌ API returned error: {output.get('error', 'Unknown error')}")
                            else:
                                print(f"❌ No output in completed job: {status_result}")
                            break
                            
                        elif current_status == "FAILED":
                            error_msg = status_result.get('error', 'Unknown error')
                            if generation_start_time:
                                generation_time = time.time() - generation_start_time
                                print(f"❌ Job failed after {generation_time:.1f}s of generation: {error_msg}")
                            else:
                                print(f"❌ Job failed: {error_msg}")
                            
                            # Check for specific error types
                            if "No space left on device" in error_msg:
                                print("💡 Solution: Increase RunPod storage to 300GB+")
                            elif "Model not found" in error_msg:
                                print("💡 Solution: Ensure models are downloaded during startup")
                            elif "403" in error_msg or "access" in error_msg.lower():
                                print("💡 Solution: Request access to FLUX.1-dev model on HuggingFace")
                            
                            break
                            
                        else:
                            if current_status == "IN_PROGRESS" and generation_start_time:
                                generation_time = time.time() - generation_start_time
                                print(f"⏳ Job status: {current_status} (generation time: {generation_time:.1f}s)")
                            else:
                                elapsed = poll_count * 0.5
                                print(f"⏳ Job status: {current_status} (elapsed: {elapsed:.1f}s)")
                    else:
                        print(f"❌ Failed to check job status: {status_response.status_code}")
                        print(f"Response: {status_response.text}")
                        break
                else:
                    if generation_start_time:
                        generation_time = time.time() - generation_start_time
                        print(f"❌ Job timed out after {generation_time:.1f}s of generation")
                    else:
                        print("❌ Job timed out after ~16.7 minutes")
                    print("💡 Possible issues:")
                    print("   - Models still downloading (check RunPod logs)")
                    print("   - Insufficient storage (increase to 300GB+)")
                    print("   - Missing model access (request FLUX.1-dev access)")
                    print("   - Complex generation taking longer than expected")
                    
            # Direct response (if job completed immediately)
            elif "output" in result:
                output = result["output"]
                
                if output.get("success"):
                    print("✅ API call successful!")
                    
                    # Save the generated image
                    if "image" in output:
                        base64_to_image(output["image"], output_image_path)
                        
                        # Print metadata
                        metadata = output.get("metadata", {})
                        print(f"📊 Metadata:")
                        print(f"   - Seed: {metadata.get('seed', 'N/A')}")
                        print(f"   - Model: {metadata.get('model_version', 'N/A')}")
                        print(f"   - Prompt: {metadata.get('prompt', 'N/A')}")
                        print(f"   - Dimensions: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
                    else:
                        print("❌ No image in response")
                        
                else:
                    print(f"❌ API returned error: {output.get('error', 'Unknown error')}")
            else:
                print(f"❌ Unexpected response format: {result}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (12.5 minutes)")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🧪 InfiniteYou RunPod API Test")
    print("=" * 40)
    
    test_infiniteyou_api()
    
    print("\n" + "=" * 40)
    print("🏁 Test completed!")
    print(f"📊 Test Summary:")
    print(f"   - Identity Image: {os.getenv('INFINITE_INPUT_FACE_IMAGE_PATH', 'Not set')}")
    output_path = os.getenv('INFINITE_OUTPUT_IMAGE_PATH', 'Not set')
    print(f"   - Output Image: {output_path}")
