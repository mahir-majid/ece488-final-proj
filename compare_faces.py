"""
Script to compare face similarity between two images.
Uses environment variables for image paths and outputs results to comparisons.csv
"""

import os
import csv
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def detect_face(image_path):
    """Detect and extract face region from image."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier (built into OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise ValueError(f"No face detected in image: {image_path}")
    if len(faces) > 1:
        print(f"⚠️  Warning: Multiple faces detected in {image_path}, using first face")
    
    # Extract first face region
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size for comparison
    face_roi = cv2.resize(face_roi, (128, 128))
    
    return face_roi


def compute_face_similarity(img1_path, img2_path):
    """Compute face similarity using histogram comparison."""
    try:
        # Detect and extract faces
        face1 = detect_face(img1_path)
        face2 = detect_face(img2_path)
        
        # Compute histograms
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms using correlation (returns 0-1, higher = more similar)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Also compute structural similarity for better accuracy
        # Resize both faces to same size if needed
        face1_resized = cv2.resize(face1, (128, 128))
        face2_resized = cv2.resize(face2, (128, 128))
        
        # Compute SSIM-like metric using template matching
        result = cv2.matchTemplate(face1_resized, face2_resized, cv2.TM_CCOEFF_NORMED)
        template_similarity = result[0][0]
        
        # Average both methods for more robust similarity score
        final_similarity = (similarity + template_similarity) / 2
        
        return final_similarity
    except Exception as e:
        raise ValueError(f"Error comparing faces: {e}")


def main():
    """Compare faces and save results to CSV."""
    # Get image paths from environment variables
    default_image_path = os.getenv('DEFAULT_PERSON_FACE_IMAGE')
    infinite_image_path = os.getenv('INFINITE_PERSON_FACE_IMAGE')
    
    if not default_image_path:
        raise ValueError("DEFAULT_PERSON_FACE_IMAGE environment variable must be set")
    if not infinite_image_path:
        raise ValueError("INFINITE_PERSON_FACE_IMAGE environment variable must be set")
    
    # Check if images exist
    if not os.path.exists(default_image_path):
        raise FileNotFoundError(f"Image not found: {default_image_path}")
    if not os.path.exists(infinite_image_path):
        raise FileNotFoundError(f"Image not found: {infinite_image_path}")
    
    print(f"📸 Comparing faces...")
    print(f"   Default: {default_image_path}")
    print(f"   Infinite: {infinite_image_path}")
    
    similarity = compute_face_similarity(default_image_path, infinite_image_path)
    
    # Print result
    print(f"\n✅ Face Similarity: {similarity:.4f} ({similarity*100:.2f}%)")
    
    # Write to CSV
    csv_path = Path('comparisons.csv')
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['DEFAULT_PERSON_FACE_IMAGE', 'INFINITE_PERSON_FACE_IMAGE', 'Face_ID_Similarity'])
        
        # Write data row
        writer.writerow([default_image_path, infinite_image_path, f"{similarity:.6f}"])
    
    print(f"💾 Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

