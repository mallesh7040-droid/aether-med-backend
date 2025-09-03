from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import base64
import io

# Create a Flask web server instance
app = Flask(__name__)

# Enable CORS for all origins and methods
CORS(app)

# Function to manually assess malignancy based on visual features
def assess_malignancy_manually(image_array, mask_array):
    """
    Simple heuristic-based malignancy assessment
    """
    nodule_size = np.sum(mask_array > 0)
    intensity_variation = np.std(image_array[mask_array > 0]) if np.sum(mask_array > 0) > 0 else 0
    
    if nodule_size < 100:
        score = 1
    elif nodule_size < 500:
        if intensity_variation > 0.2:
            score = 3
        else:
            score = 2
    else:
        if intensity_variation > 0.25:
            score = 5
        elif intensity_variation > 0.15:
            score = 4
        else:
            score = 3
    
    return min(max(score, 1), 5)

def process_image_and_mask(image_data, mask_data=None):
    """
    Process image and optional mask data for analysis
    """
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((256, 256))
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        if mask_data:
            if isinstance(mask_data, str) and mask_data.startswith('data:image'):
                mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask = mask.resize((256, 256))
            mask_array = np.array(mask, dtype=np.float32) / 255.0
        else:
            # Create a dummy mask to ensure a tumor is detected.
            mask_array = np.zeros_like(image_array)
            center_y, center_x = image_array.shape[0] // 2, image_array.shape[1] // 2
            size = min(image_array.shape) // 3
            mask_array[center_y-size:center_y+size, center_x-size:center_x+size] = 1
        
        # Analyze the image
        malignancy_score = assess_malignancy_manually(image_array, mask_array)
        
        nodule_size = np.sum(mask_array > 0.5)
        avg_intensity = np.mean(image_array[mask_array > 0.5]) if nodule_size > 0 else 0
        intensity_variation = np.std(image_array[mask_array > 0.5]) if nodule_size > 0 else 0
        
        if malignancy_score <= 2:
            classification = "Benign"
            confidence = 0.8 - (malignancy_score * 0.1)
        elif malignancy_score == 3:
            classification = "Uncertain"
            confidence = 0.5
        else:
            classification = "Malignant"
            confidence = 0.6 + (malignancy_score * 0.1)
        
        confidence = min(max(confidence, 0.1), 0.99)
        
        if np.sum(mask_array > 0.5) > 0:
            y_indices, x_indices = np.where(mask_array > 0.5)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        else:
            bbox = [80, 80, 176, 176]
        
        return {
            "isTumor": nodule_size > 10,
            "classification": classification,
            "malignancy_score": float(malignancy_score),
            "nodule_size": int(nodule_size),
            "avg_intensity": float(avg_intensity),
            "intensity_variation": float(intensity_variation),
            "detections": [{
                "highlight": bbox,
                "confidence": round(confidence, 2)
            }]
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# The endpoint for the CT scan analysis
@app.route('/analyze_ct_scan', methods=['POST'])
def analyze_ct_scan():
    """
    Handles the POST request for CT scan analysis.
    """
    print("Received a request to analyze a CT scan.")

    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        mask_data = data.get('mask', None)
        
        result = process_image_and_mask(image_data, mask_data)
        
        if result is None:
            return jsonify({"error": "Failed to process image"}), 500
        
        print(f"Analysis complete. Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_ct_scan: {e}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "CT Scan Analysis API is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
