import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import io
import pydicom
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a Flask web server instance
app = Flask(__name__)

# Enable CORS for all origins and methods
CORS(app)

# Function to manually assess malignancy based on visual features
def assess_malignancy_manually(image_array, mask_array):
    """
    Simple heuristic-based malignancy assessment
    This is a placeholder - in real scenario, use actual model predictions
    """
    # Basic features that might indicate malignancy
    nodule_size = np.sum(mask_array > 0)  # Area of the mask
    intensity_variation = np.std(image_array[mask_array > 0]) if np.sum(mask_array > 0) > 0 else 0
    
    # Simple scoring (1-5 scale)
    if nodule_size < 100:
        score = 1  # Very small, likely benign
    elif nodule_size < 500:
        if intensity_variation > 0.2:
            score = 3  # Medium size with variation
        else:
            score = 2
    else:
        if intensity_variation > 0.25:
            score = 5  # Large with high variation
        elif intensity_variation > 0.15:
            score = 4
        else:
            score = 3
    
    return min(max(score, 1), 5)  # Ensure score is between 1-5

def process_image_and_mask(image_data, mask_data=None):
    """
    Process image and optional mask data for analysis
    """
    try:
        # Decode image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 encoded image
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        image = image.resize((256, 256))
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Process mask if provided
        if mask_data:
            if isinstance(mask_data, str) and mask_data.startswith('data:image'):
                mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask = mask.resize((256, 256))
            mask_array = np.array(mask, dtype=np.float32) / 255.0
        else:
            # Create a dummy mask (center region) if no mask provided
            mask_array = np.zeros_like(image_array)
            center_y, center_x = image_array.shape[0] // 2, image_array.shape[1] // 2
            size = min(image_array.shape) // 4
            mask_array[center_y-size:center_y+size, center_x-size:center_x+size] = 1
        
        # Analyze the image
        malignancy_score = assess_malignancy_manually(image_array, mask_array)
        
        # Calculate additional metrics
        nodule_size = np.sum(mask_array > 0.5)
        avg_intensity = np.mean(image_array[mask_array > 0.5]) if nodule_size > 0 else 0
        intensity_variation = np.std(image_array[mask_array > 0.5]) if nodule_size > 0 else 0
        
        # Generate a heatmap for visualization
        heatmap = np.zeros_like(image_array)
        if np.sum(mask_array) > 0:
            # Create a simple heatmap based on distance from center
            y, x = np.ogrid[:image_array.shape[0], :image_array.shape[1]]
            center_y, center_x = np.mean(np.where(mask_array > 0.5), axis=1) if np.sum(mask_array > 0.5) > 0 else (
                image_array.shape[0] // 2, image_array.shape[1] // 2)
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            heatmap = np.exp(-distance / 50)  # Gaussian-like heatmap
        
        # Create overlay for visualization
        overlay = np.stack([image_array] * 3, axis=-1)  # Convert to RGB
        overlay[mask_array > 0.5, 0] = 1.0  # Red channel for mask
        overlay[mask_array > 0.5, 1] = 0.2
        overlay[mask_array > 0.5, 2] = 0.2
        
        # Combine heatmap with image
        heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]
        final_image = 0.7 * overlay + 0.3 * heatmap_rgb
        final_image = np.clip(final_image, 0, 1)
        
        # Convert visualization image to base64
        vis_img = Image.fromarray((final_image * 255).astype(np.uint8))
        buffered = io.BytesIO()
        vis_img.save(buffered, format="PNG")
        vis_img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Determine classification
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
        
        # Calculate bounding box for detected nodule
        if np.sum(mask_array > 0.5) > 0:
            y_indices, x_indices = np.where(mask_array > 0.5)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        else:
            bbox = [80, 80, 176, 176]  # Default center box
        
        return {
            "isTumor": nodule_size > 10,  # Assume any significant mask area is a tumor
            "classification": classification,
            "malignancy_score": float(malignancy_score),
            "nodule_size": int(nodule_size),
            "avg_intensity": float(avg_intensity),
            "intensity_variation": float(intensity_variation),
            "visualization": f"data:image/png;base64,{vis_img_str}",
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
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        mask_data = data.get('mask', None)
        
        # Process the image
        result = process_image_and_mask(image_data, mask_data)
        
        if result is None:
            return jsonify({"error": "Failed to process image"}), 500
        
        # Log the result to the console
        print(f"Analysis complete. Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_ct_scan: {e}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "CT Scan Analysis API is running"})

# Heroku will set the PORT environment variable for us
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
