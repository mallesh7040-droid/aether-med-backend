from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random

# Create a Flask web server instance
app = Flask(__name__)

# This line is the fix. It enables CORS for all origins and methods,
# allowing your frontend to connect without being blocked.
CORS(app)

# The endpoint for the CT scan analysis
@app.route('/analyze_ct_scan', methods=['POST'])
def analyze_ct_scan():
    """
    Handles the POST request for CT scan analysis.
    This is a mock endpoint. In a real-world scenario, you would
    integrate a machine learning model here to perform the analysis.
    """
    print("Received a request to analyze a CT scan.")

    # Check if a file was sent in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # For this example, we'll just return a dummy result.
    # We can simulate different outcomes based on the file name.
    is_tumor = "positive" in file.filename.lower()
    
    if is_tumor:
        classification = "Malignant" if "malignant" in file.filename.lower() else "Benign"
        response_data = {
            "isTumor": True,
            "classification": classification,
            "detections": [{
                # These highlight coordinates are for demonstration purposes.
                # A real model would provide them based on the analysis.
                "highlight": [50, 50, 150, 150],
                "confidence": round(random.uniform(0.7, 0.99), 2)
            }]
        }
    else:
        response_data = {
            "isTumor": False,
            "classification": "N/A",
            "detections": []
        }

    # Log the result to the console
    print(f"Analysis complete. Result: {response_data}")
    return jsonify(response_data)

# Heroku will set the PORT environment variable for us
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
