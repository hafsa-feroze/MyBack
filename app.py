from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model_path = os.path.join('model', 'fire_detection_model.h5')
model = tf.keras.models.load_model(model_path)

def predict_single_image(image, img_size=64):
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Resize to the model's expected size (64x64)
    img = cv2.resize(img, (img_size, img_size))
    
    # Ensure we have 3 channels
    if len(img.shape) == 2:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # If RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to create batch of size 1
    img_input = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img_input, verbose=0)
    pred_class = int(pred[0][0] > 0.5)  # 0 or 1
    confidence = float(pred[0][0]) * 100
    
    # Map to label
    label = "Fire" if pred_class == 1 else "No Fire"
    return label, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image file
        image = Image.open(file)
        
        # Make prediction
        result, confidence = predict_single_image(image)
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True) 