import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- 1. SETUP ---
app = Flask(__name__)
CORS(app)  # allow React frontend to access this API

MODEL_SAVE_PATH = 'covid_classifier_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'
TARGET_SIZE = (224, 224)

# --- 2. LOAD MODEL & CLASS INDICES ---
try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"✅ Model loaded successfully from {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model from {MODEL_SAVE_PATH}")
    print(e)
    exit(1)

try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    print(f"✅ Class indices loaded: {class_names}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load class indices from {CLASS_INDICES_PATH}")
    print(e)
    exit(1)

# --- 3. PREDICTION ROUTE ---
@app.route('/analyze', methods=['POST'])
def analyze_xray():
    if 'xray_image' not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'xray_image'."}), 400

    file = request.files['xray_image']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Convert uploaded file to PIL Image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).resize(TARGET_SIZE)
        img = img.convert('RGB')

        # Convert to numpy array for model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        # Predict
        prediction_probs = model.predict(img_array)[0]
        predicted_index = int(np.argmax(prediction_probs))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction_probs))

        # --- Confidence threshold fix ---
        # If the model says "COVID" but is not confident, assume "Normal"
        if predicted_label == "COVID" and confidence < 0.85:
            predicted_label = "Normal"

        # Prepare JSON response with all class probabilities
        response = {
            "prediction": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(prediction_probs[i]) for i in range(len(class_names))
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed due to server issue: {e}"}), 500


# --- 4. RUN SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
