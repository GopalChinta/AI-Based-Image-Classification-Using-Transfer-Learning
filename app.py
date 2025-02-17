from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('path_to_trained_model.h5')

# Function to load and preprocess the image for prediction
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))  # Adjust size according to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Route for home page
@app.route('/')
def home():
    return "Welcome to the Real-Time Image Classification API!"

# Route for predicting image class
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess image
        img = preprocess_image(file)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Return the prediction result as JSON
        return jsonify({'predicted_class': int(predicted_class), 'confidence': float(predictions[0][predicted_class])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
