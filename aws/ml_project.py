# Import necessary libraries
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from keras.applications.vgg16 import decode_predictions


# Initialize Flask application
app = Flask(__name__)

# Load your trained machine learning model


model = ResNet50(weights = 'imagenet')

# model = tf.keras.models.load_model('model.h5')

# Define image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assuming input size of your model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Define route for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Check if request contains file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if file is of allowed type
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Preprocess the image
    img = preprocess_image(file)
    
    # Make prediction
    prediction = model.predict(img)

    
    # Assuming your model outputs probabilities for each class
    # Modify this part based on your model's output format
    #       class_index = np.argmax(prediction)
    #       class_label = "Your class labels here"


    label = decode_predictions(prediction)
    print(label)
    print("the correct number is ")
    # print (label[0][0])
    print (label[0][0][1])
    print (label[0][0][2])
    return jsonify({'label': str(label[0][0][1]), 'confidence': str(label[0][0][2]) })
    # return "returned"

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
