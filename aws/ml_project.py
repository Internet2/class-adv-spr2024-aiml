# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import  decode_predictions
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.applications.vgg16 import decode_predictions
import platform


###
## added processing status
import paho.mqtt.client as mqtt
from paho.mqtt import client as mqtt_client

# MQTT Broker information
broker_address = "broker.emqx.io"
broker_port = 1883  # Default MQTT port
topic = "ml-status"

client = mqtt.Client(mqtt_client.CallbackAPIVersion.VERSION1, "Python_Client")
client.connect(broker_address, broker_port)


# Initialize Flask application
app = Flask(__name__)
CORS(app) 
# Load your trained machine learning model


model = ResNet50(weights = 'imagenet')

from werkzeug.utils import secure_filename
import os 


# Define image preprocessing function
def preprocess_image(image_path):

    filename = secure_filename(image_path.filename)
    upload_directory = ''
    file_path = os.path.join(upload_directory, filename)
    image_path.save(file_path)
    
    # img = keras.utils.load_img(image_path, target_size=(224, 224))
    img = keras.utils.load_img(file_path, target_size=(224, 224))
    if os.path.exists(file_path):
        os.remove(file_path)

    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assuming input size of your model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def health_check():
    return 'Server is Healthy'


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
    # decoded_predictions = decode_predictions(prediction)
    print(label)
    print("the correct number is ")
    # displaying the hostname of the device 
    # using the platform.node()python
    print("The hostname that processed this image was ",platform.node())


    # print (label[0][0])
    print (label[0][0][1])
    print (label[0][0][2])
    
    # client.publish(topic, 'label : ' + str(label[0][0][1]) + ' confidence : ' + str(label[0][0][2]) + ' processed by : ' + platform.node())

    # return jsonify({'label': str(label[0][0][1]), 'confidence': str(label[0][0][2]) , 'processed by' : platform.node()})
    # return "returned"

    response_data = {
        'label': label[0][0][1],  # Most likely label
        'confidence': float(label[0][0][2]),  # Confidence of the prediction
        'processed_by': platform.node()
    }
    print(response_data)
    client.publish(topic, f"Processed by: {platform.node()}, Label: {response_data['label']}, Confidence: {response_data['confidence']}")

    return jsonify(response_data)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')
