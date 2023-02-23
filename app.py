import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

# Set the path to your trained model file
# model_path = "C:\\uottawa\\other_stuff\\job_apps\\fulhaus\\resnet.h5"
model_path = "resnet.h5"   
# Load the trained model
model = load_model(model_path, compile=False)

# Compile the model with the Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
# Define the input size of the images for the model
input_size = (224, 224)

# Define the Flask app
app = Flask(__name__)

# Define a function to preprocess the input image
def preprocess_image(image):
    # Convert the image to RGB color space
    image = image.convert('RGB')
    # Resize the image to the input size of the model
    image = image.resize(input_size)
    # Convert the image to a NumPy array
    image = img_to_array(image)
    # Preprocess the image for the model
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    # Return the preprocessed image
    return image

# Define a route to predict the class of an input image
@app.route('/predict', methods=['GET'])
def predict():
    print('code runnning')
    # Get the input image from the request
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))
    # Preprocess the input image
    image = preprocess_image(image)
    # Predict the class of the input image
    predictions = model.predict(image)
    # Return the predicted class as JSON
    class_names = ['bed', 'chair', 'sofa']  # Replace with your own class names
    response = {'class': class_names[np.argmax(predictions)]}
    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
