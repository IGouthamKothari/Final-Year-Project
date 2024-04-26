import os  # For handling files and directories
import pickle  # For serializing and deserializing Python objects (storing numpy features)
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting images
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # For using the VGG16 model for feature extraction
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # For loading and preprocessing images
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences to a fixed length
from tensorflow.keras.models import Model, load_model  # For creating Keras models and loading saved models
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add  # For building neural network layers
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # For loading Tokenizer from JSON
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
from io import BytesIO
from PIL import Image
import base64


# Flask app setup
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Tokenizer from JSON file
with open('models/tokenizer.json', 'r') as file:
    data = file.read()
    tokenizer = tokenizer_from_json(data)

# Load the pre-trained models
model_vgg_caption = load_model("models/vgg16model.h5")
base_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
max_length = 35

# Function to extract features from an image
def feature_extract(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = vgg_model.predict(image)
    return features

# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption for an image
def predict_caption(model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text
@app.route('/')
def index():
    return render_template('upload.html')
# Flask route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.get_json()  # Get the JSON data sent to the route
    image_data = data['image']  # Access the image data from the JSON
    image_data = base64.b64decode(image_data.split(",")[1])  # Decode the Base64 string

    # Convert binary data to an image
    image = Image.open(BytesIO(image_data))

    # Save the image to a file
    filename = secure_filename("uploaded_image.jpg")  # You might want to dynamically generate filenames
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)  # Save image to disk

    # Example processing function, like generating a caption
    try:
        # Assuming you have a function to extract features and predict a caption
        image_features = feature_extract(file_path)  # Extract features
        caption = predict_caption(model_vgg_caption, image_features, tokenizer, max_length)  # Predict caption
        return jsonify({'filename': filename, 'caption': caption}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
