import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input as preprocess_densenet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import threading

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Tokenizers from JSON files
with open('models/tokenizer.json', 'r') as file:
    vgg_tokenizer_data = file.read()
vgg_tokenizer = tokenizer_from_json(vgg_tokenizer_data)

with open('models/tokenizer_densenet.json', 'r') as file:
    densenet_tokenizer_data = file.read()
densenet_tokenizer = tokenizer_from_json(densenet_tokenizer_data)

# Load pre-trained models for feature extraction
base_model_vgg = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=base_model_vgg.inputs, outputs=base_model_vgg.layers[-2].output)

base_model_densenet = DenseNet201(weights='imagenet')
densenet_feature_model = Model(inputs=base_model_densenet.inputs, outputs=base_model_densenet.layers[-2].output)

# Load models for captioning
model_vgg_caption = load_model("models/vgg16model.h5")
model_densenet_caption = load_model("models/densenetmodel.h5")

max_length = 35
max_length_vgg = 35
max_length_densenet = 34

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def feature_extract(model, image_path, preprocess_input):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return model.predict(image)

def predict_caption(caption_model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text
@app.route('/')
def index():
    return render_template('upload.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(",")[1])
    image = Image.open(BytesIO(image_data))
    filename = secure_filename("uploaded_image.jpg")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)

    captions = {}

    def get_vgg_caption():
        features = feature_extract(vgg_feature_model, file_path, preprocess_vgg16)
        captions['vgg_caption'] = predict_caption(model_vgg_caption, features, vgg_tokenizer, max_length_vgg)

    def get_densenet_caption():
        features = feature_extract(densenet_feature_model, file_path, preprocess_densenet)
        captions['densenet_caption'] = predict_caption(model_densenet_caption, features, densenet_tokenizer,
                                                       max_length_densenet)

    # Create threads for parallel processing
    thread_vgg = threading.Thread(target=get_vgg_caption)
    thread_densenet = threading.Thread(target=get_densenet_caption)

    # Start threads
    thread_vgg.start()
    thread_densenet.start()

    # Join threads to the main thread
    thread_vgg.join()
    thread_densenet.join()

    return jsonify(captions), 200

if __name__ == '__main__':
    app.run(debug=True)
