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
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
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

with open('models/tokenizer_inception.json', 'r') as file:
    inception_tokenizer = tokenizer_from_json(file.read())

with open('models/tokenizer_resnet.json', 'r') as file:
    resnet_tokenizer = tokenizer_from_json(file.read())

# Load pre-trained models for feature extraction
base_model_vgg = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=base_model_vgg.inputs, outputs=base_model_vgg.layers[-2].output)

base_model_densenet = DenseNet201(weights='imagenet')
densenet_feature_model = Model(inputs=base_model_densenet.inputs, outputs=base_model_densenet.layers[-2].output)

base_model_inception = InceptionV3(weights='imagenet')
inception_feature_model = Model(inputs=base_model_inception.inputs, outputs=base_model_inception.layers[-2].output)

base_model_resnet = ResNet50(weights='imagenet')
resnet_feature_model = Model(inputs=base_model_resnet.inputs, outputs=base_model_resnet.layers[-2].output)

# Load models for captioning
model_vgg_caption = load_model("models/vgg16model.h5")
model_densenet_caption = load_model("models/densenetmodel.h5")
model_inception_caption = load_model("models/inceptionv3_model.h5")
model_resnet_caption = load_model("models/resnet50_model.h5")

max_length = 35
max_length_vgg = 35
max_length_densenet = 34
max_length_inception = 35
max_length_resnet = 37

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def clean_caption(caption):
    # Remove 'startseq' and any unwanted text here
    return caption.replace('startseq', '').strip()

def feature_extract(model, image_path, preprocess_input):
    image = load_img(image_path, target_size=(299, 299) if model == inception_feature_model else (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return model.predict(image)

@tf.function(experimental_relax_shapes=True)
def predict(model, features, sequence):
    return model.predict([features, sequence])

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
        captions['vgg_caption'] = clean_caption(predict_caption(model_vgg_caption, features, vgg_tokenizer,
                                                                max_length_vgg))

    def get_densenet_caption():
        features = feature_extract(densenet_feature_model, file_path, preprocess_densenet)
        captions['densenet_caption'] = clean_caption(predict_caption(model_densenet_caption, features, densenet_tokenizer,
                                                       max_length_densenet))

    def get_inception_caption():
        features = feature_extract(inception_feature_model, file_path, preprocess_inception)
        captions['inception_caption'] = clean_caption(predict_caption(model_inception_caption, features, inception_tokenizer,
                                                        max_length_inception))

    def get_resnet_caption():
        features = feature_extract(resnet_feature_model, file_path, preprocess_resnet)
        captions['resnet_caption'] = clean_caption(predict_caption(model_resnet_caption, features, resnet_tokenizer,
                                                     max_length_resnet))

    # Create threads for parallel processing
    thread_vgg = threading.Thread(target=get_vgg_caption)
    thread_densenet = threading.Thread(target=get_densenet_caption)
    thread_inception = threading.Thread(target=get_inception_caption)
    thread_resnet = threading.Thread(target=get_resnet_caption)


    # Start threads
    thread_vgg.start()
    thread_densenet.start()
    thread_inception.start()
    thread_resnet.start()


    # Join threads to the main thread
    thread_vgg.join()
    thread_densenet.join()
    thread_inception.join()
    thread_resnet.join()

    return jsonify(captions), 200

if __name__ == '__main__':
    app.run(debug=True)
