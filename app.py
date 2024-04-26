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

# Load Tokenizer from JSON file
with open('models/tokenizer.json') as f:
    data = f.read()
    tokenizer = tokenizer_from_json(data)

model_vgg_caption = load_model("models/vgg16model.h5")
# Load VGG16 model for feature extraction
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

# Convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict caption for an image
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

# Function to generate and display the caption along with the image
def generate_image_caption(img_path):
    image = load_img(img_path)
    image_features = feature_extract(img_path)
    caption = predict_caption(model_vgg_caption, image_features, tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(caption)
    plt.imshow(image)
    plt.show()

# Example usage
generate_image_caption("60aaca3a-15a3-403f-a72d-d9f30e5f74d7.jpeg")
