import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image

import argparse
import json

image_size = 224

def parse_args():
    parser = argparse.ArgumentParser(description='Add image, model, top k number, labels json file')

    parser.add_argument('image', type=str)
    parser.add_argument('model', type=str)
    # require top_k to be 1 <= k <= len(...something ha)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--category_names', type=str, default=None)

    return parser.parse_args()

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(img_path, model, top_k):
    image = Image.open(img_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image,axis=0)
    pred =  model.predict(image)
    prediction, label = tf.math.top_k(pred, k=top_k, sorted=True, name=None)
    prob = prediction[0].numpy().tolist()
    label = label[0].numpy().tolist()

    return prob,label

def labels_to_name(labels, category_names):
    if category_names:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        names = []
        for label in labels:
            names.append(class_names[str(label)]) 
        labels = names
    return labels

args = parse_args()
#print(args)

model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer':hub.KerasLayer})
prob, labels = predict(args.image, model, args.top_k)

labels = labels_to_name(labels, args.category_names)

print(prob)
print(labels)