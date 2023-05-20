from flask import Flask, request
import json
import tensorflow as tf
import numpy as np


app = Flask('Image classifier')
resnet = tf.keras.applications.ResNet101()
with open('data/imgnet_cats_ru.txt', encoding='utf-8') as f:
    cats = f.readlines()

categories_ru = [s.rstrip() for s in cats]


model = tf.keras.models.load_model('models\\dummy_model')

@app.route('/')
def home():
    return 'Home page'


@app.route('/classify', methods=['POST', 'GET'])
def classify_imagenet():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (224, 224))
    out = resnet(img_t)
    idxs = tf.argsort(out, direction='DESCENDING')[0][:3].numpy()
    out = ', '.join([categories_ru[int(i)] for i in idxs])
    return out


@app.route('/classify/binary', methods=['POST'])
def classify_binary():
    data = request.data
    img = tf.io.decode_jpeg(data)
    img_t = tf.expand_dims(img, axis=0)
    img_t = tf.image.resize(img_t, (180, 180))
    predictions = model.predict(img_t)
    dog_probability = float(predictions[0])
    print(dog_probability)
    idx = dog_probability > 0.5
    return ('Cat','Dog')[idx]

if __name__ == '__main__':
    app.run(port=1791)
    input()