from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, render_template, url_for, redirect, request
import cv2
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]= '/static'

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img





@app.route('/')
def main():
    params={
            'content': "/static/images/pic02.jpg",
            'style': "/static/images/pic03.jpg",
            'attrib': "display:none;",
            'text' :"GET STARTED",
        }
    return render_template('index.html',**params)

@app.route('/mix', methods=['GET','POST'])
def mix():
    if request.method == 'POST':
        if request.files:

            content_image = request.files["contentbtn"]
            style_image = request.files["stylebtn"]
            content_image.save('static/images/content.jpg')
            style_image.save('static/images/style.jpg')
            file_names=[content_image.filename,style_image.filename]

            content_image = load_img('static/images/content.jpg')
            style_image = load_img('static/images/style.jpg')
            model = tf.saved_model.load('1')
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            tensor_to_image(stylized_image).save('static/images/styled_image.jpg')
            params={
            'content': "/static/images/content.jpg",
            'style': "/static/images/style.jpg",
            'result': '/static/images/styled_image.jpg',
            'attrib': "display:inline-block;",
            'text' :"RESULT",
            } 
            return render_template('index.html',**params)
    return None

if __name__ == "__main__":
    app.run(debug=True)