# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:42:22 2023

@author: achu6
"""

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for, redirect
from PIL import Image
from tensorflow import keras
from keras.models import load_model

#load the model
model=load_model('models/action.h5') 

app=Flask(__name__)

#index homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alphabets')
def alphabets():
    return render_template('alphabets.html')

@app.route('/common-phrases')
def commonphrases():
    return render_template('common-phrases.html')

@app.route('/interpret')
def interpret():
    return render_template('interpret.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/numbers')
def numbers():
    return render_template('numbers.html')

@app.route('/play-video')
def playvideo():
    return render_template('play-video.html')

@app.route('/rendering')
def rendering():
    return render_template('rendering.html')

@app.route('/selection')
def selection():
    return render_template('selection.html')

@app.route('/back')
def prev(prevpage):
    if (prevpage==0):
        return render_template('numbers.html')
    elif (prevpage==1):
        return render_template('alphabets.html')
    else:
        return render_template('common-phrases.html')

    
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # img = Image.open(request.files['imgfile'].stream)
        # img = img.resize((28,28))
        # im2arr = np.array(img)
        # im2arr = im2arr.reshape(1,28,28,1)
        # pred = model.predict(im2arr)
        # num = np.argmax(pred, axis=1)
        return render_template('rendering.html')
    else :
        return render_template('interpret.html')



if __name__ == '__main__':
    app.run(debug=True, host= '192.168.0.21')

