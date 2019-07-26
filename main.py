#load os
import os

#load Flask 
import flask
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

#load model preprocessing and needed keras packa
import numpy as np
import pandas as pd

import sys

import cv2
import scipy
import skimage
from skimage.transform import resize

import keras.models
from keras.models import model_from_json
from keras.layers import Input

#initialize app
app = flask.Flask(__name__)

# Load pre-trained model into memory
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded Model from disk")

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':           
        #Delete output.png if already exists (will be overwriting this and get error if not removed)
        if os.path.exists('output.png')==True:
            os.remove('output.png')
        
        # save file to network (note: run line below if you don't want to keep images)
        filename = photos.save(request.files['photo'])
        # rename file so you don't blow up storage with files uploaded
        os.rename('./'+filename,'./'+'output.png')

        # convert to matrix that is size needed for CNN
        img_matrix = cv2.imread('output.png')
        img_matrix_downsampled = skimage.transform.resize(img_matrix, (256,256,3)) 
        img_matrix_resized = img_matrix_downsampled.reshape(1,3,256,256)
       
        # put through pre-trained CNN and send prediction to HTML to give user response
        pred_df = pd.DataFrame(loaded_model.predict(img_matrix_resized))[1]
        if pred_df.iloc[0]<0.50:
            #Note: these numbers are based on model precision in test sample
            prediction = "Not at high risk of pneumonia. Rescan if symptoms persist"
        else:
            prediction = "At high risk of pneumonia. Please provide treatment."

        return render_template('results_page.html', prediction=prediction)
    
    else:
        #load upload page
        return render_template('upload_page.html')

if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8080)

