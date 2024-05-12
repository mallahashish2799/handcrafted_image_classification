# Flask utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import efficientnet.keras as effnet
from PIL import Image
from tqdm import tqdm
import tensorflow
import numpy as np
import pandas as pd
from os import listdir
import seaborn as sns
import pickle
import re
import os
import cv2

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
image_path = './static/images/input.jpg'
default_image_size = tuple((128, 128))
image_size = 128
model =load_model('./model/EfficientNetB2.h5')

class_labels = pd.read_pickle('label_transform.pkl')
classes = class_labels.classes_

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'}


def allowed_file(filename):
  return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index(): 
  return render_template('index.html')

@app.route('/home')
def home(): 
  return render_template('index.html')

@app.route('/service')
def service(): 
  return render_template('service.html')


@app.route('/handcraftedclf',methods=['POST'])
def handcraftedclf():
  file = None
  file = request.files['file']
  if file and allowed_file(file.filename):

    image = Image.open(file)
    image.save(os.path.join(image_path))
    image = cv2.imread(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (image_size, image_size))
    image = np.array([image])
    prediction=model.predict(image)
    pred_= prediction[0]
    pred=[]
    for ele in pred_:
      pred.append(ele)
    maxi_ele = max(pred)
    idx = pred.index(maxi_ele)
    final_class=classes
    class_name= final_class[idx]
    class_text = "HandCrafted Image Classify As: " + class_name
    class_text = class_text.upper()

  return render_template('service.html',pred_class=class_text,image_path = image_path)



if __name__ == '__main__':
    app.run(debug=False)



