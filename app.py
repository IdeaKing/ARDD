####################################################
#                                                  #
# ARDD website hoster.                             #
# Created by Thomas Chia and Cindy Wu              #
# Medical Research by Sreya Devarakonda            #
# Created for the 2021 Congressional App Challenge #
# Winning "webapp" of Virginia's 10th District     #
#                                                  #
####################################################

import sys, os, glob, re, cv2, random, base64, openpyxl
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from utils import base64_to_pil, overlay_image, np_to_base64, diagnose, model_predict, convert_data

# Instantiate the Flask Webapp
app = Flask(__name__)

# Initialize random seed 
random.seed(101)

# Fix tensorflow inference error ONLY REQUIRED FOR CUDA ENABLED DEVICES
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo_model = load_model('./configurations/yolo_weights/ARDD_yolo_model_v3.h5')
disc_model = load_model('./configurations/glaucoma_weights/ARDD_disc_model.h5')
seg_model = load_model('./configurations/glaucoma_weights/ARDD_mnet_model.h5')

print("APP is deployed at local_host_IP:5000. To enter webapp please enter your host_ip:5000.")

@app.route('/', methods=['GET'])
def index():
    # Main webpage
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from the post request
        img = base64_to_pil(request.json)
        file_name = str(random.randint(1111111, 9999999))  + '.jpg'

        # Save file and get file names
        file_path = os.path.join('./uploads', 'original', file_name)
        final_path = os.path.join('./uploads', 'output', file_name)
        img.save(file_path)

        # Predict and return image
        final_output, diagnosis_sheet = model_predict(file_path, file_name, yolo_model = yolo_model, disc_model = disc_model, seg_model = seg_model)
        # Save the output image to folder for outputs
        final_output.save(final_path) 
        # Read spreadsheet as a pandas dataframe
        data_pd = convert_data(diagnosis_sheet, file_name)

        # Change image to base64 encoding
        annotated_img = u"data:image/png;base64," + base64.b64encode(open(final_path, "rb").read()).decode("ascii")

        return jsonify(result = annotated_img, table = data_pd)

    return None

if __name__ == '__main__':
    # Run the web-app, access from your browser.
    """ 
    # Use this to serve on an actual server and not on a demo.
    http_server = WSGIServer(('0.0.0.0', 4340), app)
    http_server.serve_forever()
    """
    # Use this to serve as a demo.
    app.run(debug = False)
