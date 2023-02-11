import numpy as np
import os
import joblib
from keras.preprocessing import image
import tensorflow as tf
global graph
graph=tf.compat.v1.get_default_graph()
from flask import Flask, request, render_template

app = Flask(__name__)
joblib_file = "modelf.pkl"
model = joblib.load(joblib_file)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    x_test=[[float(x) for x in request.form.values()]]
    prediction=model.predict(x_test)
    print(prediction)
    output=prediction
    if (output==1):
        prediction_text1="There are more chances of patient dying within 1 year"
    elif (output==2):
        prediction_text1="Patient will Survive"
    return render_template('index.html', prediction_text=prediction_text1)


if __name__=="__main__":
    app.run(debug=True)


