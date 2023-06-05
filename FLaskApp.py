from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

#initialize the Flask App
app = Flask(__name__)

#Read ML model
pickle_in = open(r"C:\Users\ssant\Downloads\RFClassifier.pkl",'rb')
RFClassifier = pickle.load(pickle_in)

#Test home page
@app.route('/')
def welcomePage():
    return "Welcome to testing module"

#stores the parameters
@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = RFClassifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Predicted value is : "+str(prediction)
#http://127.0.0.1:5000/predict?variance=2&skewness=-3&curtosis=-2&entropy=1

#stores the parameters
@app.route('/predict_file',methods=["POST"])
def predict_note_authentication_file():
    df_file = pd.read_csv(request.files.get("file"))
    prediction = RFClassifier.predict(df_file)
    return "The Predicted values for the given CSV file is : "+str(list(prediction))
#http://127.0.0.1:5000/predict_file?variance=2&skewness=-3&curtosis=-2&entropy=1


if __name__=='__main__':
    app.run()