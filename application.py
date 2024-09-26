from flask import Flask,request,render_template,jsonify
import pandas
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


application = Flask(__name__)
app = application
linear_model = pickle.load(open("models/lr.pkl","rb"))
Standard_Scaler = pickle.load(open("models/ss.pkl","rb"))



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predictiondata",methods = ["GET","POST"])
def predict():
    if request.method =="POST":
        Temperature = float(request.form.get("Temperature"))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        dc = float(request.form.get('DC'))
        isi = float(request.form.get('ISI'))
        bui = float(request.form.get('BUI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_sc= Standard_Scaler.transform([[Temperature, rh, ws, rain, ffmc, dmc,dc, isi,bui, classes, region]])
        result = linear_model.predict(new_sc)
        return render_template("home.html",results=result[0])
        
    else:
        return render_template("home.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)