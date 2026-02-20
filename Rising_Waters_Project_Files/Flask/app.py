from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

model = load("floods.save")
sc = load("transform.save")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def index():
    return render_template("index.html")

@app.route("/data_predict", methods=["POST"])
def predict():
    Temp = float(request.form["Temp"])
    Humidity = float(request.form["Humidity"])
    Cloud_Cover = float(request.form["Cloud_Cover"])
    ANNUAL = float(request.form["ANNUAL"])
    Jan_Feb = float(request.form["Jan_Feb"])
    Mar_May = float(request.form["Mar_May"])
    Jun_Sep = float(request.form["Jun_Sep"])
    Oct_Dec = float(request.form["Oct_Dec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])

    data = [[Temp, Humidity, Cloud_Cover, ANNUAL,
             Jan_Feb, Mar_May, Jun_Sep, Oct_Dec,
             avgjune, sub]]

    data = sc.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("noChance.html")

if __name__ == "__main__":
    app.run(debug=True)
