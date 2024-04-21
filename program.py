from flask import Flask ,render_template ,request
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/carpriceprediction")
def carpriceprediction():
    
    dataset = pd.read_csv("clean_data.csv")
    companies = sorted(dataset["company"].unique())
    names = sorted(dataset["name"].unique())
    return render_template("carpriceprediction.html", companies = companies, names = names)

@app.route("/carpriceresult")
def carpriceresult():
    company = request.args.get("company")
    name = request.args.get("name")
    year = request.args.get("year")
    kms_driven = request.args.get("kms_driven")
    fuel_type = request.args.get("fuel_type")

    pipe = pkl.load(open('LinearRegresionModel.pkl', 'rb'))

    columns = ["name", "company", "year", "kms_driven", "fuel_type"]
    data = np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5)
    myinput = pd.DataFrame(columns = columns, data = data)
    result = pipe.predict(myinput)
    return render_template("carpriceresult.html", company = company, name = name, year = year, kms_driven = kms_driven, fuel_type = fuel_type, result = result)
