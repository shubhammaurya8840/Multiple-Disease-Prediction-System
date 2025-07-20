from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from datetime import datetime
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB connection
client = MongoClient("mongodb+srv://shubhammauryagkp:shubham123@cluster0.vornv2m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["disease_prediction_db"]
collection = db["reports"]

# Load models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
parkinson_model = pickle.load(open("models/parkinson_model.pkl", "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Admin Login
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect("/admin-dashboard")
    return render_template("admin_login.html")

# Admin Dashboard
@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect("/admin")
    reports = list(collection.find().sort("timestamp", -1))
    return render_template("admin_dashboard.html", reports=reports)

# Patient Login
@app.route("/patient", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        session["patient"] = request.form["name"]
        return redirect("/patient-dashboard")
    return render_template("patient_login.html")

# Patient Dashboard
@app.route("/patient-dashboard")
def patient_dashboard():
    name = session.get("patient")
    if not name:
        return redirect("/patient")
    reports = list(collection.find({"name": {"$regex": f"^{name}$", "$options": "i"}}).sort("timestamp", -1))
    return render_template("patient_dashboard.html", reports=reports)

# Disease Prediction Routes
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        name = form_data.pop("name")
        data = [float(value) for value in form_data.values()]
        pred = diabetes_model.predict([data])[0]
        result = "Diabetes Detected" if pred == 1 else "No Diabetes"
        collection.insert_one({
            "name": name,
            "disease": "Diabetes",
            "data": form_data,
            "result": result,
            "timestamp": datetime.now()
        })
    return render_template("diabetes.html", result=result)

@app.route("/heart", methods=["GET", "POST"])
def heart():
    result = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        name = form_data.pop("name")
        data = [float(value) for value in form_data.values()]
        pred = heart_model.predict([data])[0]
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
        collection.insert_one({
            "name": name,
            "disease": "Heart",
            "data": form_data,
            "result": result,
            "timestamp": datetime.now()
        })
    return render_template("heart.html", result=result)

@app.route("/parkinson", methods=["GET", "POST"])
def parkinson():
    result = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        name = form_data.pop("name")
        data = [float(value) for value in form_data.values()]
        pred = parkinson_model.predict([data])[0]
        result = "Parkinson's Detected" if pred == 1 else "No Parkinson's"
        collection.insert_one({
            "name": name,
            "disease": "Parkinson",
            "data": form_data,
            "result": result,
            "timestamp": datetime.now()
        })
    return render_template("parkinson.html", result=result)

# ✅ Add working route aliases
@app.route("/login/admin")
def login_admin():
    return redirect("/admin")

@app.route("/login/patient")
def login_patient():
    return redirect("/patient")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
