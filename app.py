from flask import Flask, render_template, request, redirect, session
from pymongo import MongoClient
from datetime import datetime
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB Setup
client = MongoClient("mongodb+srv://shubhammauryagkp:shubham123@cluster0.vornv2m.mongodb.net/?retryWrites=true&w=majority&tls=true")
db = client["disease_prediction_db"]
collection = db["reports"]

# Load ML Models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
parkinson_model = pickle.load(open("models/parkinson_model.pkl", "rb"))

# Routes

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":
            session["admin"] = True
            return redirect("/admin_dashboard")
    return render_template("admin_login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect("/admin_login")
    reports = list(collection.find().sort("timestamp", -1))
    return render_template("admin_dashboard.html", reports=reports)

@app.route("/patient_login", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        name = request.form["name"]
        session["patient"] = name
        return redirect("/patient_dashboard")
    return render_template("patient_login.html")

@app.route("/patient_dashboard")
def patient_dashboard():
    name = session.get("patient")
    if not name:
        return redirect("/patient_login")
    reports = list(collection.find({"name": {"$regex": f"^{name}$", "$options": "i"}}).sort("timestamp", -1))
    return render_template("patient_dashboard.html", reports=reports)

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

if __name__ == "__main__":
    app.run(debug=True)
