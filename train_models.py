import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# ------------- Diabetes Model -------------
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(model, open("models/diabetes_model.pkl", "wb"))

# ------------- Heart Disease Model -------------
df_heart = pd.read_csv("heart.csv")
X = df_heart.drop("target", axis=1)
y = df_heart["target"]

X = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("models/heart_model.pkl", "wb"))

# ------------- Parkinson's Disease Model -------------
df_parkinson = pd.read_csv("parkinsons.csv")

# Drop name and use features only
X = df_parkinson.drop(["status", "name"], axis=1)
y = df_parkinson["status"]

X = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("models/parkinson_model.pkl", "wb"))

print("All Models Trained and Saved Successfully.")
