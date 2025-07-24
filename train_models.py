import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

os.makedirs("models", exist_ok=True)

# ----------------- Diabetes -----------------
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler_diabetes = StandardScaler()
X_scaled = scaler_diabetes.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model_diabetes = LogisticRegression(max_iter=1000)
model_diabetes.fit(X_train, y_train)

pickle.dump(model_diabetes, open("models/diabetes_model.pkl", "wb"))
pickle.dump(scaler_diabetes, open("models/diabetes_scaler.pkl", "wb"))

# ----------------- Heart -----------------
df_heart = pd.read_csv("heart.csv")
X = df_heart.drop("target", axis=1)
y = df_heart["target"]

scaler_heart = StandardScaler()
X_scaled = scaler_heart.fit_transform(X)

model_heart = LogisticRegression(max_iter=1000)
model_heart.fit(X_scaled, y)

pickle.dump(model_heart, open("models/heart_model.pkl", "wb"))
pickle.dump(scaler_heart, open("models/heart_scaler.pkl", "wb"))

# ----------------- Parkinson -----------------
df_parkinson = pd.read_csv("parkinsons.csv")
X = df_parkinson.drop(["status", "name"], axis=1)
y = df_parkinson["status"]

scaler_parkinson = StandardScaler()
X_scaled = scaler_parkinson.fit_transform(X)

model_parkinson = LogisticRegression(max_iter=1000)
model_parkinson.fit(X_scaled, y)

pickle.dump(model_parkinson, open("models/parkinson_model.pkl", "wb"))
pickle.dump(scaler_parkinson, open("models/parkinson_scaler.pkl", "wb"))

print("✅ All models and scalers saved successfully.")
