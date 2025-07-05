# iris_classifier.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
species_map = dict(zip(range(3), iris.target_names))
y = y.map(species_map)  # Convert 0/1/2 -> setosa/versicolor/virginica

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
print("✅ Model and scaler saved successfully.")



import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

