import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

print("=" * 50)
print("Network Intrusion Detection System")
print("=" * 50)

# Load dataset
data = pd.read_csv("dataset/KDDTrain+.txt", sep="\t", header=None)

# Split features and labels
X = data.iloc[:, :-2]
y = data.iloc[:, -2]

# Encode categorical features
for col in [1, 2, 3]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("\nData preprocessing completed!")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix - NIDS")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
joblib.dump(model, "models/nids_model.pkl")

print("\nModel saved successfully in models/nids_model.pkl")

import numpy as np

print("\n--- NIDS Prediction System Ready ---")

def predict_sample(sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return prediction[0]

# Test prediction using one sample from test data
sample = X_test.iloc[0].values
result = predict_sample(sample)

print("\nSample Prediction:", result)