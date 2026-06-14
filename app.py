import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.write("Upload network data CSV to detect attacks")

model = joblib.load("models/nids_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)

    st.write("### Input Data")
    st.write(data.head())

    # ----------------------------
    # KEEP ONLY FIRST 41 FEATURES
    # ----------------------------
    X = data.iloc[:, :41]

    # Encode categorical columns (same as training)
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    X[1] = le1.fit_transform(X[1].astype(str))
    X[2] = le2.fit_transform(X[2].astype(str))
    X[3] = le3.fit_transform(X[3].astype(str))

    # Prediction
    predictions = model.predict(X)

    data["Prediction"] = predictions

    st.write("### Results")
    st.write(data)

    st.success("Prediction completed successfully!")