import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Prédiction du Churn")

features = [st.number_input(f"Feature {i+1}") for i in range(14)]

if st.button("Prédire"):
    response = requests.post(API_URL, json={"features": features})
    if response.status_code == 200:
        st.success(f"Résultat : {response.json()['prediction']}")
    else:
        st.error(" Erreur API")
