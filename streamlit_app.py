import requests
import streamlit as st

# URL de Firebase para leer los datos
firebase_url = "https://ecg-database-6a37a-default-rtdb.firebaseio.com/ecg_data.json"

def load_data():
    response = requests.get(firebase_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return [entry['value'] for entry in data.values()]
    return []

st.title("Visualización de datos ECG en tiempo real")
ecg_data = load_data()
st.line_chart(ecg_data)

if st.button("Actualizar datos"):
    ecg_data = load_data()
    st.line_chart(ecg_data)