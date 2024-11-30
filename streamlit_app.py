import requests
import streamlit as st
import plotly.graph_objects as go
import time

# URL de Firebase para leer los datos
firebase_url = "https://ecg-database-6a37a-default-rtdb.firebaseio.com/ecg_data.json"

# Frecuencia de muestreo (250 Hz)
sampling_rate = 250
window_seconds = 15  # Duración de la ventana en segundos
window_size = sampling_rate * window_seconds  # Número de muestras por ventana

def load_data():
    response = requests.get(firebase_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Extraer todos los valores de los bloques
            ecg_data = []
            for batch in data.values():
                for entry in batch:  # Iterar dentro del lote
                    ecg_data.append(entry['value'])
            return ecg_data
    else:
        st.error("Error al conectar con Firebase. Código de estado: {}".format(response.status_code))
    return []

# Streamlit: Visualización de datos ECG en tiempo real
st.title("Visualización de datos ECG en tiempo real")

# Contenedor para la gráfica
placeholder = st.empty()

# Actualización automática
while True:
    ecg_data = load_data()
    if ecg_data:
        # Mostrar solo los últimos 30 segundos
        if len(ecg_data) > window_size:
            ecg_data = ecg_data[-window_size:]  # Limitar a los últimos 7,500 puntos
        
        # Crear la gráfica
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ecg_data, mode='lines', name='ECG'))
        fig.update_layout(
            title="Señal ECG (Últimos 30 segundos)",
            xaxis_title="Tiempo (muestras)",
            yaxis_title="Amplitud (milivoltios)",
            xaxis=dict(range=[0, window_size])  # Mantener el eje X con un rango fijo
        )
        placeholder.plotly_chart(fig)
    else:
        st.write("No se encontraron datos en Firebase.")
    
    time.sleep(1)  # Actualizar cada segundo
