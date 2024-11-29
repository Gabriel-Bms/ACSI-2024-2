import streamlit as st
from streamlit_option_menu import option_menu
import requests

# Configuración inicial
st.set_page_config(page_title="main", layout="wide")

# Inicialización de claves en session_state
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}  # Usuario por defecto

if "page" not in st.session_state:
    st.session_state["page"] = "Inicio"  # Página predeterminada

# Funciones para autenticación
def login_page():
    st.title("Iniciar Sesión")
    st.markdown("Por favor, introduce tus credenciales para continuar.")

    username = st.text_input("Usuario", placeholder="Introduce tu usuario")
    password = st.text_input("Contraseña", type="password", placeholder="Introduce tu contraseña")

    if st.button("Iniciar Sesión"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["authentication_status"] = True
            st.success(f"¡Bienvenido de nuevo, {username}!")
            st.session_state["page"] = "Inicio"
        else:
            st.session_state["authentication_status"] = False
            st.error("Usuario o contraseña incorrectos.")

def signup_page():
    st.title("Registrarse")
    st.markdown("Crea una cuenta para utilizar la aplicación.")

    new_username = st.text_input("Nuevo Usuario", placeholder="Elige un nombre de usuario")
    new_password = st.text_input("Nueva Contraseña", type="password", placeholder="Elige una contraseña")

    if st.button("Registrar"):
        if new_username in st.session_state["users"]:
            st.error("Este usuario ya existe. Por favor, elige otro.")
        elif len(new_password) < 6:
            st.error("La contraseña debe tener al menos 6 caracteres.")
        else:
            st.session_state["users"][new_username] = new_password
            st.success("¡Usuario registrado con éxito! Ahora puedes iniciar sesión.")
            st.session_state["page"] = "Login"

# Función para la página principal
def main_page():
    file_id = "1fSc3BbF_ZVC0_3Y0vs9jmga4bgWuF3CU"
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    response = requests.get(url)

    # Fondo personalizado
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://wallpapercave.com/wp/wp6690947.png");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .title {
            font-size: 84px;
            font-family: 'Helvetica', sans-serif; 
            color: #FFFFFF;
            text-align: left; 
            margin-bottom: 30px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">FixCaps:<br>Clasificador de lesiones de piel</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1: 
        st.image(response.content,width=400)
    with col2:
        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; border: 2px solid #ddd;">
            <h3 style="color: #000;">Análisis de Tumores en la Piel</h3>
            <p>El cáncer de piel es uno de los tipos de cáncer más comunes diagnosticados en los Estados Unidos. Un informe ha mostrado que la tasa de supervivencia a cinco años del melanoma maligno localizado es del 99% cuando se diagnostica y trata de manera temprana, mientras que la tasa de supervivencia del melanoma avanzado es solo del 25%<br><br>Por lo tanto, es particularmente importante detectar y clasificar imágenes dermatoscópicas para que el cáncer de piel pueda ser diagnosticado de manera temprana.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.title("Sube tu Imagen para Predicción") 
    # Subida de imagen
    image_file = st.file_uploader("Selecciona una imagen de tu piel", type=["jpg", "jpeg", "png"])
    # Espaciado
    st.write("")  
    # Botón para generar predicción
    if st.button("Generar predicción y mapa de calor"):
        if image_file is not None:
            st.success("Lesión predicha: ")
        else:
            st.warning("Por favor, sube una imagen antes de generar la predicción.")

def skin_lession_page():
    st.title("Lesiones de Piel")
    st.write("A continuación, te presentamos los principales tipos de lesiones de piel. Haz clic en una imagen para obtener más detalles.")

    # Datos de las lesiones
    lesions = [
        {"name": "Melanoma", "image": "https://www.isdin.com/es/blog/wp-content/uploads/2024/05/image-4-1-1-900x463.png", "description": "Lesión cutánea peligrosa que puede propagarse rápidamente."},
        {"name": "Carcinoma de células basales", "image": "https://via.placeholder.com/150", "description": "Forma más común de cáncer de piel, crecimiento lento."},
        {"name": "Carcinoma de células escamosas", "image": "https://via.placeholder.com/150", "description": "Tipo de cáncer de piel que puede ser agresivo si no se trata."},
        {"name": "Quiste Sebáceo", "image": "https://via.placeholder.com/150", "description": "Bulto no canceroso debajo de la piel, relleno de grasa."},
        {"name": "Lentigo Solar", "image": "https://via.placeholder.com/150", "description": "Manchas oscuras en la piel causadas por la exposición al sol."},
        {"name": "Queratosis Actínica", "image": "https://via.placeholder.com/150", "description": "Lesión escamosa precancerosa causada por la exposición al sol."},
        {"name": "Nevus", "image": "https://via.placeholder.com/150", "description": "Lunar o marca de nacimiento que puede variar en tamaño y color."},
    ]

    # Mostrar las lesiones en un formato de rejilla
    cols = st.columns(3)  # Configurar 3 columnas para organizar las imágenes
    for index, lesion in enumerate(lesions):
        with cols[index % 3]:  # Distribuir imágenes en las columnas
            st.image(lesion["image"], caption=lesion["name"], use_column_width=True)
            st.markdown(f"**{lesion['name']}**")
            st.write(lesion["description"])


selected = option_menu(
    menu_title=None,
    options=["Inicio", "Algoritmo", "Lesiones de piel", "Contacto", "Registrarse", "Inicio Sesión"],
    icons=["house", "capsule", "heart-pulse-fill", "envelope", "person-add", "person-check"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Actualización de la página actual basada en selección
if selected == "Inicio":
    st.session_state["page"] = "Inicio"
elif selected == "Algoritmo":
    st.session_state["page"] = "Algoritmo"
elif selected == "Lesiones de piel":
    st.session_state["page"] = "Lesiones"
elif selected == "Contacto":
    st.session_state["page"] = "Contacto"
elif selected == "Registrarse":
    st.session_state["page"] = "Signup"
elif selected == "Inicio Sesión":
    st.session_state["page"] = "Login"

# Renderizado dinámico de páginas
if st.session_state["page"] == "Inicio":
    main_page()
elif st.session_state["page"] == "Algoritmo":
    st.title("Detección de Tumores en la Piel")
    st.write("Sube una imagen para analizar.")
elif st.session_state["page"] == "Lesiones":
    skin_lession_page()
elif st.session_state["page"] == "Contacto":
    st.title("Contacto")
    st.write("Para más información, puedes contactarnos en: [email@example.com](mailto:email@example.com)")
elif st.session_state["page"] == "Signup":
    signup_page()
elif st.session_state["page"] == "Login":
    login_page()