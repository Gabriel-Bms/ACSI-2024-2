# FRONT END LIBRARIES------------------------------
import streamlit as st
from streamlit_option_menu import option_menu
import requests
# DEEP LEARNING LIBRARIES--------------------------
from PIL import Image
import torch
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary import summary
import seaborn as sns
# OTHER LIBRARIES------------------------------------
import path
import numpy as np
import time
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import random
from thop.profile import profile
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

st.set_page_config(page_title="main", layout="wide")

# DEEP LEARNING FUNCTIONS===================================
loc_time = time.strftime("%H%M%S", time.localtime())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ratio = 8


def gradcam(network, data, device):
    network.eval()
    
    class_names = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    # dataset = data_loader.dataset
    # random_index = random.randint(0, len(dataset) - 1)
    # data, target = dataset[random_index]

    # #Plotear la imagen original
    # img = data.clone().squeeze(0).permute(1, 2, 0)
    # img = img * 0.5 + 0.5
    # img = torch.clamp(img, 0, 1)

    # Asegurarse de que solo estamos usando una imagen
    # data = data.unsqueeze(0)  # Añadir dimensión de batch

    # Mover los datos al dispositivo correcto
    imagen_tensor = transform(data).unsqueeze(0)  # Añadir una dimensión batch
    
    img = imagen_tensor.clone().squeeze(0).permute(1, 2, 0)
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)

    imagen_tensor.requires_grad = True

    imagen_tensor = imagen_tensor.to(device)
    output = network(imagen_tensor) #torch.Size([1, 7, 16, 1])

    magnitudes = torch.sqrt(torch.sum(output**2, dim=2)).squeeze()
    predicted_class = torch.argmax(magnitudes).item()
    

    # Get the gradient of the output with respect to the parameters of the model
    output[:, predicted_class, 0, :].backward()

    # Pull the gradients out of the model
    gradients = network.get_activations_gradient()

    # Get the activations of the last convolutional layer
    activations = network.get_activations()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # weight the channels by corresponding gradients
    for i in range(128):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.numpy(), 0)

    # normalize the heatmap
    heatmap /= np.max(heatmap)

    # Convertir img a un array numpy de uint8
    img_np = (img.numpy() * 255).astype(np.uint8)

    # Redimensionar el heatmap al tamaño de la imagen original
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Normalizar el heatmap para que esté en el rango [0, 255]
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Aplicar colormap al heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Superponer el heatmap en la imagen original
    img_overlayed = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.5, heatmap_colored, 0.5, 0)

    # Mostrar la imagen original, el heatmap y el heatmap superpuesto
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].axis('off')
    axes[0].set_title('Original image')
    axes[1].imshow(heatmap_resized, cmap='jet', interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Heatmap')
    axes[2].imshow(cv2.cvtColor(img_overlayed, cv2.COLOR_BGR2RGB))
    axes[2].axis('off')
    axes[2].set_title('Superimposed Heatmap')

    st.pyplot(plt)

def test_single_image(network, data_loader, device):
    network.eval()

    dataset = data_loader.dataset
    random_index = random.randint(0, len(dataset) - 1)
    data, target = dataset[random_index]

    # Asegurarse de que solo estamos usando una imagen
    data = data.unsqueeze(0)  # Añadir dimensión de batch

    # Mover los datos al dispositivo correcto
    data = data.to(device)
    data.requires_grad = True
    output = network(data)

    print(f"Shape of the output: {output.shape}")
    # print(f"Output: {output}")

    return output, target, data.cpu(), random_index
class FixCapsNet(nn.Module):
    def __init__(self,conv_inputs,conv_outputs,
                 primary_units,primary_unit_size,
                 output_unit_size,num_classes=7,
                 init_weights=False,mode="DS"):
        super().__init__()

        self.Convolution = make_features(cfgs[mode],f_c=conv_inputs,out_c=conv_outputs)

        self.CBAM = Conv_CBAM(conv_outputs,conv_outputs)

        self.primary = Primary_Caps(in_channels=conv_outputs,#128
                                    caps_units=primary_units,#8
                                    )

        self.digits = Digits_Caps(in_units=primary_units,#8
                                   in_channels=primary_unit_size,#16*6*6=576
                                   num_units=num_classes,#classification_num
                                   unit_size=output_unit_size,#16
                                   )
        if init_weights:
            self._initialize_weights()

        # placeholder for the gradients and activations
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.Convolution(x)
        x = self.CBAM(x)

        # We get the activations and gradients from the layer of interest (layer up4 in this case)
        h = x.register_hook(self.activations_hook)
        self.activations = x.clone().detach()

        out = self.digits(self.primary(x))
        return out

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self):
        return self.activations

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    #margin_loss
    def loss(self, img_input, target, size_average=True):
        batch_size = img_input.size(0)
        # ||vc|| from the paper.
        v_mag = torch.sqrt(torch.sum(img_input**2, dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).to(device)
        m_plus, m_minus = 0.9, 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = torch.sum(L_c,1)

        if size_average:
            L_c = torch.mean(L_c)

        return L_c

class Primary_Caps(nn.Module):
    def __init__(self, in_channels, caps_units):
        super(Primary_Caps, self).__init__()

        self.in_channels = in_channels
        self.caps_units = caps_units

        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_channels)
            self.add_module("Caps_" + str(unit_idx), unit)
            return unit
        self.units = [create_conv_unit(i) for i in range(self.caps_units)]

    #no_routing
    def forward(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.caps_units)]
        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.caps_units, -1)
        # Return squashed outputs.
        return squash(u)

class Digits_Caps(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(Digits_Caps, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units

        self.W = nn.Parameter(torch.randn(1, in_channels, self.num_units, unit_size, in_units))

    #routing
    def forward(self, x):
        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)
        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).to(device)

        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            #c_ij = F.softmax(b_ij, dim=0)
            c_ij = b_ij.softmax(dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            # s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            s_j = torch.sum(c_ij * u_hat, dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = squash(s_j)#CapsuleLayer.squash

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        Caps_out = in_channels // ratio
        self.Cpas = nn.Sequential(
                        nn.Conv2d(in_channels,Caps_out,9,stride=2,groups=Caps_out, bias=False),
                    )

    def forward(self, x):
        output = self.Cpas(x)
        return output

def squash(s):
    mag_sq = torch.sum(s**2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s

class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv_CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)#LayerNorm(c2, eps=1e-6, data_format="channels_first")#
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2, reduction=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
# SAM:This is different from the paper[S. Woo, et al. "CBAM: Convolutional Block Attention Module,"].
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        me_c = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(channels, me_c, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(me_c, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def make_features(cfg: list,f_c,out_c=None,g=1,step=2):
    layers = []
    output = out_c
    f_channels = f_c
    for i in range(len(cfg)):
        if cfg[i] == 'N':
            g = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(2, 2)]
        elif v == "A":
            layers += [nn.AdaptiveMaxPool2d(20)]
        elif v == "F":
            layers += [nn.FractionalMaxPool2d(2, output_size=(20,20))]
        elif v == "B":
            f_channels = out_c
            layers += [nn.BatchNorm2d(f_channels,affine=True)]
            # layers += [LayerNorm(f_channels, eps=1e-6, data_format="channels_first")]
        elif v == "R":
            layers += [nn.ReLU(inplace=True)]
        elif v == "N":
            layers += [nn.Conv2d(f_channels,out_c,1,stride=step)]
        elif v == "C":
            layers += [nn.Conv2d(f_channels,f_channels,3,stride=step)]
        else:
            layers += [nn.Conv2d(f_channels, v, 18,stride=step,groups=g)]
            f_channels = v
    return nn.Sequential(*layers)

# Definir las clases
CLASES = ['Clase 1', 'Clase 2', 'Clase 3', 'Clase 4', 'Clase 5', 'Clase 6', 'Clase 7']

# Transformaciones para preprocesar la imagen antes de enviarla al modelo
transform = transforms.Compose([transforms.Resize((302, 302)),
                                   transforms.CenterCrop((299, 299)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])
img_title = "HAM10000"
BatchSize = 168
V_size = 40
T_size = 32
train_doc = "train525e384"
val_doc = "val525png384"
test_doc = "test525png384"
nw = min([os.cpu_count(), BatchSize if BatchSize > 1 else 0, 6])
print(f'Using {nw} dataloader workers every process.')
# get_data()
cfgs= {
    "DS": [3,'N','B','R','F'],# g = 3,  primary_unit_size = 16 * 6 * 6
    "DS2": ["C",3,'N','B','R','F'],# g = 3,  primary_unit_size = 16 * 6 * 6
    "256" : [256,'R','F'],# g = 1,  primary_unit_size = 32 * 6 * 6
    "128" : [128,'R','F'],# g = 1, primary_unit_size = 16 * 6 * 6
    "64"  : [64,'R','F'],# g = 1 , primary_unit_size = 8 * 6 * 6

}

# Create capsule network.
n_channels = 3
n_classes = 7
conv_outputs = 128 #Feature_map
num_primary_units = 8
primary_unit_size = 16 * 6 * 6  # fixme get from conv2d
output_unit_size = 16
img_size = 299
mode='128'#'DS'#'128'
network = FixCapsNet(conv_inputs=n_channels,
                    conv_outputs=conv_outputs,
                    primary_units=num_primary_units,
                    primary_unit_size=primary_unit_size,
                    num_classes=n_classes,
                    output_unit_size=16,
                    init_weights=True,
                    mode=mode)
network = network.to(device)
summary(network,(n_channels,img_size,img_size))

# network.Convolution
# network.load_state_dict(torch.load(save_PATH)).pth
#best_model_path = 'C:/Users/ldani/Documents/ACSEI/FixCaps/best_HAM10000_0923_060705'#'D:/ACSII_proyecto/FixCaps-main/augmentation/train525s8'
#workspace_dir = path.cwd()  # Directorio actual del workspace
#file_path = workspace_dir / "manati.pth"
#best_model_path = '/workspaces/blank-app/best_HAM10000_0923_060705.pth'
state_dict = torch.load("best_HAM10000_0923_060705.pth", map_location=torch.device('cpu'))

# Cargar el state_dict ignorando las claves adicionales
network.load_state_dict(state_dict, strict=False)
network = network.to(device)

class_names = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}

def predecir_clase(imagen):
    # Preprocesar la imagen
    imagen_tensor = transform(imagen).unsqueeze(0)  # Añadir una dimensión batch
    imagen_tensor = imagen_tensor.to(device)
    imagen_tensor.requires_grad = True
    output = network(imagen_tensor)
    magnitudes = torch.sqrt(torch.sum(output**2, dim=2)).squeeze()
    predicted_class = torch.argmax(magnitudes).item()
    return predicted_class

# MAIN CODE ========================================================================

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
    
def algorithm_page():
    image_file = st.file_uploader("Selecciona una imagen de tu piel", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        imagen = Image.open(image_file)
        st.image(imagen, caption="Imagen cargada", use_column_width=True)
        st.write("Procesando la imagen...")
        clase_predicha = predecir_clase(imagen)
        st.write(f"Predicted: {class_names[clase_predicha]}")
        gradcam(network, imagen, device)
        # Simulación del análisis de la imagen
        import time
        time.sleep(2)
              
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
    algorithm_page()
elif st.session_state["page"] == "Lesiones":
    skin_lession_page()
elif st.session_state["page"] == "Contacto":
    st.title("Contacto")
    st.write("Para más información, puedes contactarnos en: [email@example.com](mailto:email@example.com)")
elif st.session_state["page"] == "Signup":
    signup_page()
elif st.session_state["page"] == "Login":
    login_page()
