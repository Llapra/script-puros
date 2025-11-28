import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
from PIL import Image
from scipy.spatial import cKDTree
import joblib

# =================================================
# CONFIGURACIÓN
# =================================================
MODEL_FILE = "modelo_hibrido_final.keras"
IMG_FOLDER = "Recortes-satelitales"
LUT_FILE = "calibracion_color_temp.npz"

INPUT_LEN_VIS = 6
FILTROS_A_MOSTRAR = 16
CMAP_ACTIVACION = 'inferno' 

def cargar_lut():
    data = np.load(LUT_FILE)
    tree = cKDTree(data['rgb'])
    return tree, data['temp']

def imagen_a_temperatura(ruta_img, tree, temps):
    try:
        img = Image.open(ruta_img).convert('RGB')
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img)
        flat = arr.reshape(-1, 3)
        _, idx = tree.query(flat)
        matriz = temps[idx].reshape((128, 128))
        return matriz # Devolvemos temperatura REAL (°C)
    except:
        return np.zeros((128, 128))

def obtener_secuencia_visual(fecha, tree, temps):
    imgs_temp = []
    imgs_norm = []
    
    times = [fecha - timedelta(minutes=10*i) for i in range(INPUT_LEN_VIS)]
    times.reverse() 
    
    print(f"Cargando secuencia desde {times[0]} hasta {times[-1]}...")
    
    for t in times:
        fname = t.strftime("%Y-%m-%d-%H%M") + "_band-13.png"
        fpath = os.path.join(IMG_FOLDER, fname)
        
        if os.path.exists(fpath):
            temp_real = imagen_a_temperatura(fpath, tree, temps)
            imgs_temp.append(temp_real)
            
            # Normalización para el modelo (0-1)
            norm = (temp_real + 90.0) / 130.0
            imgs_norm.append(np.clip(norm, 0.0, 1.0))
        else:
            print(f"⚠️ Falta imagen: {fname}")
            zeros = np.zeros((128, 128))
            imgs_temp.append(zeros)
            imgs_norm.append(zeros)
            
    return np.array(imgs_temp), np.array(imgs_norm)[np.newaxis, ..., np.newaxis]

def plot_imagen_entrada_con_contexto(img_temp, fecha_str, output_dir):
    """
    Grafica la imagen de entrada (Temperatura) con una barra de color
    que indica qué tipo de nube es.
    """
    plt.figure(figsize=(8, 7))
    
    # Crear un Colormap personalizado Meteorológico (IR Enhancement)
    # Azul/Blanco = Frío (Nubes altas), Rojo/Amarillo = Templado, Negro = Caliente
    # Invertimos 'jet' o 'turbo' para que frío sea "brillante"
    cmap = plt.get_cmap('turbo_r') 
    
    # Definimos límites físicos para la visualización
    vmin, vmax = -90, 40
    
    im = plt.imshow(img_temp, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"Entrada al Modelo (T° Brillo)\n{fecha_str}", fontsize=14)
    plt.axis('off')
    
    # Barra de Color con Etiquetas Físicas
    cbar = plt.colorbar(im, shrink=0.8, aspect=20, pad=0.05)
    cbar.set_label('Temperatura de Brillo (°C)', rotation=270, labelpad=15)
    
    # Añadir marcas de texto para nubes
    # Posiciones aproximadas en la barra
    cbar.ax.text(1.5, -80, 'Cumulonimbus (Tope)', va='center', fontsize=8, color='blue')
    cbar.ax.text(1.5, -40, 'Nubes Medias/Altas', va='center', fontsize=8, color='green')
    cbar.ax.text(1.5, 0,   'Nubes Bajas', va='center', fontsize=8, color='orange')
    cbar.ax.text(1.5, 25,  'Suelo / Océano', va='center', fontsize=8, color='red')
    
    save_path = os.path.join(output_dir, "00_Input_Realidad_Fisica.png")
    plt.savefig(save_path)
    plt.close()
    print(f" -> Guardado Contexto Físico: {save_path}")

def visualizar_activaciones(model, input_visual, imgs_temp, fecha_str):
    output_dir = f"Mapas_Activacion_{fecha_str.replace(':','-').replace(' ','_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Graficar la Realidad Física (Input)
    # Usamos la última imagen de la secuencia (la más reciente)
    plot_imagen_entrada_con_contexto(imgs_temp[-1], fecha_str, output_dir)

    # 2. Extraer Rama Visual
    try:
        cnn_branch = model.get_layer("Rama_Visual_CNN")
    except:
        print("❌ Error: No se encontró la capa 'Rama_Visual_CNN'.")
        return

    # 3. Identificar capas
    layer_outputs = []
    layer_names = []
    
    for layer in cnn_branch.layers:
        if "time_distributed" in layer.name and "activation" in layer.layer.name:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    
    if not layer_outputs: return

    # 4. Inferencia de Activaciones
    activation_model = tf.keras.models.Model(inputs=cnn_branch.inputs, outputs=layer_outputs)
    activations = activation_model.predict(input_visual)
    
    print(f"Generando mapas de activación...")

    for layer_name, layer_activation in zip(layer_names, activations):
        # Último frame
        features = layer_activation[0, -1, :, :, :]
        
        # Seleccionar filtros más activos
        filter_activity = features.mean(axis=(0, 1))
        top_indices = filter_activity.argsort()[-FILTROS_A_MOSTRAR:][::-1]
        
        cols = 4
        rows = FILTROS_A_MOSTRAR // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"Qué 've' la IA en: {layer_name}\n(Filtros más activos)", fontsize=16)
        
        axes_flat = axes.flatten()
        
        for i, filter_idx in enumerate(top_indices):
            ax = axes_flat[i]
            img_tensor = features[:, :, filter_idx]
            
            im = ax.imshow(img_tensor, cmap=CMAP_ACTIVACION)
            ax.set_title(f"Filtro {filter_idx}", fontsize=9)
            ax.axis('off')
            
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label="Intensidad de Activación (Adimensional)")
        
        save_path = os.path.join(output_dir, f"{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f" -> Guardado Capa: {save_path}")

    print(f"\n¡Listo! Carpeta: {output_dir}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print("Error: Falta modelo.")
        exit()
        
    print("Cargando LUT...")
    lut_tree, lut_temps = cargar_lut()
    
    print("Cargando Modelo...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    entrada = input("Fecha para radiografía (YYYY-MM-DD HH:MM): ")
    try:
        fecha = pd.to_datetime(entrada)
        
        # Obtenemos tanto la temp real (para graficar) como la normalizada (para la IA)
        imgs_temp, X_vis_model = obtener_secuencia_visual(fecha, lut_tree, lut_temps)
        
        visualizar_activaciones(model, X_vis_model, imgs_temp, entrada)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()