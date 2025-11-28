import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

# Configuración de Visualización
FILTROS_A_MOSTRAR = 16  # Veremos los 16 filtros más activos de cada capa
CMAP = 'inferno'        # Mapa de color (inferno, viridis, magma, jet)

def cargar_lut():
    data = np.load(LUT_FILE)
    tree = cKDTree(data['rgb'])
    return tree, data['temp']

def imagen_a_temperatura_normalizada(ruta_img, tree, temps):
    """
    Replica EXACTAMENTE el preprocesamiento del entrenamiento.
    """
    try:
        img = Image.open(ruta_img).convert('RGB')
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img)
        
        flat = arr.reshape(-1, 3)
        _, idx = tree.query(flat)
        matriz_temp = temps[idx].reshape((128, 128))
        
        # Normalización (-90 a 40 -> 0 a 1)
        norm = (matriz_temp + 90.0) / 130.0
        return np.clip(norm, 0.0, 1.0)
    except:
        return np.zeros((128, 128))

def obtener_secuencia_visual(fecha, tree, temps):
    # Secuencia temporal hacia atrás (t, t-10, ... t-50)
    imgs = []
    # El modelo espera orden cronológico: [t-50, t-40, ..., t]
    times = [fecha - timedelta(minutes=10*i) for i in range(INPUT_LEN_VIS)]
    times.reverse() 
    
    print(f"Cargando secuencia desde {times[0]} hasta {times[-1]}...")
    
    for t in times:
        fname = t.strftime("%Y-%m-%d-%H%M") + "_band-13.png"
        fpath = os.path.join(IMG_FOLDER, fname)
        if os.path.exists(fpath):
            matriz = imagen_a_temperatura_normalizada(fpath, tree, temps)
            imgs.append(matriz)
        else:
            print(f"⚠️ Falta imagen: {fname}")
            imgs.append(np.zeros((128, 128)))
            
    # Shape final: (1, 6, 128, 128, 1)
    return np.array(imgs)[np.newaxis, ..., np.newaxis]

def visualizar_activaciones(model, input_visual, fecha_str):
    # 1. Extraer Rama Visual
    try:
        cnn_branch = model.get_layer("Rama_Visual_CNN")
        print("✅ Rama Visual encontrada dentro del modelo híbrido.")
    except:
        print("❌ Error: No se encontró la capa 'Rama_Visual_CNN'.")
        return

    # 2. Identificar capas convolucionales
    layer_outputs = []
    layer_names = []
    
    for layer in cnn_branch.layers:
        # Buscamos las capas TimeDistributed que envuelven convoluciones
        # Queremos ver la salida después de la activación (ReLU)
        if "time_distributed" in layer.name and "activation" in layer.layer.name:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    
    if not layer_outputs:
        print("No se encontraron capas de activación para visualizar.")
        return

    # 3. Modelo de Diagnóstico
    activation_model = tf.keras.models.Model(inputs=cnn_branch.inputs, outputs=layer_outputs)
    
    # 4. Inferencia
    activations = activation_model.predict(input_visual)
    
    # 5. Graficar cada capa
    # activations es una lista de tensores. Cada tensor: (1, Time, H, W, Filters)
    
    output_dir = f"Mapas_Activacion_{fecha_str.replace(':','-').replace(' ','_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerando gráficos en carpeta: {output_dir} ...")

    for layer_name, layer_activation in zip(layer_names, activations):
        # Tomamos el ÚLTIMO frame temporal (t=0, lo más reciente)
        # Shape: (H, W, Filters)
        features = layer_activation[0, -1, :, :, :]
        
        n_features = features.shape[-1]
        size = features.shape[0]
        
        # Seleccionar los filtros con mayor "energía" (más activos)
        # Calculamos la activación media de cada filtro
        filter_activity = features.mean(axis=(0, 1))
        # Índices de los N más activos
        top_indices = filter_activity.argsort()[-FILTROS_A_MOSTRAR:][::-1]
        
        # Configurar Grid de Matplotlib (ej: 4x4 para 16 filtros)
        cols = 4
        rows = FILTROS_A_MOSTRAR // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"Capa: {layer_name} - Top {FILTROS_A_MOSTRAR} Filtros Activos\n(Frame más reciente)", fontsize=16)
        
        axes_flat = axes.flatten()
        
        for i, filter_idx in enumerate(top_indices):
            ax = axes_flat[i]
            img_tensor = features[:, :, filter_idx]
            
            # Plotear mapa de calor
            im = ax.imshow(img_tensor, cmap=CMAP)
            ax.set_title(f"Filtro #{filter_idx}", fontsize=10)
            ax.axis('off')
            
        # Barra de color común
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        
        save_path = os.path.join(output_dir, f"{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f" -> Guardado: {save_path}")

    print("\n¡Proceso terminado! Revisa la carpeta generada.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print("Error: Falta el modelo .keras")
        exit()
        
    print("Cargando LUT...")
    lut_tree, lut_temps = cargar_lut()
    
    print("Cargando Modelo...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    entrada = input("Fecha para radiografía (YYYY-MM-DD HH:MM): ")
    try:
        fecha = pd.to_datetime(entrada)
        
        # Preparar datos idéntico al entrenamiento
        X_vis = obtener_secuencia_visual(fecha, lut_tree, lut_temps)
        
        # Visualizar
        visualizar_activaciones(model, X_vis, entrada)
        
    except Exception as e:
        print(f"Error: {e}")