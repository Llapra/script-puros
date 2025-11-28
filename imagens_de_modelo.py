import os
import numpy as np
import pandas as pd
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
SCALER_FILE = "scaler_hibrido.gz"
CSV_FILE = "master_dataset_continuo.csv"
IMG_FOLDER = "Recortes-satelitales"
LUT_FILE = "calibracion_color_temp.npz"

# Parámetros fijos del modelo
INPUT_LEN_VIS = 6
INPUT_LEN_TAB = 48

# Capas de interés para visualizar (Nombres internos de la CNN)
# Puedes ver los nombres con model.summary() o model.get_layer('Rama_Visual_CNN').summary()
# Usualmente son 'time_distributed', 'time_distributed_1', etc.
# Como es un modelo anidado, accederemos a la sub-capa.

def cargar_herramientas():
    print("Cargando modelo y datos...")
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    df = pd.read_csv(CSV_FILE, index_col='TIMESTAMP', parse_dates=True)
    
    data_lut = np.load(LUT_FILE)
    tree = cKDTree(data_lut['rgb'])
    temps = data_lut['temp']
    
    return model, scaler, df, tree, temps

def preparar_imagen(fecha, tree, temps):
    # Lógica de carga idéntica a la aplicación
    imgs = []
    times = [fecha - timedelta(minutes=10*i) for i in range(INPUT_LEN_VIS)]
    times.reverse()
    
    for t in times:
        fname = t.strftime("%Y-%m-%d-%H%M") + "_band-13.png"
        fpath = os.path.join(IMG_FOLDER, fname)
        if os.path.exists(fpath):
            img = Image.open(fpath).convert('RGB').resize((128, 128))
            arr = np.array(img).reshape(-1, 3)
            _, idx = tree.query(arr)
            matriz = temps[idx].reshape(128, 128)
            norm = (matriz + 90.0) / 130.0
            imgs.append(np.clip(norm, 0.0, 1.0))
        else:
            imgs.append(np.zeros((128, 128)))
            
    return np.array(imgs)[np.newaxis, ..., np.newaxis] # (1, 6, 128, 128, 1)

def visualizar_capas_internas(model, input_visual):
    """
    Extrae y visualiza los mapas de características de la rama CNN.
    """
    # 1. Acceder a la Rama Visual (que es un modelo anidado dentro del principal)
    try:
        cnn_branch = model.get_layer("Rama_Visual_CNN")
    except ValueError:
        print("Error: No se encontró la capa 'Rama_Visual_CNN'. Verifica el nombre en model.summary()")
        return

    # 2. Crear un modelo extractor de características
    # Queremos ver la salida de cada bloque TimeDistributed(Conv2D)
    # Identificamos las capas convolucionales por nombre o tipo
    layer_outputs = []
    layer_names = []
    
    print("\nEstructura de la Rama Visual:")
    for layer in cnn_branch.layers:
        # Buscamos capas TimeDistributed que contengan Conv2D o MaxPooling
        if "time_distributed" in layer.name and "conv2d" in layer.layer.name:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
            print(f" -> Capa seleccionada para visualización: {layer.name} ({layer.output.shape})")

    if not layer_outputs:
        print("No se encontraron capas convolucionales para visualizar.")
        return

    # Modelo de Activación: Input -> Mapas de Características
    activation_model = tf.keras.models.Model(inputs=cnn_branch.inputs, outputs=layer_outputs)

    # 3. Predicción (Forward Pass)
    # Solo pasamos la parte visual
    activations = activation_model.predict(input_visual)

    # 4. Visualización
    # Mostramos la activación del ÚLTIMO frame de la secuencia temporal (t=0, el más reciente)
    # Las activaciones tienen forma (1, Time, Height, Width, Filters)
    # Tomamos Time=-1 (último)
    
    images_per_row = 8
    
    for layer_name, layer_activation in zip(layer_names, activations):
        # layer_activation shape: (1, 6, H, W, Filters)
        # Tomamos el último instante de tiempo
        features = layer_activation[0, -1, :, :, :] 
        
        n_features = features.shape[-1] # Número de filtros
        size = features.shape[1]       # Ancho/Alto (ej 64, 32, 16...)
        
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = features[:, :, col * images_per_row + row]
                
                # Procesamiento para mejorar contraste visual
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-5)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(f"Activaciones de Capa: {layer_name} (Frame más reciente)")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
        # Guardar
        save_name = f"activacion_{layer_name}.png"
        plt.savefig(save_name)
        print(f"Guardado mapa de características: {save_name}")
    
    plt.show()

if __name__ == "__main__":
    model, scaler, df, tree, temps = cargar_herramientas()
    
    fecha_str = input("Fecha para visualizar (YYYY-MM-DD HH:MM): ")
    fecha = pd.to_datetime(fecha_str)
    
    print("Preparando imagen...")
    # El input tabular no se usa para visualizar la CNN, pero el modelo lo pide si usáramos el completo.
    # Aquí usamos directamente el sub-modelo visual, así que solo necesitamos X_vis.
    X_vis = preparar_imagen(fecha, tree, temps)
    
    visualizar_capas_internas(model, X_vis)