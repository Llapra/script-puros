import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from PIL import Image
from scipy.spatial import cKDTree

# =================================================
# CONFIGURACIÓN
# =================================================
MODEL_FILE = "modelo_hibrido_final.keras"
SCALER_FILE = "scaler_hibrido.gz"
CSV_FILE = "master_dataset_continuo.csv"
IMG_FOLDER = "Recortes-satelitales" 
LUT_FILE = "calibracion_color_temp.npz"

INPUT_LEN_TAB = 48
INPUT_LEN_VIS = 6
OUTPUT_LEN = 28

TARGET_VARS = ['NETRAD', 'RH_1_1_1', 'TA_1_1_1', 'T_DP_1_1_1', 'TKE']
FEATURE_VARS = [
    'TA_1_1_1', 'WS', 'TKE', 'H2O_density', 'Uz', 'RH_1_1_1', 'NETRAD', 
    'SW_IN', 'LW_IN','SW_OUT', 'LW_OUT', 'PA', 'WD_SONIC', 'USTAR',
    'T_DP_1_1_1', 'T_SONIC_SIGMA', 'Uz_SIGMA', 'e_amb', 'G_plate_1_1_1', 'H_QC'
]

LABELS = {
    'NETRAD': 'Radiación Neta (W/m²)',
    'RH_1_1_1': 'Humedad Relativa (%)',
    'TA_1_1_1': 'Temperatura Aire (°C)',
    'T_DP_1_1_1': 'Punto de Rocío (°C)',
    'TKE': 'Turbulencia (TKE)'
}

# =================================================
# FUNCIONES
# =================================================
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
        matriz_temp = temps[idx].reshape((128, 128))
        
        # Normalización (-90 a 40 -> 0 a 1)
        norm = (matriz_temp + 90.0) / 130.0
        return np.clip(norm, 0.0, 1.0)
    except:
        return np.zeros((128, 128))

def preparar_input_hibrido(fecha_inicio, df, scaler, tree, temps):
    # --- 1. TABULAR ---
    t_end = fecha_inicio
    t_start = t_end - timedelta(minutes=30 * (INPUT_LEN_TAB - 1))
    
    chunk = df.loc[t_start : t_end].copy()
    
    if len(chunk) < INPUT_LEN_TAB:
        print(f"Faltan datos tabulares. Padding...")
        # Lógica de emergencia: rellenar con la última fila
        # (Mejor que fallar, pero la predicción será plana)
        # No implementado aquí por brevedad, retorna None
        return None, None

    # Rellenar huecos
    chunk = chunk.ffill().bfill().fillna(0)

    chunk['hour_sin'] = np.sin(2 * np.pi * chunk.index.hour / 24.0)
    chunk['hour_cos'] = np.cos(2 * np.pi * chunk.index.hour / 24.0)
    chunk['day_sin'] = np.sin(2 * np.pi * chunk.index.dayofyear / 365.0)
    chunk['day_cos'] = np.cos(2 * np.pi * chunk.index.dayofyear / 365.0)
    
    cols_feat = FEATURE_VARS + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    # Escalar y Sanitizar Tabular
    tab_scaled = scaler.transform(chunk[cols_feat])
    tab_scaled = np.nan_to_num(tab_scaled, nan=0.0) # BLINDAJE
    
    # --- 2. VISUAL ---
    imgs_array = []
    tiempos_img = [fecha_inicio - timedelta(minutes=10 * i) for i in range(INPUT_LEN_VIS)]
    tiempos_img.reverse()
    
    for t in tiempos_img:
        fname = t.strftime("%Y-%m-%d-%H%M") + "_band-13.png"
        fpath = os.path.join(IMG_FOLDER, fname)
        
        if os.path.exists(fpath):
            matriz = imagen_a_temperatura(fpath, tree, temps)
        else:
            # Si falta imagen, usar cuadro negro
            matriz = np.zeros((128, 128))
            
        imgs_array.append(matriz)
    
    # Sanitizar Visual (por si acaso)
    vis_input = np.array(imgs_array)[..., np.newaxis]
    vis_input = np.nan_to_num(vis_input, nan=0.0)
    
    return np.array([vis_input]), np.array([tab_scaled])

if __name__ == "__main__":
    print("--- SISTEMA DE PRONÓSTICO HÍBRIDO (BLINDADO) ---")
    
    if not os.path.exists(MODEL_FILE):
        print("Error: No hay modelo entrenado.")
        exit()

    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    df = pd.read_csv(CSV_FILE, index_col='TIMESTAMP', parse_dates=True, low_memory=False)
    for col in FEATURE_VARS: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    lut_tree, lut_temps = cargar_lut()
    
    print("Listo.")
    
    while True:
        try:
            entrada = input("\nFecha (YYYY-MM-DD HH:MM): ")
            if not entrada: break
            
            fecha_pivote = pd.to_datetime(entrada)
            
            X_vis, X_tab = preparar_input_hibrido(fecha_pivote, df, scaler, lut_tree, lut_temps)
            
            if X_vis is None:
                print("Error en datos de entrada.")
                continue
                
            print("Prediciendo...")
            pred_scaled = model.predict([X_vis, X_tab], verbose=0)
            
            # BLINDAJE SALIDA: Limpiar NaNs de la red
            if np.isnan(pred_scaled).any():
                print("⚠️ ALERTA: La red predijo NaNs. Reemplazando por ceros.")
                pred_scaled = np.nan_to_num(pred_scaled)

            # Des-escalar
            cols_scaler = FEATURE_VARS + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            dummy = np.zeros((OUTPUT_LEN, len(cols_scaler)))
            target_indices = [cols_scaler.index(t) for t in TARGET_VARS]
            dummy[:, target_indices] = pred_scaled[0]
            
            pred_final = scaler.inverse_transform(dummy)[:, target_indices]
            
            fechas_futuras = [fecha_pivote + timedelta(minutes=30 * (i+1)) for i in range(OUTPUT_LEN)]
            df_pred = pd.DataFrame(pred_final, index=fechas_futuras, columns=TARGET_VARS)
            
            # CLIPPING FÍSICO
            if 'RH_1_1_1' in df_pred: df_pred['RH_1_1_1'] = df_pred['RH_1_1_1'].clip(0, 100)
            if 'TKE' in df_pred: df_pred['TKE'] = df_pred['TKE'].clip(lower=0)
            
            # Graficar
            df_real = df.loc[fechas_futuras[0] : fechas_futuras[-1]]
            
            fig, axes = plt.subplots(len(TARGET_VARS), 1, figsize=(12, 15), sharex=True)
            for i, var in enumerate(TARGET_VARS):
                ax = axes[i]
                ax.plot(df_pred.index, df_pred[var], 'o-', label='Pronóstico IA', color='blue')
                
                if not df_real.empty:
                    idx = df_real.index.intersection(df_pred.index)
                    if len(idx)>0:
                        ax.plot(idx, df_real.loc[idx, var], '--', label='Real', color='gray')
                
                ax.set_ylabel(LABELS.get(var, var))
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                
            plt.xlabel("Hora")
            plt.suptitle(f"Pronóstico Híbrido: {fecha_pivote}", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

def interpretar_nubosidad(df_pred):
    print("\n--- REPORTE METEOROLÓGICO INTERPRETADO ---")
    
    for tiempo, fila in df_pred.iterrows():
        rad = fila['NETRAD']
        hora = tiempo.hour
        es_dia = 7 <= hora <= 19
        
        estado = ""
        
        if es_dia:
            # Umbrales aproximados (ajustar según tu latitud/estación)
            if rad > 400: estado = "☀️ SOLEADO"
            elif rad > 200: estado = "⛅ PARCIALMENTE NUBLADO"
            else: estado = "☁️ NUBLADO / CUBIERTO"
        else:
            # Noche
            if rad < -40: estado = "🌙 DESPEJADO (Frío)"
            else: estado = "☁️ NUBLADO (Templado)"
            
        print(f"{tiempo.strftime('%H:%M')} -> {estado} ({rad:.1f} W/m²)")