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
OUTPUT_FOLDER = "Pronosticos_CSV" # Carpeta para guardar los CSVs

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

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =================================================
# FUNCIONES AUXILIARES
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
        return matriz_temp, np.clip(norm, 0.0, 1.0)
    except:
        return np.zeros((128, 128)), np.zeros((128, 128))

def preparar_input_hibrido(fecha_inicio, df, scaler, tree, temps):
    # --- 1. TABULAR ---
    t_end = fecha_inicio
    t_start = t_end - timedelta(minutes=30 * (INPUT_LEN_TAB - 1))
    
    chunk = df.loc[t_start : t_end].copy()
    
    if len(chunk) < INPUT_LEN_TAB:
        print(f"Faltan datos tabulares. Padding...")
        return None, None, None

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
    imgs_norm_array = []
    imgs_real_array = []
    
    tiempos_img = [fecha_inicio - timedelta(minutes=10 * i) for i in range(INPUT_LEN_VIS)]
    tiempos_img.reverse()
    
    for t in tiempos_img:
        fname = t.strftime("%Y-%m-%d-%H%M") + "_band-13.png"
        fpath = os.path.join(IMG_FOLDER, fname)
        
        if os.path.exists(fpath):
            matriz_real, matriz_norm = imagen_a_temperatura(fpath, tree, temps)
        else:
            # Si falta imagen, usar cuadro negro
            matriz_real, matriz_norm = np.zeros((128, 128)), np.zeros((128, 128))
            
        imgs_norm_array.append(matriz_norm)
        imgs_real_array.append(matriz_real)
    
    # Sanitizar Visual
    vis_input = np.array(imgs_norm_array)[..., np.newaxis]
    vis_input = np.nan_to_num(vis_input, nan=0.0)
    
    return np.array([vis_input]), np.array([tab_scaled]), imgs_real_array[-1]

def interpretar_nubosidad(df_pred):
    print("\n" + "="*40)
    print("   REPORTE METEOROLÓGICO AUTOMÁTICO   ")
    print("="*40)
    print(f"{'HORA':<10} | {'ESTADO DEL CIELO':<25} | {'RADIACIÓN':<10}")
    print("-" * 50)
    
    for tiempo, fila in df_pred.iterrows():
        rad = fila['NETRAD']
        hora = tiempo.hour
        es_dia = 7 <= hora <= 19
        estado = "Desconocido"
        icono = "❓"
        
        if es_dia:
            if rad > 400: estado, icono = "DESPEJADO / SOLEADO", "☀️"
            elif rad > 200: estado, icono = "PARCIALMENTE NUBLADO", "⛅"
            else: estado, icono = "NUBLADO / CUBIERTO", "☁️"
        else:
            if rad < -40: estado, icono = "DESPEJADO (Frío)", "🌙"
            else: estado, icono = "NUBLADO (Templado)", "☁️"
            
        print(f"{tiempo.strftime('%H:%M')}      | {icono} {estado:<22} | {rad:>6.1f} W/m²")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    print("--- SISTEMA DE PRONÓSTICO HÍBRIDO (VISUALIZACIÓN COMPLETA) ---")
    
    if not os.path.exists(MODEL_FILE):
        print("Error: No hay modelo entrenado.")
        exit()

    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    df = pd.read_csv(CSV_FILE, index_col='TIMESTAMP', parse_dates=True, low_memory=False)
    for col in FEATURE_VARS: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    lut_tree, lut_temps = cargar_lut()
    
    print("Sistema listo.")
    
    while True:
        try:
            entrada = input("\nFecha Pronóstico (YYYY-MM-DD HH:MM): ")
            if not entrada: break
            
            fecha_pivote = pd.to_datetime(entrada)
            
            # Obtenemos datos Y la imagen real
            X_vis, X_tab, img_input_real = preparar_input_hibrido(fecha_pivote, df, scaler, lut_tree, lut_temps)
            
            if X_vis is None:
                print("Error: Datos insuficientes.")
                continue
                
            print("Calculando inferencia...")
            pred_scaled = model.predict([X_vis, X_tab], verbose=0)
            
            if np.isnan(pred_scaled).any():
                print("⚠️ ALERTA: NaNs detectados. Reemplazando por ceros.")
                pred_scaled = np.nan_to_num(pred_scaled)

            # Des-escalar
            cols_scaler = FEATURE_VARS + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            dummy = np.zeros((OUTPUT_LEN, len(cols_scaler)))
            target_indices = [cols_scaler.index(t) for t in TARGET_VARS]
            dummy[:, target_indices] = pred_scaled[0]
            
            pred_final = scaler.inverse_transform(dummy)[:, target_indices]
            
            fechas_futuras = [fecha_pivote + timedelta(minutes=30 * (i+1)) for i in range(OUTPUT_LEN)]
            df_pred = pd.DataFrame(pred_final, index=fechas_futuras, columns=TARGET_VARS)
            
            # Clipping Físico
            if 'RH_1_1_1' in df_pred: df_pred['RH_1_1_1'] = df_pred['RH_1_1_1'].clip(0, 100)
            if 'TKE' in df_pred: df_pred['TKE'] = df_pred['TKE'].clip(lower=0)
            
            # Reporte Texto
            interpretar_nubosidad(df_pred)
            
            # --- GUARDAR CSV ---
            nombre_archivo_csv = f"pronostico_{fecha_pivote.strftime('%Y%m%d_%H%M')}.csv"
            ruta_csv = os.path.join(OUTPUT_FOLDER, nombre_archivo_csv)
            df_pred.to_csv(ruta_csv)
            print(f"\n✅ Pronóstico guardado en: {ruta_csv}")

            
            # --- GRAFICAR MEJORADO (SEPARADO) ---
            df_real = df.loc[fechas_futuras[0] : fechas_futuras[-1]]
            
            # 1. Figura para la Imagen Satelital (Independiente)
            plt.figure(figsize=(6, 6))
            cmap_ir = plt.get_cmap('turbo_r') 
            im = plt.imshow(img_input_real, cmap=cmap_ir, vmin=-90, vmax=40)
            plt.title(f"Estado del Cielo (Tope Nubes)\n{fecha_pivote}", fontsize=14)
            plt.axis('off')
            
            cbar = plt.colorbar(im, orientation='vertical', pad=0.05, fraction=0.046)
            cbar.set_label("Temp. Brillo (°C)")
            cbar.ax.text(1.5, -90, "Nubes Altas", va='center', fontsize=8, color='blue')
            cbar.ax.text(1.5, 40, "Suelo", va='center', fontsize=8, color='red')
            plt.tight_layout()
            plt.show()

            # 2. Figura para los Gráficos de Series de Tiempo
            fig, axes = plt.subplots(len(TARGET_VARS), 1, figsize=(12, 16), sharex=True)
            
            for i, var in enumerate(TARGET_VARS):
                ax = axes[i]
                
                # Pronóstico
                ax.plot(df_pred.index, df_pred[var], 'o-', label='Pronóstico IA', color='blue', markersize=4)
                
                texto_error = ""
                # Real
                if not df_real.empty:
                    idx = df_real.index.intersection(df_pred.index)
                    if len(idx)>0:
                        y_true = df_real.loc[idx, var]
                        y_pred = df_pred.loc[idx, var]
                        ax.plot(idx, y_true, '--', label='Real', color='gray', alpha=0.7)
                        
                        # CÁLCULO DE ERROR
                        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
                        mae = np.mean(np.abs(y_pred - y_true))
                        texto_error = f"RMSE: {rmse:.2f} | MAE: {mae:.2f}"

                ax.set_ylabel(LABELS.get(var, var), fontsize=10)
                if i == 0: ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                
                # Mostrar error
                if texto_error:
                    ax.text(0.02, 0.88, texto_error, transform=ax.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=9, fontweight='bold')
                
                if i == len(TARGET_VARS) - 1:
                    plt.xticks(rotation=30, ha='right')
            
            plt.suptitle(f"Pronóstico Meteorológico: {fecha_pivote}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        