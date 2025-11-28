import pandas as pd
import os
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import matplotlib.ticker as mticker

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
UNIFIED_DATA_FILE = "master_dataset_continuo.csv"
MODEL_FILENAME = "nowcasting_transformer_clouds.keras"
SCALER_FILENAME = "data_scaler_clouds.gz"

TARGET_VARS = ['NETRAD', 'RH_1_1_1', 'TA_1_1_1', 'T_DP_1_1_1', 'TKE']
FEATURE_VARS = [
    'TA_1_1_1', 'WS', 'TKE', 'H2O_density', 'Uz', 'RH_1_1_1', 'NETRAD', 
    'SW_IN', 'LW_IN','SW_OUT', 'LW_OUT', 'PA', 'WD_SONIC', 'USTAR',
    'T_DP_1_1_1', 'T_SONIC_SIGMA', 'Uz_SIGMA', 'e_amb', 'G_plate_1_1_1', 'H_QC'
]

INPUT_LENGTH = 48
OUTPUT_LENGTH = 12

VARIABLE_LABELS = {
    'NETRAD': 'Radiación Neta (W/m²)',
    'RH_1_1_1': 'Humedad Relativa (%)',
    'TA_1_1_1': 'Temp. Aire (°C)',
    'T_DP_1_1_1': 'Punto de Rocío (°C)',
    'TKE': 'Turbulencia (TKE)'
}

def create_cyclical_features(df):
    df_out = df.copy()
    df_out['hour_sin'] = np.sin(2 * np.pi * df_out.index.hour / 24.0)
    df_out['hour_cos'] = np.cos(2 * np.pi * df_out.index.hour / 24.0)
    df_out['day_sin'] = np.sin(2 * np.pi * df_out.index.dayofyear / 365.0)
    df_out['day_cos'] = np.cos(2 * np.pi * df_out.index.dayofyear / 365.0)
    return df_out

if __name__ == "__main__":
    print("--- MODO PRUEBA VISUAL (BLINDADO) ---")
    
    if not os.path.exists(MODEL_FILENAME):
        print("❌ Error: No hay modelo entrenado.")
        exit()
        
    # Cargar modelo manejando objetos custom si es necesario
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
    except Exception as e:
        print(f"Error cargando modelo (quizás necesites recompilar): {e}")
        # Si falla por capas custom, a veces hay que pasar custom_objects, 
        # pero con capas estándar de Keras no debería pasar.
        exit()

    scaler = joblib.load(SCALER_FILENAME)
    
    # Carga segura del CSV
    df = pd.read_csv(UNIFIED_DATA_FILE, index_col='TIMESTAMP', parse_dates=True, low_memory=False)
    # Asegurar numérico
    for col in FEATURE_VARS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\nRango datos disponibles: {df.index.min()} a {df.index.max()}")
    
    while True:
        try:
            user_input = input(f"\nFecha inicio pronóstico (YYYY-MM-DD HH:MM): ")
            if not user_input: continue

            start_time = pd.to_datetime(user_input)
            
            # Definir ventanas de tiempo
            end_input = start_time + timedelta(minutes=30 * (INPUT_LENGTH - 1))
            start_forecast = end_input + timedelta(minutes=30)
            end_forecast = start_forecast + timedelta(minutes=30 * (OUTPUT_LENGTH - 1))
            
            # Extraer chunk de entrada
            chunk = df.loc[start_time : end_input].copy()
            
            if len(chunk) != INPUT_LENGTH:
                print(f"❌ Datos insuficientes. Se encontraron {len(chunk)} filas, se necesitan {INPUT_LENGTH}.")
                continue
            
            # --- 1. SANITIZACIÓN DE ENTRADA (NIVEL 1) ---
            # Rellenar huecos
            chunk = chunk.ffill().bfill().fillna(0)
            
            # Preparar features
            input_feat = create_cyclical_features(chunk[FEATURE_VARS])
            
            # Escalar
            input_scaled = scaler.transform(input_feat)
            
            # --- 2. SANITIZACIÓN DE ENTRADA (NIVEL 2) ---
            # Eliminar cualquier NaN/Infinito residual que el scaler pudiera haber dejado o creado
            if np.isnan(input_scaled).any() or np.isinf(input_scaled).any():
                print("⚠️ Advertencia: Se detectaron NaNs/Infs en la entrada escalada. Limpiando...")
                input_scaled = np.nan_to_num(input_scaled, nan=0.0, posinf=1.0, neginf=0.0)

            input_tensor = np.expand_dims(input_scaled, axis=0)
            
            # Predicción
            print(f"Generando pronóstico...")
            pred_scaled = model.predict(input_tensor, verbose=0)
            
            # --- 3. SANITIZACIÓN DE SALIDA ---
            if np.isnan(pred_scaled).any():
                print("⚠️ ALERTA: El modelo predijo NaNs. Reemplazando con ceros.")
                pred_scaled = np.nan_to_num(pred_scaled)

            # Des-escalar
            cols_all = input_feat.columns.tolist()
            dummy = np.zeros((OUTPUT_LENGTH, len(cols_all)))
            target_indices = [cols_all.index(v) for v in TARGET_VARS]
            dummy[:, target_indices] = pred_scaled[0]
            
            pred_unscaled = scaler.inverse_transform(dummy)[:, target_indices]
            
            forecast_index = pd.date_range(start=start_forecast, periods=OUTPUT_LENGTH, freq='30min')
            pred_df = pd.DataFrame(pred_unscaled, index=forecast_index, columns=TARGET_VARS)
            
            # --- 4. CLIPPING FÍSICO (Límites de la realidad) ---
            # Humedad relativa entre 0 y 100
            if 'RH_1_1_1' in pred_df.columns:
                pred_df['RH_1_1_1'] = pred_df['RH_1_1_1'].clip(0, 100)
            
            # TKE no puede ser negativa
            if 'TKE' in pred_df.columns:
                pred_df['TKE'] = pred_df['TKE'].clip(lower=0)

            # Datos reales para comparar
            real_future = df.loc[start_forecast : end_forecast]

            # Graficar
            fig, axes = plt.subplots(len(TARGET_VARS), 1, figsize=(12, 15), sharex=True)
            
            for i, var in enumerate(TARGET_VARS):
                ax = axes[i]
                # Línea Pronóstico
                ax.plot(pred_df.index, pred_df[var], label='Pronóstico IA', color='blue', marker='o', linewidth=2)
                
                # Línea Real
                if not real_future.empty:
                    valid_idx = real_future.index.intersection(pred_df.index)
                    ax.plot(valid_idx, real_future.loc[valid_idx, var], label='Real', color='gray', linestyle='--', alpha=0.7, linewidth=2)
                
                ax.set_ylabel(VARIABLE_LABELS[var])
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Formateo seguro del eje Y para evitar el bug de "NAN" en los ticks
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

            plt.xlabel("Hora Local")
            plt.suptitle(f"Pronóstico - Inicio Input: {start_time}", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            break

        except Exception as e:
            print(f"❌ Error crítico: {e}")
            import traceback
            traceback.print_exc()