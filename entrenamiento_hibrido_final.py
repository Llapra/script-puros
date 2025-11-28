import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import joblib
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- IMPORTACIONES ---
from generador_datos_hibrido import HybridDataGenerator
from modelo_hibrido_main import build_hybrid_model

# =================================================
# 1. CONFIGURACIÓN
# =================================================
RAW_CSV_FILE = "master_dataset_continuo.csv"
NPY_FOLDER = "Datos_Temperatura_NPY"
MANIFEST_FILE = "secuencias_entrenamiento.json"

SCALER_SAVE = "scaler_hibrido.gz"
MODEL_SAVE = "modelo_hibrido_final.keras"
TEMP_PROCESSED_CSV = "temp_processed_data.csv"

INPUT_LEN_TAB = 48
INPUT_LEN_VIS = 6
OUTPUT_LEN = 28

TARGET_VARS = ['NETRAD', 'RH_1_1_1', 'TA_1_1_1', 'T_DP_1_1_1', 'TKE']
FEATURE_VARS = [
    'TA_1_1_1', 'WS', 'TKE', 'H2O_density', 'Uz', 'RH_1_1_1', 'NETRAD', 
    'SW_IN', 'LW_IN','SW_OUT', 'LW_OUT', 'PA', 'WD_SONIC', 'USTAR',
    'T_DP_1_1_1', 'T_SONIC_SIGMA', 'Uz_SIGMA', 'e_amb', 'G_plate_1_1_1', 'H_QC'
]

TRANSFORMER_HPARAMS = {
    "num_transformer_blocks": 4,
    "num_heads": 8,
    "head_size": 64,
    "ff_dim": 256,
    "dropout": 0.1
}

# =================================================
# 2. PREPROCESAMIENTO
# =================================================
def prepare_tabular_data():
    print("--- PREPARANDO DATOS TABULARES ---")
    
    # Carga robusta
    df = pd.read_csv(RAW_CSV_FILE, index_col='TIMESTAMP', parse_dates=True, low_memory=False)
    
    # Limpieza de tipos y NaNs
    for col in FEATURE_VARS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Cíclicos
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)
    
    features_finales = FEATURE_VARS + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    # Filtro fecha
    TRAIN_LIMIT_DATE = "2025-10-01 23:59:59"
    df_train = df.loc[:TRAIN_LIMIT_DATE]
    
    # Escalar
    print("Ajustando Scaler...")
    scaler = MinMaxScaler()
    scaler.fit(df_train[features_finales])
    joblib.dump(scaler, SCALER_SAVE)
    
    # Transformar todo y guardar CSV temporal
    df_scaled = pd.DataFrame(scaler.transform(df[features_finales]), 
                             index=df.index, 
                             columns=features_finales)
    
    # BLINDAJE: Limpiar NaNs post-escalado
    df_scaled = df_scaled.fillna(0)
    
    df_scaled.to_csv(TEMP_PROCESSED_CSV)
    return features_finales

# =================================================
# 3. EJECUCIÓN
# =================================================
if __name__ == "__main__":
    # A. Preparar CSV
    final_feature_cols = prepare_tabular_data()
    
    # B. Configurar Manifiesto (Con recorte de fechas)
    with open(MANIFEST_FILE, 'r') as f:
        full_manifest = json.load(f)
    
    json_sequences = full_manifest['secuencias']
    train_val_sequences = []
    
    TRAIN_LIMIT_DATE = "2025-10-01 23:59:59"
    cutoff_dt = pd.to_datetime(TRAIN_LIMIT_DATE)
    
    for seq in json_sequences:
        seq_start = pd.to_datetime(seq[0].split('_')[0]) # Parsear nombre archivo img
        # Estimación rápida del fin basada en largo
        # Mejor: Asumimos que si empieza antes del corte, usamos lo que se pueda
        # Pero 'generador_datos_hibrido' hace validación interna fila a fila.
        # Para simplificar, pasamos todas las secuencias y dejamos que el generador decida,
        # O filtramos grueso aquí.
        
        # Como el JSON tiene nombres de archivo, y el generador chequea contra el CSV,
        # si el CSV está cortado o si limitamos el generador, funciona.
        # Pero aquí vamos a pasar todo al generador y dejar que él filtre si falta data en CSV.
        train_val_sequences.append(seq)

    # División Train/Val
    n_train = int(len(train_val_sequences) * 0.85)
    # Si es una sola secuencia larga, duplicamos para que el generador haga slicing interno
    if n_train == 0:
        train_seqs = train_val_sequences
        val_seqs = train_val_sequences
    else:
        train_seqs = train_val_sequences[:n_train]
        val_seqs = train_val_sequences[n_train:]
    
    # Guardar JSONs temporales
    with open("temp_train.json", "w") as f: json.dump({'secuencias': train_seqs}, f)
    with open("temp_val.json", "w") as f: json.dump({'secuencias': val_seqs}, f)
    
    # C. Generadores
    print("Iniciando Generadores...")
    train_gen = HybridDataGenerator(
        csv_path=TEMP_PROCESSED_CSV,
        npy_folder=NPY_FOLDER,
        manifest_path="temp_train.json",
        target_vars=TARGET_VARS,
        feature_vars=final_feature_cols,
        input_length_tab=INPUT_LEN_TAB,
        input_length_vis=INPUT_LEN_VIS,
        output_length=OUTPUT_LEN,
        batch_size=32, # Ajusta a 64/128 si tu PC aguanta
        shuffle=True
    )
    
    val_gen = HybridDataGenerator(
        csv_path=TEMP_PROCESSED_CSV,
        npy_folder=NPY_FOLDER,
        manifest_path="temp_val.json",
        target_vars=TARGET_VARS,
        feature_vars=final_feature_cols,
        input_length_tab=INPUT_LEN_TAB,
        input_length_vis=INPUT_LEN_VIS,
        output_length=OUTPUT_LEN,
        batch_size=32,
        shuffle=False
    )
    
    # D. Modelo
    print("Construyendo Modelo Híbrido...")
    shape_vis = (INPUT_LEN_VIS, 128, 128, 1) 
    shape_tab = (INPUT_LEN_TAB, len(final_feature_cols))
    
    model = build_hybrid_model(
        input_shape_vis=shape_vis,
        input_shape_tab=shape_tab,
        output_length=OUTPUT_LEN,
        num_targets=len(TARGET_VARS),
        transformer_params=TRANSFORMER_HPARAMS
    )
    
    # E. Entrenar
    checkpoint = ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[checkpoint, early_stop]
    )
    
    print("¡Entrenamiento Híbrido Finalizado!")
    
    # Limpieza
    if os.path.exists("temp_train.json"): os.remove("temp_train.json")
    if os.path.exists("temp_val.json"): os.remove("temp_val.json")