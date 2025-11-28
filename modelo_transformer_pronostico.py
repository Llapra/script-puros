import pandas as pd
import os
import numpy as np
import json
import sys
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# =============================================================================
# BLOQUE 0: CONFIGURACIÓN CENTRAL
# =============================================================================
UNIFIED_DATA_FILE = "master_dataset_continuo.csv"
JSON_MANIFEST_FILE = "secuencias_entrenamiento_tabular.json"
MODEL_SAVE_NAME = "nowcasting_transformer_clouds.keras"
SCALER_SAVE_NAME = "data_scaler_clouds.gz"

# --- FECHA LÍMITE DE ENTRENAMIENTO ---
# El modelo solo verá datos hasta este momento. 
# Todo lo posterior se guarda para tu prueba visual (Oct 2 - Oct 7).
TRAIN_LIMIT_DATE = "2025-10-01 23:59:59"

# --- Variables ---
TARGET_VARS = ['NETRAD', 'RH_1_1_1', 'TA_1_1_1', 'T_DP_1_1_1', 'TKE']
FEATURE_VARS = [
    'TA_1_1_1', 'WS', 'TKE', 'H2O_density', 'Uz', 'RH_1_1_1', 'NETRAD', 
    'SW_IN', 'LW_IN','SW_OUT', 'LW_OUT', 'PA', 'WD_SONIC', 'USTAR',
    'T_DP_1_1_1', 'T_SONIC_SIGMA', 'Uz_SIGMA', 'e_amb', 'G_plate_1_1_1', 'H_QC'
]

INPUT_LENGTH = 48   # 24 horas entrada
OUTPUT_LENGTH = 12  # 6 horas pronóstico

HPARAMS = {
    "num_transformer_blocks": 4,
    "num_heads": 8,          # Aumentado de 6 a 8
    "head_size": 64,
    "ff_dim": 256,           # Aumentado para mayor capacidad
    "mlp_units": [256, 128], # MLP más profunda al final
    "dropout": 0.1,
    "mlp_dropout": 0.1
}

# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================
def load_data_and_manifest():
    if not os.path.exists(UNIFIED_DATA_FILE) or not os.path.exists(JSON_MANIFEST_FILE):
        raise FileNotFoundError("Faltan archivos. Asegúrate de tener el CSV continuo y el JSON.")
    
    # Cargar ignorando errores de tipo mixto
    df = pd.read_csv(UNIFIED_DATA_FILE, index_col='TIMESTAMP', parse_dates=True, low_memory=False)
    
    # LIMPIEZA DE EMERGENCIA
    for col in FEATURE_VARS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    with open(JSON_MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)
    return df, manifest['secuencias']

def create_cyclical_features(df):
    df_out = df.copy()
    df_out['hour_sin'] = np.sin(2 * np.pi * df_out.index.hour / 24.0)
    df_out['hour_cos'] = np.cos(2 * np.pi * df_out.index.hour / 24.0)
    df_out['day_sin'] = np.sin(2 * np.pi * df_out.index.dayofyear / 365.0)
    df_out['day_cos'] = np.cos(2 * np.pi * df_out.index.dayofyear / 365.0)
    return df_out

def generate_dataset_from_json(df, sequences_list, input_len, output_len, target_indices, scaler):
    X_list, y_list = [], []
    for seq_info in sequences_list:
        start_time = pd.to_datetime(seq_info['start'])
        end_time = pd.to_datetime(seq_info['end'])
        
        try:
            chunk = df.loc[start_time : end_time].copy()
            if chunk.empty: continue
            
            chunk = create_cyclical_features(chunk[FEATURE_VARS]) 
            
            chunk_scaled_vals = scaler.transform(chunk)
            
            # BLINDAJE CONTRA NANs
            chunk_scaled_vals = np.nan_to_num(chunk_scaled_vals, nan=0.0, posinf=1.0, neginf=0.0)
            
            if len(chunk_scaled_vals) < input_len + output_len:
                continue

            for i in range(len(chunk_scaled_vals) - input_len - output_len + 1):
                input_seq = chunk_scaled_vals[i : i + input_len, :]
                output_seq = chunk_scaled_vals[i + input_len : i + input_len + output_len, target_indices]
                X_list.append(input_seq)
                y_list.append(output_seq)
        except Exception as e:
            print(f"Saltando secuencia corrupta ({start_time}): {e}")
            continue
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# =============================================================================
# MODELO
# =============================================================================
def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization before Attention (Pre-Norm) suele ser más estable
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs 
    
    # Feed Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="gelu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout, output_length, num_target_vars):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Bloques Transformer
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_layer(x, head_size, num_heads, ff_dim, dropout)
    
    # CAMBIO CRÍTICO: Usar Flatten en lugar de GlobalAveragePooling1D
    # Esto preserva la información de "cuándo" ocurrió cada evento en la secuencia de entrada
    x = Flatten()(x)
    
    # Cabezal MLP (Decoder simple)
    for dim in mlp_units:
        x = Dense(dim, activation="gelu")(x)
        x = Dropout(mlp_dropout)(x)
        
    # Salida
    outputs = Dense(output_length * num_target_vars)(x)
    outputs = Reshape((output_length, num_target_vars))(outputs)
    
    model = Model(inputs, outputs)
    
    # Optimización robusta con clipnorm=1.0 para evitar NaNs
    model.compile(
        loss="mse", 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    )
    return model

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("--- RE-ENTRENANDO MODELO TABULAR (CORREGIDO) ---")
    
    try:
        full_df, json_sequences = load_data_and_manifest()
        
        # 1. Ajustar Escalador
        print(f"Filtrando datos hasta: {TRAIN_LIMIT_DATE}")
        train_subset_df = full_df.loc[:TRAIN_LIMIT_DATE]
        
        df_features_train = create_cyclical_features(train_subset_df[FEATURE_VARS])
        df_features_train = df_features_train.fillna(0)
        
        scaler = MinMaxScaler()
        scaler.fit(df_features_train)
        print("Escalador ajustado.")

        # 2. Filtrar Secuencias (CORRECCIÓN LÓGICA FECHA)
        train_val_sequences = []
        cutoff_dt = pd.to_datetime(TRAIN_LIMIT_DATE)
        
        ignored_count = 0
        modified_count = 0
        
        for seq in json_sequences:
            seq_start = pd.to_datetime(seq['start'])
            seq_end = pd.to_datetime(seq['end'])
            
            if seq_start > cutoff_dt:
                ignored_count += 1
                continue
            
            if seq_end <= cutoff_dt:
                train_val_sequences.append(seq)
            else:
                new_seq = seq.copy()
                new_seq['end'] = str(cutoff_dt)
                train_val_sequences.append(new_seq)
                modified_count += 1
                
        print(f"Secuencias procesadas: {len(json_sequences)}")
        print(f"  -> Recortadas (usadas parcialmente): {modified_count}")
        print(f"  -> Ignoradas (futuro): {ignored_count}")
        print(f"  -> Total para entrenar: {len(train_val_sequences)}")

        if not train_val_sequences:
            print("Error: No hay secuencias válidas.")
            sys.exit()

        # 3. División Train/Val
        n_train = int(len(train_val_sequences) * 0.85)
        
        if n_train == 0:
            train_seqs = train_val_sequences
            val_seqs = train_val_sequences 
            print("ALERTA: Pocas secuencias continuas. Usando la misma data para Train/Val (Cuidado con overfitting)")
        else:
            train_seqs = train_val_sequences[:n_train]
            val_seqs = train_val_sequences[n_train:]
        
        print(f"Listas de secuencias: {len(train_seqs)} Train | {len(val_seqs)} Val")

        # 4. Generar Datos
        cols_finales = df_features_train.columns.tolist()
        target_indices = [cols_finales.index(col) for col in TARGET_VARS]
        
        print("Generando matrices...")
        X_train, y_train = generate_dataset_from_json(full_df, train_seqs, INPUT_LENGTH, OUTPUT_LENGTH, target_indices, scaler)
        
        if n_train == 0:
            split_idx = int(len(X_train) * 0.85)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        else:
            X_val, y_val = generate_dataset_from_json(full_df, val_seqs, INPUT_LENGTH, OUTPUT_LENGTH, target_indices, scaler)
        
        # Limpieza final de NaNs
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("¡ALERTA! Limpiando NaNs residuales...")
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_val = np.nan_to_num(X_val)
            y_val = np.nan_to_num(y_val)

        print(f"Tensores Finales: Train {X_train.shape}, Val {X_val.shape}")

        # 5. Entrenar
        model = build_transformer_model(
            input_shape=X_train.shape[1:],
            output_length=OUTPUT_LENGTH,
            num_target_vars=len(TARGET_VARS),
            **HPARAMS
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 6. Guardar
        model.save(MODEL_SAVE_NAME)
        joblib.dump(scaler, SCALER_SAVE_NAME)
        print("¡Modelo Tabular Blindado Guardado!")
        
    except Exception as e:
        print(f"ERROR CRÍTICO EN EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()