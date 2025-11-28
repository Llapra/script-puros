import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

class HybridDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 csv_path, 
                 npy_folder, 
                 manifest_path, 
                 target_vars, 
                 feature_vars,
                 input_length_tab=48, 
                 input_length_vis=6, 
                 output_length=28, 
                 batch_size=32, 
                 shuffle=True):
        
        # 1. Llamada al constructor padre (Corrige el Warning de PyDataset)
        super().__init__()
        
        self.npy_folder = npy_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_length_tab = input_length_tab
        self.input_length_vis = input_length_vis
        self.output_length = output_length
        
        # 2. Cargar Datos Tabulares
        print(f"Cargando datos tabulares desde {csv_path}...")
        self.df = pd.read_csv(csv_path, index_col='TIMESTAMP', parse_dates=True)
        
        # Filtrar columnas y convertir a numpy (float32 explícito para evitar errores de tipo)
        self.feature_vars = feature_vars
        self.target_vars = target_vars
        self.data_features = self.df[feature_vars].values.astype(np.float32)
        self.data_targets = self.df[target_vars].values.astype(np.float32)
        
        # Crear mapa de tiempo
        self.time_to_index = {t: i for i, t in enumerate(self.df.index)}
        
        # 3. Cargar Manifiesto
        print(f"Cargando manifiesto de secuencias desde {manifest_path}...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.indices_validos = []
        self.preparar_indices(manifest['secuencias'])
        print(f"Generador inicializado con {len(self.indices_validos)} muestras.")

    def preparar_indices(self, secuencias):
        max_lookback = max(self.input_length_tab, self.input_length_vis)
        
        for seq_archivos in secuencias:
            fechas_seq = [self.parse_filename(f) for f in seq_archivos]
            
            for i in range(max_lookback, len(fechas_seq) - self.output_length):
                fecha_presente = fechas_seq[i]
                
                if fecha_presente in self.time_to_index:
                    idx_csv = self.time_to_index[fecha_presente]
                    
                    if idx_csv < self.input_length_tab:
                        continue

                    if (idx_csv + self.output_length) < len(self.df):
                        self.indices_validos.append({
                            'csv_idx': idx_csv,
                            'img_names': seq_archivos[i - self.input_length_vis + 1 : i + 1]
                        })

    def parse_filename(self, filename):
        try:
            str_date = filename.split('_')[0]
            return datetime.strptime(str_date, "%Y-%m-%d-%H%M")
        except Exception:
            return None

    def __len__(self):
        return int(np.floor(len(self.indices_validos) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices_validos[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_x_vis = []
        batch_x_tab = []
        batch_y = []
        
        for sample in indices:
            # --- VISUAL ---
            seq_imgs = []
            for fname in sample['img_names']:
                npy_name = fname.replace('.png', '.npy')
                npy_path = os.path.join(self.npy_folder, npy_name)
                
                try:
                    img_temp = np.load(npy_path).astype(np.float32)
                    # Normalización (-90 a 40 -> 0 a 1)
                    img_norm = (img_temp + 90.0) / 130.0
                    img_norm = np.clip(img_norm, 0.0, 1.0)
                except (FileNotFoundError, OSError):
                    img_norm = np.zeros((128, 128), dtype=np.float32)
                
                seq_imgs.append(img_norm)
            
            # (Time, H, W, 1)
            x_vis = np.array(seq_imgs)[..., np.newaxis]
            batch_x_vis.append(x_vis)
            
            # --- TABULAR ---
            idx = sample['csv_idx']
            x_tab = self.data_features[idx - self.input_length_tab + 1 : idx + 1]
            batch_x_tab.append(x_tab)
            
            # --- TARGET ---
            y = self.data_targets[idx + 1 : idx + 1 + self.output_length]
            batch_y.append(y)
            
        # CORRECCIÓN CRÍTICA AQUÍ:
        # Devolvemos una TUPLA para inputs: (x_visual, x_tabular)
        # Y un array para outputs: y
        # Convertimos todo explícitamente a tensores o arrays de float32
        inputs = (np.array(batch_x_vis, dtype=np.float32), np.array(batch_x_tab, dtype=np.float32))
        targets = np.array(batch_y, dtype=np.float32)
        
        return inputs, targets

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices_validos)