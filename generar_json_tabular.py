import pandas as pd
import json
import os

# --- CONFIGURACIÓN ---
CSV_FILE = "master_dataset_continuo.csv"
OUTPUT_JSON = "secuencias_entrenamiento_tabular.json"
MAX_GAP_MINUTES = 30  # Si hay un hueco mayor a 30 min, cortamos secuencia

def generar_manifiesto_tabular():
    if not os.path.exists(CSV_FILE):
        print(f"Error: No encuentro '{CSV_FILE}'")
        return

    print(f"Leyendo {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, index_col='TIMESTAMP', parse_dates=True)
    df = df.sort_index()
    
    # Calcular diferencias de tiempo entre filas consecutivas
    # diff() da la diferencia con la fila anterior
    time_diffs = df.index.to_series().diff()
    
    # Umbral de corte (ej. 35 minutos para dar tolerancia a los 30 min nominales)
    threshold = pd.Timedelta(minutes=MAX_GAP_MINUTES + 5)
    
    # Encontrar los puntos de quiebre (donde el salto es grande)
    break_points = time_diffs[time_diffs > threshold].index
    
    print(f"Detectados {len(break_points)} cortes en la continuidad temporal.")
    
    secuencias = []
    start_idx = df.index[0]
    
    for end_idx in break_points:
        # El fin de este bloque es el índice ANTERIOR al quiebre
        # Pero como slicing con fechas es inclusivo/exclusivo dependiendo...
        # Mejor tomamos el bloque usando slicing de tiempo
        
        # El punto de quiebre es el PRIMER dato del NUEVO bloque
        # Así que el bloque actual termina justo antes
        # Pandas slicing con timestamps es inclusivo en ambos extremos si son exactos
        
        # Truco: Usar indices posicionales para ser exactos
        # Buscamos la posición entera de start_idx y end_idx
        
        # Vamos a iterar de forma más segura usando bucle simple para generar metadatos
        pass 

    # RE-IMPLEMENTACIÓN MÁS SIMPLE Y ROBUSTA:
    # Iterar sobre los quiebres para definir (inicio, fin)
    
    lista_secuencias = []
    
    # El primer bloque empieza en el índice 0
    current_start = df.index[0]
    
    for break_time in break_points:
        # El bloque termina en el instante anterior al quiebre
        # Obtenemos el índice posicional del quiebre
        pos = df.index.get_loc(break_time)
        current_end = df.index[pos - 1]
        
        # Guardar metadatos (convertir a string ISO)
        lista_secuencias.append({
            "start": str(current_start),
            "end": str(current_end)
        })
        
        # El siguiente bloque empieza en el quiebre
        current_start = break_time
        
    # Agregar el último bloque (desde el último quiebre hasta el final)
    lista_secuencias.append({
        "start": str(current_start),
        "end": str(df.index[-1])
    })
    
    # Filtrar bloques muy cortos (menos de 48 pasos = 24 horas)
    # Si son muy cortos no sirven para entrenar (INPUT_LENGTH=48)
    MIN_LENGTH = 60 
    secuencias_utiles = []
    
    for seq in lista_secuencias:
        t_start = pd.to_datetime(seq['start'])
        t_end = pd.to_datetime(seq['end'])
        
        # Estimamos cantidad de pasos (aprox)
        duration = (t_end - t_start).total_seconds() / 60 # minutos
        steps = duration / 30 # asumiendo paso de 30 min
        
        if steps >= MIN_LENGTH:
            secuencias_utiles.append(seq)
            
    print(f"Secuencias válidas finales (Largas): {len(secuencias_utiles)}")
    
    output_data = {
        "metadata": {"source": CSV_FILE, "gap_threshold_min": MAX_GAP_MINUTES},
        "secuencias": secuencias_utiles
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"¡Listo! Archivo '{OUTPUT_JSON}' generado.")

if __name__ == "__main__":
    generar_manifiesto_tabular()