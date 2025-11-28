import os
import glob
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
INPUT_FOLDER = "Recortes-satelitales"
OUTPUT_JSON = "secuencias_entrenamiento.json"
INTERVALO_ESPERADO = 10  # minutos
TOLERANCIA = 2           # minutos extra aceptables antes de cortar

def obtener_fecha(nombre_archivo):
    # Extrae fecha de '2025-08-01-0021_band-13.png'
    try:
        parte_fecha = nombre_archivo.split('_')[0]
        return datetime.strptime(parte_fecha, "%Y-%m-%d-%H%M")
    except ValueError:
        return None

def generar_manifiesto():
    # 1. Obtener y ordenar archivos
    patron = os.path.join(INPUT_FOLDER, "*.png")
    archivos_completos = glob.glob(patron)
    archivos_completos.sort()
    
    if not archivos_completos:
        print("No hay imágenes para procesar.")
        return

    print(f"Procesando {len(archivos_completos)} imágenes...")

    # 2. Algoritmo de Segmentación
    secuencias = []
    bloque_actual = []
    
    # Inicializar con el primer archivo
    primer_nombre = os.path.basename(archivos_completos[0])
    bloque_actual.append(primer_nombre)
    t_previo = obtener_fecha(primer_nombre)

    for ruta in archivos_completos[1:]:
        nombre_actual = os.path.basename(ruta)
        t_actual = obtener_fecha(nombre_actual)
        
        if not t_actual or not t_previo:
            continue

        # Calcular diferencia
        delta = (t_actual - t_previo).total_seconds() / 60
        
        # ¿Es un salto aceptable? (10 min +/- tolerancia)
        if delta <= (INTERVALO_ESPERADO + TOLERANCIA):
            # Sí, es continuo -> Añadir al bloque actual
            bloque_actual.append(nombre_actual)
        else:
            # No, hay un corte -> Cerrar bloque anterior y empezar uno nuevo
            if len(bloque_actual) > 0:
                secuencias.append(bloque_actual)
            
            # Iniciar nuevo bloque
            bloque_actual = [nombre_actual]
        
        t_previo = t_actual

    # Guardar el último bloque si quedó pendiente
    if bloque_actual:
        secuencias.append(bloque_actual)

    # 3. Estadísticas y Filtrado para ML
    # Para entrenar, generalmente necesitas secuencias de al menos X pasos
    # (ej. 4 input + 4 output = 8 pasos). Filtremos bloques inútiles (< 8 imgs).
    MIN_LONGITUD = 8 
    secuencias_utiles = [s for s in secuencias if len(s) >= MIN_LONGITUD]

    print(f"\n--- RESULTADOS ---")
    print(f"Bloques totales detectados: {len(secuencias)}")
    print(f"Bloques ÚTILES para entrenamiento (>= {MIN_LONGITUD} imgs): {len(secuencias_utiles)}")
    
    # Guardar JSON
    datos_salida = {
        "metadata": {
            "generado": str(datetime.now()),
            "total_imagenes": sum(len(s) for s in secuencias_utiles),
            "num_secuencias": len(secuencias_utiles),
            "criterio_continuidad_min": INTERVALO_ESPERADO
        },
        "secuencias": secuencias_utiles
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(datos_salida, f, indent=2)
        
    print(f"Manifiesto guardado en: {os.path.abspath(OUTPUT_JSON)}")
    print("Usa este JSON en tu Data Loader para cargar solo secuencias validas.")

if __name__ == "__main__":
    generar_manifiesto()