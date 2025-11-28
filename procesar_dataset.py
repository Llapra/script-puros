import os
import glob
from PIL import Image

# --- CONFIGURACIÓN ---
ESTACION_X = 745
ESTACION_Y = 313
CROP_SIZE = 512
HALF_SIZE = CROP_SIZE // 2

# Caja de Recorte
x_start = ESTACION_X - HALF_SIZE
y_start = ESTACION_Y - HALF_SIZE
x_end   = ESTACION_X + HALF_SIZE
y_end   = ESTACION_Y + HALF_SIZE
crop_box = (x_start, y_start, x_end, y_end)

INPUT_FOLDER = "todas_las_imagenes" 
INPUT_PATTERN = os.path.join(INPUT_FOLDER, "*.png")
OUTPUT_FOLDER = "Recortes-satelitales"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def formatear_nombre(ruta_completa):
    try:
        nombre_archivo = os.path.basename(ruta_completa)
        partes = nombre_archivo.split('---')
        fecha_sucia = partes[-1] 
        fecha_str = fecha_sucia.replace('.png', '')
        
        if len(fecha_str) != 14 or not fecha_str.isdigit():
            return None 
            
        year = fecha_str[0:4]
        month = fecha_str[4:6]
        day = fecha_str[6:8]
        hour = fecha_str[8:10]
        minute = fecha_str[10:12]
        
        return f"{year}-{month}-{day}-{hour}{minute}_band-13.png"
    except:
        return None

def procesar():
    archivos = glob.glob(INPUT_PATTERN)
    print(f"Procesando {len(archivos)} imágenes desde '{INPUT_FOLDER}'...")
    
    procesados = 0
    for archivo_ruta in archivos:
        nuevo_nombre = formatear_nombre(archivo_ruta)
        if nuevo_nombre is None: continue
            
        try:
            with Image.open(archivo_ruta) as img:
                if img.mode == 'RGBA': img = img.convert('RGB')
                recorte = img.crop(crop_box)
                ruta_salida = os.path.join(OUTPUT_FOLDER, nuevo_nombre)
                recorte.save(ruta_salida)
                procesados += 1
                if procesados % 500 == 0: print(f"  -> {procesados} listos.")
        except Exception as e:
            print(f"Error en {archivo_ruta}: {e}")

    print(f"¡Listo! {procesados} imágenes en '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    procesar()