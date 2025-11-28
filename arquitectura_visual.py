import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn_branch(input_shape, output_dim=128, dropout_rate=0.3):
    """
    Construye la rama visual del modelo (Custom CNN con TimeDistributed).
    
    Args:
        input_shape: Tupla (TimeSteps, Height, Width, Channels)
                     Ej: (6, 128, 128, 1)
        output_dim: Tamaño del vector de características de salida (Ej: 128)
        
    Returns:
        Un modelo Keras (o sub-modelo) que toma la secuencia de imágenes 
        y devuelve un vector de características.
    """
    
    # Entrada de la secuencia de video (imágenes de temperatura)
    visual_input = layers.Input(shape=input_shape, name="input_visual_temp")
    
    # --- BLOQUE 1: Extracción de Bordes y Gradientes Térmicos ---
    # TimeDistributed aplica la capa a cada frame temporal independientemente
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same'))(visual_input)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    # Salida: (Batch, Time, 64, 64, 32)

    # --- BLOQUE 2: Extracción de Formas (Nubes, Celdas) ---
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    # Salida: (Batch, Time, 32, 32, 64)
    
    # --- BLOQUE 3: Patrones de Mesoescala ---
    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    # Salida: (Batch, Time, 16, 16, 128)
    
    # --- BLOQUE 4: Estructura Global del Sistema ---
    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    # Salida: (Batch, Time, 8, 8, 256)

    # --- COLAPSO ESPACIAL (Feature Maps -> Vectores) ---
    # Promediamos el espacio (8x8) para obtener un vector por cada instante de tiempo
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    # Salida: (Batch, Time, 256) -> Tenemos 6 vectores de 256 características
    
    x = layers.Dropout(dropout_rate)(x)

    # --- COLAPSO TEMPORAL (Secuencia -> Resumen Único) ---
    # Opción A: Usar LSTM/GRU si el orden secuencial estricto es vital.
    # Opción B: Usar GlobalAveragePooling1D si queremos el "resumen" de la última hora.
    # Para fenómenos convectivos donde la "presencia" de la nube importa más que el segundo exacto:
    x = layers.GlobalAveragePooling1D()(x) 
    # Salida: (Batch, 256)
    
    # --- PROYECCIÓN FINAL (Cuello de botella) ---
    # Reducimos a la dimensión acordada para fusionar con la rama tabular
    visual_embedding = layers.Dense(output_dim, activation="relu", name="visual_embedding")(x)
    
    # Crear el modelo encapsulado
    model = models.Model(inputs=visual_input, outputs=visual_embedding, name="Rama_Visual_CNN")
    
    return model

if __name__ == "__main__":
    # Bloque de prueba para verificar que las dimensiones cuadran
    input_shape_test = (6, 128, 128, 1) # (Time, H, W, Channels)
    model = build_custom_cnn_branch(input_shape_test)
    model.summary()
    print("\n¡Arquitectura Visual compilada correctamente!")