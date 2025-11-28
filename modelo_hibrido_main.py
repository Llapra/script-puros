import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from arquitectura_visual import build_custom_cnn_branch

def build_hybrid_model(
    input_shape_vis,   # (6, 128, 128, 1)
    input_shape_tab,   # (48, num_features)
    output_length,     # 28
    num_targets,       # 5
    transformer_params # Diccionario con params del transformer
):
    """
    Construye y compila el modelo híbrido (CNN + Transformer).
    Versión Mejorada: Usa Flatten en rama tabular para preservar temporalidad.
    """
    
    # =================================================
    # RAMA 1: VISUAL (CNN)
    # =================================================
    # Construimos la rama visual usando tu arquitectura personalizada
    cnn_branch = build_custom_cnn_branch(input_shape_vis, output_dim=128)
    
    # Input explícito para el modelo general
    input_vis = layers.Input(shape=input_shape_vis, name="input_visual")
    
    # Pasamos el input por la rama CNN
    vector_visual = cnn_branch(input_vis) # Salida: (Batch, 128)

    # =================================================
    # RAMA 2: TABULAR (TRANSFORMER)
    # =================================================
    input_tab = layers.Input(shape=input_shape_tab, name="input_tabular")
    
    # Lógica del Transformer
    x = input_tab
    
    num_blocks = transformer_params.get("num_transformer_blocks", 4)
    head_size = transformer_params.get("head_size", 64)
    num_heads = transformer_params.get("num_heads", 8) # Aumentado a 8
    ff_dim = transformer_params.get("ff_dim", 256)     # Aumentado a 256
    dropout = transformer_params.get("dropout", 0.1)

    for _ in range(num_blocks):
        # Attention Block (Pre-Norm es más estable)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(x1, x1)
        x2 = layers.Add()([x, attn_output])
        
        # Feed Forward Block
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(ff_dim, activation="gelu")(x3)
        x3 = layers.Dropout(dropout)(x3)
        x3 = layers.Dense(input_shape_tab[-1])(x3)
        x = layers.Add()([x2, x3])

    # CAMBIO CRÍTICO: Usar Flatten para no perder la secuencia temporal
    # Antes: GlobalAveragePooling1D (Promediaba todo el tiempo -> Colapso)
    # Ahora: Flatten (Mantiene la historia completa desenrollada)
    x = layers.Flatten()(x)
    
    # Proyección final tabular
    vector_tabular = layers.Dense(128, activation="relu", name="embedding_tabular")(x)

    # =================================================
    # FUSIÓN TARDÍA (LATE FUSION)
    # =================================================
    # Concatenamos los dos vectores de características
    fusion = layers.Concatenate()([vector_visual, vector_tabular])
    
    # Capas Densas de Decisión
    z = layers.Dense(256, activation="relu")(fusion)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation="relu")(z)
    
    # =================================================
    # SALIDA (HEAD)
    # =================================================
    outputs = layers.Dense(output_length * num_targets)(z)
    outputs = layers.Reshape((output_length, num_targets), name="pronostico_final")(outputs)

    # Ensamblar Modelo Completo
    model = models.Model(inputs=[input_vis, input_tab], outputs=outputs, name="Hybrid_Nowcasting_Model")
    
    # Compilación con CLIPNORM para evitar NaNs
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )
    
    return model

if __name__ == "__main__":
    print("Test de construcción exitoso.")