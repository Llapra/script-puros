DEEP LEARNING MULTIMODAL PARA NOWCASTING METEOROLÓGICO (HÍBRIDO CNN-TRANSFORMER)

Este repositorio contiene la implementación oficial del sistema de pronóstico a corto plazo (Nowcasting) desarrollado para la estimación de variables meteorológicas críticas. El sistema utiliza una arquitectura de red neuronal híbrida que integra visión computacional y procesamiento de series temporales mediante una estrategia de fusión tardía (Late Fusion).

================================================================================

DESCRIPCIÓN DEL SISTEMA

El modelo combina dos fuentes de datos heterogéneas para generar predicciones con un horizonte de 14 horas:

Componente Espacial (Rama Visual):
Procesa secuencias de imágenes satelitales (GOES-19, Banda 13) utilizando una Red Neuronal Convolucional (CNN) distribuida en el tiempo para extraer características morfológicas y dinámicas de la cobertura nubosa.

Componente Temporal (Rama Tabular):
Procesa series de tiempo de una estación meteorológica local (SEETRUE) utilizando una arquitectura Transformer (Encoder) para modelar dependencias temporales complejas y ciclos diurnos.

================================================================================

ESTRUCTURA DEL REPOSITORIO

Los scripts están numerados secuencialmente para facilitar el flujo de trabajo.

[01] PREPROCESAMIENTO DE DATOS ==================================================

01_preprocess_images_crop.py: Realiza el recorte espacial de las imágenes satelitales crudas centrado en las coordenadas de la estación.

01_preprocess_images_to_npy.py: Convierte las imágenes procesadas a tensores numéricos (Numpy), aplicando calibración física de temperatura y normalización.

01_preprocess_visual_index.py: Genera el índice JSON de secuencias de imágenes válidas y continuas.

01_preprocess_tabular_index.py: Genera el índice JSON para las secuencias de datos tabulares.

[02] ARQUITECTURA DEL MODELO ==================================================

02_model_architecture_cnn.py: Define la estructura de la rama CNN (TimeDistributed) para extracción de características visuales.

02_model_architecture_hybrid.py: Define la clase principal del modelo híbrido, integrando la rama visual, el bloque Transformer tabular y el módulo de fusión.

02_data_generator_hybrid.py: Implementa el generador de datos (Keras Sequence) que alimenta el modelo durante el entrenamiento, sincronizando temporalmente ambas fuentes de datos.

[03] ENTRENAMIENTO ==================================================

03_train_tabular_transformer.py: Script para entrenar exclusivamente la rama tabular (Transformer) como línea base de rendimiento.

03_train_hybrid_system.py: Script principal para el entrenamiento del sistema híbrido completo (End-to-End).

[04] INFERENCIA Y APLICACIÓN ==================================================

04_inference_hybrid_production.py: Sistema de inferencia listo para producción. Carga el modelo entrenado y genera pronósticos operativos a partir de una fecha dada.

04_inference_hybrid_metrics.py: Ejecuta la inferencia sobre un conjunto de prueba y calcula métricas de error (RMSE, MAE) comparando con datos reales.

04_inference_hybrid_demo.py: Versión demostrativa que incluye visualización de la entrada satelital junto a las curvas de pronóstico.

[UTILS] HERRAMIENTAS AUXILIARES ==================================================

utils_visualization_activations.py: Visualizador básico de mapas de activación internos de la CNN.

utils_visualization_activations_context.py: Visualizador avanzado que incluye interpretación física de la entrada (tipos de nubes) junto a las activaciones de la red.

utils_create_temperature_lut.py: Genera la tabla de búsqueda (LUT) para la conversión de color RGB a temperatura física.

================================================================================

REQUISITOS DE INSTALACIÓN

El entorno requiere Python 3.8 o superior y las siguientes librerías científicas:
pip install tensorflow pandas numpy matplotlib scikit-learn pillow scipy

================================================================================

INSTRUCCIONES DE USO

Preparación del Entorno:
Asegúrese de que las carpetas de datos ("Recortes-satelitales", "master_dataset_continuo.csv") estén presentes en la raíz.

Generación de Índices:
Ejecute los scripts "01_" para preparar los índices de entrenamiento.

Entrenamiento:
Ejecute "03_train_hybrid_system.py". Los pesos del modelo se guardarán como "modelo_hibrido_final.keras".

Predicción:
Utilice "04_inference_hybrid_production.py" para generar pronósticos. El sistema solicitará una fecha en formato "YYYY-MM-DD HH:MM".

================================================================================

METODOLOGÍA

El sistema utiliza una ventana de observación histórica de 6 imágenes satelitales (1 hora) y 48 registros de estación (24 horas) para predecir las siguientes variables objetivo:

Radiación Neta (NETRAD)

Humedad Relativa (RH)

Temperatura del Aire (TA)

Punto de Rocío (T_DP)

Energía Cinética Turbulenta (TKE)

================================================================================

AUTORES

Creado para Tecnicas Experimentales, curso impartido por Profesor Dario Pérez.

Desarrollado como parte de investigación para dicho curso para Nowcasting local con datos obtenidos desde SEETRUE e imágenes satelitales de CIRA GOES-19 band-13.

Proyecto realizado por:

Eduardo Llantén     |   eduardo.llanten.p@mail.pucv.cl 
Juan Pablo Araya    |   juan.araya.v@mail.pucv.cl
Simón González      |   matias.gonzalez.l01@mail.pucv.cl
Bruno Tapia         |   bruno.tapia.c@mail.pucv.cl

