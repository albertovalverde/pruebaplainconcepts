## Análisis y Procesamiento de Imágenes

Este flujo de trabajo implementa un análisis y procesamiento de imágenes, abordando varias etapas clave.

### 1. Análisis Exploratorio

- Se cargan imágenes desde un directorio, se verifica que todas estén correctamente cargadas, y se visualizan en una cuadrícula.
- Se generan histogramas de color para cada imagen para evaluar la distribución de los valores de los canales RGB.

###  2. Técnicas de Limpieza

- Se verifica si las imágenes están corruptas.
- Se utilizan medidas como el Structural Similarity Index (SSIM) para detectar imágenes duplicadas basándose en la similitud estructural.

###  3. Preprocesamiento y Aumento de Datos

- Se redimensionan y normalizan las imágenes.
- Se aplica un generador de aumento de datos para introducir variaciones en las imágenes mediante transformaciones aleatorias (rotación, desplazamiento, zoom, etc.).

###  4. Extracción de Características y Detección

- Se utiliza el modelo preentrenado VGG16 para la extracción de características.
- También se realiza un análisis de textura mediante el filtro Sobel, destacando bordes en las imágenes.

###  5. Métricas

- Se calcula el error cuadrático medio (MSE) entre las imágenes preprocesadas para comparar similitudes.

###  6. Despliegue

- Se ha creado un servidor Flask para recibir nuevas imágenes y añadirlas a la base de datos, además de realizar predicciones con el modelo VGG16.
- La ruta `/predict` permite que los usuarios suban una imagen para obtener una predicción.

Este flujo de trabajo cubre desde la carga de imágenes hasta la implementación de un modelo para análisis predictivo, con un enfoque en la limpieza y el preprocesamiento de los datos.
