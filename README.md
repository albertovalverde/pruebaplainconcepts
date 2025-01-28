# Image Analysis and Processing Project

Este proyecto implementa un flujo de trabajo completo para el análisis de imágenes, incluyendo exploración, limpieza, preprocesamiento, extracción de características, y aumento de datos. También se utiliza para la detección de imágenes duplicadas y el cálculo de la similitud entre imágenes utilizando métricas como el Error Cuadrático Medio (MSE) y el Índice de Similitud Estructural (SSIM).

## Requerimientos

Este proyecto requiere las siguientes bibliotecas de Python, que pueden ser instaladas con el archivo `requirements.txt`:

- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-image

Puedes instalar todas las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt

Image-Analysis-Project/
│
├── circuits/                 # Directorio donde se guardan las imágenes procesadas
├── requirements.txt          # Archivo con las dependencias necesarias
├── image_analysis.py         # Script principal para el análisis de imágenes
└── README.md                 # Este archivo

Flujo de Trabajo
1. Exploración de Datos
Se cargan las imágenes desde el directorio /circuits y se realizan análisis exploratorios como el cálculo de histogramas de colores.

2. Técnicas de Limpieza
Se verifican las imágenes corruptas y se detectan duplicados utilizando el Índice de Similitud Estructural (SSIM).

3. Preprocesamiento y Aumento de Datos
Las imágenes se redimensionan y normalizan, y luego se aplican técnicas de aumento de datos (rotaciones, traslaciones, etc.) para generar más variaciones de las imágenes originales.

4. Extracción de Características y Detección
Se extraen características de las imágenes utilizando técnicas de procesamiento de imágenes como la detección de bordes y el cálculo de similitudes entre imágenes usando MSE y SSIM.

5. Resultados y Visualización
Se calcula la similitud entre las imágenes y se muestran los resultados a través de métricas y visualizaciones como gráficos de los histogramas de colores.
