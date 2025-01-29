# Prueba Plain Concepts

## Análisis y Procesamiento de Imágenes `Analisis_Procesamiento_Imagenes.ipynb`

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

- Se ha creado un servidor Flask para recibir nuevas imágenes y añadirlas al directorio, además de realizar predicciones con el modelo VGG16.
- La ruta `/predict` permite que los usuarios suban una imagen para obtener una predicción.

Este flujo de trabajo cubre desde la carga de imágenes hasta la implementación de un modelo para análisis predictivo, con un enfoque en la limpieza y el preprocesamiento de los datos.


## torchtraining.py

El proyecto incluye un nuevo código denominado `torchtraining.py`:

- `circuits/`: Contiene las imágenes de circuitos para entrenamiento.
- `circuits/test/`: Contiene las imágenes de circuitos para pruebas.

Esta separación asegura que el modelo sea entrenado con un conjunto de datos y evaluado con otro conjunto independiente, lo que permite comprobar su capacidad de generalización sin sobreajuste.

## Descripción de `torchtraining.py`

El archivo `torchtraining.py` se encarga del preprocesamiento de las imágenes, entrenamiento del modelo y evaluación del rendimiento. Los pasos clave son:

- **Preprocesamiento de imágenes**: Se redimensionan las imágenes a 224x224 píxeles y se normalizan para adaptarlas al modelo preentrenado.
  
- **Carga del conjunto de datos**: Se utiliza `ImageFolder` de PyTorch para cargar las imágenes de los directorios `circuits` y `circuits/test`, que contienen las imágenes de circuitos para entrenar y probar el modelo, respectivamente.

- **Creación de DataLoaders**: Utiliza `DataLoader` para manejar los lotes de imágenes tanto para entrenamiento como para pruebas.

- **Entrenamiento del modelo**: El modelo se entrena utilizando un conjunto de imágenes de entrenamiento, optimizando los parámetros con un optimizador y calculando la pérdida.

- **Evaluación y precisión**: Se calcula la precisión del modelo en el conjunto de pruebas al comparar las predicciones con las etiquetas reales.

- **Guardado del modelo**: El modelo se guarda cuando se alcanza la mejor precisión en el conjunto de pruebas.

Este código está diseñado para predecir las clases de circuitos, como resistores, transistores y capacitores, mediante un modelo de red neuronal.


## Predicción con el modelo entrenado `predict.py`

Una vez que hayas entrenado el modelo y guardado los pesos en el archivo `circuitsModel.pth`, puedes usar el script `predict.py` para realizar predicciones sobre nuevas imágenes.

### Instrucciones:
1. Asegúrate de que el archivo `circuitsModel.pth`.
2. Coloca la imagen que deseas clasificar en el directorio de tu elección y actualiza la variable `image_path` en `predict.py` con la ruta de la imagen.
3. Ejecuta `predict.py` para obtener la clase predicha para la imagen.

El script cargará el modelo entrenado y realizará una predicción, mostrando la clase estimada (por ejemplo, resistor, transistor, capacitor, etc.) para la imagen proporcionada.

