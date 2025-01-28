import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import mean_squared_error as mse

# Ruta al directorio de imágenes
image_folder = 'circuits/'

# Obtener una lista de archivos de imagen válidos en el directorio
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Cargar imágenes
images = [cv2.imread(img_path) for img_path in image_files]
# Resize and normalize
target_size = (224, 224)
resized_images = [cv2.resize(img, target_size) for img in images]
normalized_images = [img / 255.0 for img in resized_images]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images = []
for img in normalized_images:
    img = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img, batch_size=1)
    augmented_images.extend([next(aug_iter)[0] for _ in range(5)])  # 5 augmented images per original image

# Calcular MSE entre imágenes normalizadas
for i in range(len(normalized_images)):
    for j in range(i + 1, len(normalized_images)):
        mse_value = mse(normalized_images[i], normalized_images[j])
        print(f'MSE between image {i+1} and image {j+1}: {mse_value}')
