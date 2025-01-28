import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

app = Flask(__name__)

# Ruta donde se guardarán las imágenes
UPLOAD_FOLDER = 'circuits'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cargar el modelo VGG16 (o el que estés usando)
model = VGG16(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    # Guardar la imagen en el directorio /circuits
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # Procesar la imagen
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Redimensionar a 224x224, tamaño esperado por VGG16
    image = np.expand_dims(image, axis=0)  # Expande las dimensiones de la imagen para que tenga forma (1, 224, 224, 3)
    image = preprocess_input(image)  # Preprocesamiento específico para VGG16
    
    # Realizar predicción
    preds = model.predict(image)
    
    # Retornar los resultados como JSON
    return jsonify(preds.tolist())

if __name__ == '__main__':
    app.run(debug=True)
