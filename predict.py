import torch
from torchvision import models, transforms
from PIL import Image

# Cargar el modelo pre-entrenado ResNet-18
model = models.resnet18(pretrained=True)  # Usamos pretrained=True para cargar los pesos de ImageNet

# Cambiar la última capa para que tenga 4 salidas (ajusta el número de clases)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=4)  # Cambia 4 a tu número de clases

# Cargar los pesos entrenados, ignorando la última capa que no coincide
checkpoint = torch.load('circuitsModel.pth')

# Cargar solo los pesos de las capas que no sean la última
state_dict = checkpoint
state_dict.pop('fc.weight', None)  # Eliminar la última capa 'weight' de los pesos
state_dict.pop('fc.bias', None)    # Eliminar la última capa 'bias' de los pesos

# Cargar los pesos restantes en el modelo
model.load_state_dict(state_dict, strict=False)  # Esto cargará solo las capas que coinciden

model.eval()  # Establecer el modelo en modo evaluación (no entrenamiento)

# Definir las transformaciones para preprocesar la imagen (como en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar la imagen
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Cargar la imagen desde un archivo
image_path = 'imagetopredict.jpg'  
image = Image.open(image_path).convert('RGB')  # Convertir la imagen a RGB si es necesario

# Aplicar las transformaciones
input_image = transform(image).unsqueeze(0)  # Añadir una dimensión para el batch (1 imagen)

# Realizar la predicción
with torch.no_grad():  # Deshabilitar gradientes, ya que solo estamos haciendo predicción
    output = model(input_image)  # Pasar la imagen al modelo

# Obtener la clase predicha
_, predicted_class = torch.max(output, 1)  # Obtener la clase con mayor probabilidad

# Lista de clases (ajusta según las clases de tu dataset)
class_names = ['resistor', 'transistor', 'capacitor', 'other_class']  # Cambia estos nombres según tu dataset

# Obtener el nombre de la clase predicha
predicted_class_name = class_names[predicted_class.item()]
print('Predicted class:', predicted_class_name)
