import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
from PIL import Image

# Definir la transformación de las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Redimensionar las imágenes
    transforms.ToTensor(),           # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización (por ejemplo, para imagen RGB)
])

# Definir un dataset personalizado para cargar imágenes desde un directorio sin subcarpetas
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Verificar si el directorio está vacío
        if not os.path.exists(root_dir):
            raise ValueError(f"El directorio {root_dir} no existe.")
        
        # Leer todas las imágenes del directorio
        for file_name in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file_name)
            if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Asegurarse de que sean imágenes
                self.images.append(file_path)
                self.labels.append(0)  # Asignamos una etiqueta dummy si no hay subcarpetas de clase
        
        # Imprimir cuántas imágenes se cargan
        print(f"Total de imágenes cargadas en {root_dir}: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Verificar que el directorio y las imágenes estén bien cargados
train_dataset = CustomDataset(root_dir='circuits', transform=transform)
test_dataset = CustomDataset(root_dir='circuits/test', transform=transform)

print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
print(f"Tamaño del dataset de prueba: {len(test_dataset)}")

# Crear los DataLoaders para el entrenamiento y la prueba
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir el modelo (puedes cambiar esto por cualquier red neuronal que estés utilizando)
model = models.resnet18(pretrained=True)  # Usamos ResNet-18 preentrenado por ejemplo
model.fc = nn.Linear(model.fc.in_features, 2)  # Cambiar la capa final para que tenga 2 clases

# Definir la función de pérdida y el optimizador
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función para guardar el modelo
def saveModel():
    path = "./circuitsModel.pth"
    torch.save(model.state_dict(), path)

# Función para probar la precisión en el conjunto de test
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # Ejecutar el modelo en el conjunto de prueba para predecir las etiquetas
            outputs = model(images.to(device))
            # La etiqueta con la mayor energía será nuestra predicción
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()
    
    # Verificar que no haya división por cero
    if total == 0:
        print("No hay imágenes en el conjunto de prueba.")
        return 0.0

    # Calcular la precisión sobre todas las imágenes de prueba
    accuracy = (100 * accuracy / total)
    return accuracy

# Función para entrenar el modelo
def train(num_epochs):
    best_accuracy = 0.0

    # Definir el dispositivo de ejecución
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("El modelo se ejecutará en el dispositivo", device)
    model.to(device)

    for epoch in range(num_epochs):  # Recorrer varias veces el dataset
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # Obtener las imágenes y etiquetas
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # Poner a cero los gradientes de los parámetros
            optimizer.zero_grad()
            # Realizar la predicción con el modelo
            outputs = model(images)
            # Calcular la pérdida basada en la salida del modelo y las etiquetas reales
            loss = loss_fn(outputs, labels)
            # Realizar la retropropagación de la pérdida
            loss.backward()
            # Ajustar los parámetros con los gradientes calculados
            optimizer.step()

            # Mostrar estadísticas cada 1000 imágenes
            running_loss += loss.item()  # Extraer el valor de la pérdida
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # Calcular y mostrar la precisión promedio de este epoch en el conjunto de prueba
        accuracy = testAccuracy()
        print(f'Para el epoch {epoch + 1}, la precisión en el conjunto de prueba es {accuracy:.2f} %')
        
        # Guardar el modelo si la precisión es la mejor hasta ahora
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# Llamar a la función de entrenamiento
train(num_epochs=10)
