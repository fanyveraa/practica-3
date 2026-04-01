import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Cargar imagen a color
img_path = 'bob.jpg'
imagen_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

if imagen_bgr is None:
    raise ValueError("No se pudo cargar la imagen. Verificar ruta o nombre del archivo.")

imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)  # Conversión BGR a RGB

# 2. Visualizar imagen
plt.figure(figsize=(6,6))
plt.imshow(imagen_rgb)
plt.title("Imagen en formato RGB")
plt.axis('off')
plt.show()

# Valores RGB de todos los píxeles
alto, ancho, canales = imagen_rgb.shape
pixeles = imagen_rgb.reshape((-1, 3))  # matriz N x 3

print("Dimensiones de la imagen:", imagen_rgb.shape)
print("Número total de píxeles:", pixeles.shape[0])
print("Primeros 5 valores RGB:\n", pixeles[:5])

# 3. Gráfico 3D del espacio RGB

# Evitar error si hay menos de 5000 píxeles
num_muestras = min(5000, pixeles.shape[0])

muestra = pixeles[np.random.choice(pixeles.shape[0], size=num_muestras, replace=False)]

R = muestra[:, 0]
G = muestra[:, 1]
B = muestra[:, 2]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(R, G, B, c=muestra/255.0, marker='o', s=5)

ax.set_xlabel('Rojo (R)')
ax.set_ylabel('Verde (G)')
ax.set_zlabel('Azul (B)')
ax.set_title('Distribución de píxeles en el espacio RGB')

plt.show()