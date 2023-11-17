# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

# Obtener datos
from ucimlrepo import fetch_ucirepo
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Seleccionar variables predictoras
selected_features = ['Alcohol', 'Alcalinity_of_ash', 'Nonflavanoid_phenols']
X_selected = X[selected_features]

# Particionar la base en 70 % de la data total para Train y 30% para Test
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Entrenar modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular accuracy y matriz de confusión
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir resultados
print(f'Accuracy: {accuracy}')
print('Matriz de Confusión:')
print(conf_matrix)

# Guardar resultados en un archivo PDF
with open('resultados.pdf', 'wb') as pdf_file:
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, f'Accuracy: {accuracy}')
    c.drawString(100, 780, 'Matriz de Confusión:')
    c.drawString(100, 760, f'{conf_matrix}')
    c.save()

# Mostrar la matriz de confusión como una imagen
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.show()
