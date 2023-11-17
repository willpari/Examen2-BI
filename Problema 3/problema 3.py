# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

# Leer datos desde el archivo CSV
df = pd.read_csv(r'C:\Users\cacaz\OneDrive\Escritorio\Examen BI\Problema 3\aids_clinical_1-1.csv', delimiter=';')

# Seleccionar variables predictoras (X) y variable a predecir (y)
X = df.drop('str2', axis=1)  # Excluir columna a predecir
y = df['str2']

# Particionar la base en 80% de la data total para Train y 20% para Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Predecir en el conjunto de entrenamiento
y_train_pred = model.predict(X_train)

# Calcular el AUC en el conjunto de entrenamiento
auc_train = roc_auc_score(y_train, y_train_pred)

# Calcular la matriz de confusión en el conjunto de entrenamiento
conf_matrix_train = confusion_matrix(y_train, y_train_pred)

# Imprimir resultados en el conjunto de entrenamiento
print(f'AUC en Train: {auc_train}')
print('Matriz de Confusión en Train:')
print(conf_matrix_train)

# Guardar resultados en un archivo PDF
with open('resultados_arbol_decision.pdf', 'wb') as pdf_file:
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, f'AUC en Train: {auc_train}')
    c.drawString(100, 780, 'Matriz de Confusión en Train:')
    c.drawString(100, 760, f'{conf_matrix_train}')
    c.save()

# Mostrar la matriz de confusión como una imagen
plt.imshow(conf_matrix_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión en Train')
plt.colorbar()
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.show()
