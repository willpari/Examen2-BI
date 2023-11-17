# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

# Leer datos desde el archivo CSV
df = pd.read_csv(r'C:\Users\cacaz\OneDrive\Escritorio\Examen BI\Problema 4\glioma_grading_1.csv', delimiter=';')

# Seleccionar variables predictoras (X) y variable a predecir (y)
X = df.drop('Grade', axis=1)  # Excluir columna a predecir
y = df['Grade']

# Definir las columnas categóricas
categorical_cols = X.columns

# Particionar la base en 80% de la data total para Train y 20% para Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un preprocesador que aplica One-Hot Encoding solo a las columnas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Crear el modelo de RandomForest con el preprocesador
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entrenar el modelo con los datos de entrenamiento
model_rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = model_rf.predict(X_test)

# Calcular el AUC y Accuracy en el conjunto de prueba para Random Forest
auc_rf = roc_auc_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Resto del código ...

# Imprimir resultados
print(f'AUC para Random Forest: {auc_rf}')
print(f'Accuracy para Random Forest: {accuracy_rf}')

# Guardar resultados en un archivo PDF
with open('resultados_algoritmos.pdf', 'wb') as pdf_file:
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, f'AUC para Random Forest: {auc_rf}')
    c.drawString(100, 780, f'Accuracy para Random Forest: {accuracy_rf}')
    # Agregar el resto de la información según sea necesario
    c.save()

# Mostrar la matriz de confusión para Random Forest como una imagen
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.imshow(conf_matrix_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión para Random Forest')
plt.colorbar()
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.show()
