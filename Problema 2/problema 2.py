# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

# Leer datos desde el archivo CSV con el delimitador correcto
df = pd.read_csv(r'C:\Users\cacaz\OneDrive\Escritorio\Examen BI\Problema 2\breast_wisconsin_1.csv', delimiter=';')

# Verificar si la columna 'fractal_dimension3' está presente
if 'fractal_dimension3' in df.columns:
    # Seleccionar la variable a predecir (y)
    y = df['fractal_dimension3']

    # Seleccionar variables predictoras (X)
    X = df.drop(['COD_identificacion_dni', 'fractal_dimension3'], axis=1, errors='ignore')

    # Particionar la base en 70 % de la data total para Train y 30% para Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializar el modelo predictivo (usaremos regresión lineal en este ejemplo)
    model = LinearRegression()

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Imprimir el MSE
    print(f'Error Cuadrático Medio (MSE): {mse}')

    # Guardar resultados en un archivo PDF
    with open('resultados_regresion.pdf', 'wb') as pdf_file:
        c = canvas.Canvas(pdf_file)
        c.drawString(100, 800, f'Error Cuadrático Medio (MSE): {mse}')
        c.save()

    # Mostrar el gráfico de predicciones vs. valores reales
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs. Valores Reales')
    plt.show()

else:
    print("La columna 'fractal_dimension3' no está presente en el DataFrame.")
    print("Nombres de las columnas en el DataFrame:", df.columns)
