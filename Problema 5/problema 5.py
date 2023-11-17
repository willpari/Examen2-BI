# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

# Leer datos desde el archivo CSV
df = pd.read_csv(r'C:\Users\cacaz\OneDrive\Escritorio\Examen BI\Problema 5\aids_clinical_1-2.csv', delimiter=';')

# Seleccionar las variables de interés
variables_of_interest = ['preanti', 'wtkg']
df_subset = df[variables_of_interest]

# Calcular la matriz de correlación
correlation_matrix = df_subset.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(variables_of_interest)), variables_of_interest, rotation=45)
plt.yticks(range(len(variables_of_interest)), variables_of_interest)
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png')  # Guardar la figura como una imagen PNG

# Imprimir la correlación específica entre "preanti" y "wtkg"
correlation_preanti_wtkg = correlation_matrix.loc['preanti', 'wtkg']
print(f'Correlación entre "preanti" y "wtkg": {correlation_preanti_wtkg}')

# Indicar si la correlación es positiva o negativa
correlation_direction = 'positiva' if correlation_preanti_wtkg > 0 else 'negativa'

# Guardar resultados en un archivo PDF
with open('resultados_correlacion.pdf', 'wb') as pdf_file:
    c = canvas.Canvas(pdf_file)
    c.drawString(100, 800, f'Correlación entre "preanti" y "wtkg": {correlation_preanti_wtkg}')
    c.drawString(100, 780, f'Dirección de la correlación: {correlation_direction}')
    # Agregar el resto de la información según sea necesario
    c.showPage()  # Nueva página para la siguiente visualización
    c.drawInlineImage('correlation_matrix.png', 100, 500)  # Agregar la imagen de la matriz de correlación
    c.save()
