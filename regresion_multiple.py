import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# 1. Cargar los datos
df = pd.read_csv("science_data1.csv")
print("Primeras filas del dataset:")
print(df.head())

# 2. Selección de predictores y variable objetivo
print("\nSelección de predictores y variable objetivo")
X = df[['temperatura', 'intensidad_luz', 'niveles_co2']]
y = df['crecimiento_planta']
print("Variables independientes:\n", X.head())
print("Variable dependiente:\n", y.head())

# 3. Entrenar modelo de regresión lineal múltiple
print("\nEntrenar modelo de regresión lineal múltiple")
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

# Mostrar coeficientes
print("\nCoeficientes del modelo:")
for var, coef in zip(X.columns, modelo.coef_):
    print(f"  {var}: {coef:.4f}")
print(f"Intercepto: {modelo.intercept_:.4f}")

# 4. Visualización: Relación entre intensidad_luz y crecimiento_planta
print("gráfico de dispersión (intensidad_luz vs crecimiento)")
plt.figure(figsize=(8,6))
plt.scatter(df['intensidad_luz'], df['crecimiento_planta'], color='blue', label='Datos')
plt.scatter(df['intensidad_luz'], y_pred, color='red', alpha=0.5, label='Predicción')
plt.xlabel('Intensidad de Luz (lúmenes)')
plt.ylabel('Crecimiento de Planta (cm)')
plt.title('Regresión Múltiple: Intensidad de Luz vs Crecimiento (con predicción)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regresion_multiple_luz_vs_crecimiento.png")
plt.show()

# 5. Calcular métricas
print("\nEvaluar el modelo.")
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# 6. Interpretación
print("\nInterpretación:")
if r2 > 0.6:
    conclusion = "una relación fuerte entre las variables predictoras y el crecimiento"
elif r2 > 0.4:
    conclusion = "una relación moderada"
else:
    conclusion = "una relación débil"

print(f"El R² obtenido indica {conclusion}.")
print("Es importante notar que una buena correlación no garantiza causalidad.")
print("Una posible variable de confusión podría ser la 'humedad', que influye tanto en la fotosíntesis como en el metabolismo de la planta.")





# Crear el PDF
with PdfPages("informe_regresion_multiple.pdf") as pdf:
    # Página 1: Gráfico
    fig1, ax1 = plt.subplots(figsize=(11.69, 8.27))  # A4 horizontal
    ax1.scatter(df['intensidad_luz'], df['crecimiento_planta'], color='blue', label='Datos observados')
    ax1.scatter(df['intensidad_luz'], y_pred, color='red', alpha=0.5, label='Predicción del modelo')
    ax1.set_xlabel('Intensidad de Luz (lúmenes)')
    ax1.set_ylabel('Crecimiento de Planta (cm)')
    ax1.set_title('Regresión Múltiple: Intensidad de Luz vs Crecimiento (con predicción)')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close()

    # Página 2: Interpretación
    fig2, ax2 = plt.subplots(figsize=(11.69, 8.27))
    ax2.axis('off')
    # Texto interpretativo detallado para el informe
    texto = f"""
    **Informe de Regresión Lineal Múltiple**

    Este gráfico de dispersión representa una regresión múltiple que examina la relación entre la intensidad de la luz (en lúmenes) y el crecimiento 
    de la planta (en cm), incluyendo una predicción basada en el modelo.

    **Interpretación del gráfico**
    - **Relación directa**: Se observa una relación positiva entre la intensidad de la luz y el crecimiento. A mayor intensidad, mayor desarrollo vegetal.
    - **Datos observados (azules)**: Representan los valores reales de crecimiento de cada planta en función de la luz recibida.
    - **Predicción del modelo (rojos)**: Los puntos rojos indican el crecimiento estimado por el modelo. La cercanía entre estos puntos y los reales sugiere un buen ajuste.
    - **Variabilidad**: Existe dispersión en los datos reales, lo cual es esperable en fenómenos biológicos. No toda la variación se debe a la intensidad de luz.

    ---

    **Coeficientes del modelo**
    - **Temperatura**: {modelo.coef_[0]:.4f} → Por cada 1 °C adicional, el crecimiento promedio aumenta en {modelo.coef_[0]:.2f} cm.
    - **Intensidad de luz**: {modelo.coef_[1]:.4f} → Cada aumento de 1000 lúmenes aporta aproximadamente {modelo.coef_[1]*1000:.2f} cm de crecimiento.
    - **Niveles de CO₂**: {modelo.coef_[2]:.4f} → Cada incremento en ppm de CO₂ se asocia con un leve aumento en crecimiento.
    - **Intercepto**: {modelo.intercept_:.4f} → Valor estimado de crecimiento cuando todas las variables predictoras son 0 
    (no interpretable directamente).

    ---

    **Evaluación del modelo**
    - **Mean Squared Error (MSE)**: {mse:.2f} → Error cuadrático promedio.
    - **Mean Absolute Error (MAE)**: {mae:.2f} → Error absoluto medio (~{mae:.1f} cm de diferencia promedio).
    - **R²**: {r2:.4f} → El {r2*100:.1f}% de la variabilidad del crecimiento se explica por las variables seleccionadas.

    **Conclusión**:
    El modelo muestra una relación **fuerte** entre las variables predictoras y el crecimiento de la planta. La intensidad de la luz, 
    junto con la temperatura y el CO₂, son buenos predictores, aunque no explican toda la variabilidad.

    **Advertencia**:
    Una buena correlación no implica causalidad. Por ejemplo, la humedad podría ser una variable de confusión relevante, ya que influye tanto 
    la fotosíntesis como en la fisiología de la planta.
    """
    ax2.text(0.01, 0.98, texto.strip(), fontsize=10, va='top', ha='left')
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close()


print("\n✅ Informe PDF generado exitosamente como 'informe_regresion_multiple.pdf'")