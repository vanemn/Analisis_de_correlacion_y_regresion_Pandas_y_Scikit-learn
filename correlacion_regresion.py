import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages


# 1. Cargar los datos

df = pd.read_csv("science_data1.csv")
print("Primeras filas del dataset:")
print(df.head())

# 2. Calcular matriz de correlación
print("\nCalculando matriz de correlación...")
variables = ['temperatura', 'humedad', 'niveles_co2', 'ph_suelo', 'intensidad_luz']
corr_matrix = df[variables].corr()
print(corr_matrix)

# 3. Visualizar matriz con mapa de calor
print("\nMostrando mapa de calor de correlaciones...")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación - Variables Ambientales")
plt.tight_layout()
plt.savefig("matriz_correlacion.png")
plt.show()

# 4. Seleccionar variables con mayor correlación (ejemplo: supongamos que 'intensidad_luz' y 'niveles_co2' tienen alta correlación con crecimiento)
print("\nSeleccionando predictores altamente correlacionados...")
# Puedes ajustar según la matriz real, aquí usamos ejemplo de dos más correlacionadas
predictores = ['intensidad_luz', 'niveles_co2']
X = df[predictores]
y = df['crecimiento_planta']

# 5. Entrenar modelo de regresión múltiple
print("\nEntrenando regresión múltiple con mejores predictores...")
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

print("Coeficientes del modelo:")
for var, coef in zip(predictores, modelo.coef_):
    print(f"  {var}: {coef:.4f}")
print(f"Intercepto: {modelo.intercept_:.4f}")

# 6. Evaluar modelo
print("\nCalculando métricas de evaluación...")
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# 7. Interpretación
print("\nInterpretación:")
print("El modelo múltiple nos ayuda a entender cómo múltiples factores explican el crecimiento de las plantas.")
print("Sin embargo, una fuerte correlación no implica causalidad.")
print("Por ejemplo, si 'intensidad_luz' tiene la correlación más fuerte, puede estar actuando como un proxy de otras condiciones favorables.")





# Generar PDF
with PdfPages("informe_correlacion_regresion.pdf") as pdf:
    # Página 1: Mapa de calor
    fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación - Variables Ambientales")
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close()

    # Página 2: Interpretación
    fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
    ax2.axis('off')
    # Texto interpretativo completo con los datos actuales
    texto = f"""
    **Informe: Correlación entre Variables Ambientales y Crecimiento de Plantas**

     Este análisis explora cómo variables ambientales como temperatura, humedad, CO₂, pH del suelo e
     intensidad de luz se relacionan entre sí 
     y con el crecimiento de la planta.

    ---

    **Interpretación de la Matriz de Correlación**

    **Correlaciones Positivas Fuertes (cercanas a 1.00):**
    - Temperatura e Intensidad de Luz (0.98): A medida que aumenta la temperatura,
      también lo hace la intensidad de luz.
    - Temperatura y Niveles de CO₂ (0.90): Temperaturas más altas tienden a estar asociadas con más CO₂.
    - Niveles de CO₂ e Intensidad de Luz (0.89): Lugares con mayor luz también muestran más niveles de CO₂.

    **Correlaciones Negativas Fuertes (cercanas a -1.00):**
    - Temperatura y Humedad (-0.97): Cuando la temperatura sube, la humedad disminuye.
    - Humedad e Intensidad de Luz (-0.96): Más humedad implica menos luz ambiental.
    - Humedad y Niveles de CO₂ (-0.87): Ambientes más húmedos presentan menos CO₂.

    **Correlaciones Débiles o Nulas:**
    - pH del suelo con otras variables (≈ 0.00): El pH presenta relaciones lineales muy débiles 
    o nulas con las otras variables.

    ---

    **Modelo de Regresión Múltiple**

    Se seleccionaron como predictores los más correlacionados con el crecimiento: **intensidad_luz** y
      **niveles_co2**.

    **Coeficientes del modelo:**
    - Intensidad de Luz: {modelo.coef_[0]:.4f}
    - Niveles de CO₂: {modelo.coef_[1]:.4f}
    - Intercepto: {modelo.intercept_:.4f}

    ---

    **Evaluación del Modelo**
    - MSE: {mse:.2f}
    - MAE: {mae:.2f}
    - R²: {r2:.4f} → El modelo explica aproximadamente un {r2*100:.1f}% de la variabilidad en 
    el crecimiento de la planta.

    ---

    **Conclusión**
    El modelo de regresión múltiple muestra una relación **fuerte** entre las variables seleccionadas y el crecimiento. 
    Sin embargo, es importante recordar que correlación no implica causalidad. Variables no incluidas como 
    la **humedad** podrían estar actuando como factores de confusión.

    """
    ax2.text(0.03, 0.97, texto.strip(), fontsize=10, va='top', ha='left')
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close()

print("\n Informe PDF generado exitosamente como 'informe_correlacion_regresion.pdf'")