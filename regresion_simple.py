import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 7. Generar informe en PDF con gráfico y análisis
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec


# 1. Cargar los datos

df = pd.read_csv("science_data1.csv")
print("Primeras filas del conjunto de datos:")
print(df.head())

# 2. Calcular el coeficiente de correlación de Pearson entre temperatura y crecimiento

pearson_corr = np.corrcoef(df['temperatura'], df['crecimiento_planta'])[0, 1]
print(f"Coeficiente de Pearson (r): {pearson_corr:.4f}")

# 3. Regresión lineal simple (temperatura → crecimiento_planta)
print("\nAplicación de regresión lineal simple")
X = df[['temperatura']]
y = df['crecimiento_planta']

modelo = LinearRegression()
modelo.fit(X, y)

print(f"Intersección (intercept): {modelo.intercept_:.4f}")
print(f"Pendiente (coeficiente): {modelo.coef_[0]:.4f}")

# 4. Visualización de dispersión + línea de regresión

plt.figure(figsize=(8, 6))
plt.scatter(df['temperatura'], df['crecimiento_planta'], color='blue', label='Datos')
plt.plot(df['temperatura'], modelo.predict(X), color='red', label='Línea de regresión')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Crecimiento de Planta (cm)')
plt.title(f'Regresión Lineal: Temperatura vs Crecimiento de Planta\n$R^2$ = {modelo.score(X, y):.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regresion_lineal_simple.png")
plt.show()

# 5. Calcular métricas de evaluación
print("\nCálculo de métricas de evaluación")
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# 6. Interpretación
print("\nInterpretación:")
if abs(pearson_corr) > 0.7:
    interpret = "una correlación fuerte"
elif abs(pearson_corr) > 0.5:
    interpret = "una correlación moderada"
else:
    interpret = "una correlación débil"

print(f"La correlación entre temperatura y crecimiento es {interpret}.")
print("Nota: correlación no implica causalidad. Aunque existe una relación, no se puede afirmar que una variable cause la otra directamente.")



# 7. Generar informe PDF en una sola hoja con gráfico e interpretación

with PdfPages("informe_regresion.pdf") as pdf:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Parte superior: gráfico
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(df['temperatura'], df['crecimiento_planta'], color='blue', label='Datos')
    ax1.plot(df['temperatura'], modelo.predict(X), color='red', label='Línea de regresión')
    ax1.set_xlabel('Temperatura (°C)')
    ax1.set_ylabel('Crecimiento de Planta (cm)')
    ax1.set_title(f'Regresión Lineal: Temperatura vs Crecimiento de Planta\n$R^2$ = {r2:.2f}')
    ax1.legend()
    ax1.grid(True)

    # Parte inferior: descripción e interpretación
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    texto = f"""
     **Informe de Regresión Lineal: Temperatura vs Crecimiento de Planta**

     El gráfico superior muestra una clara relación lineal positiva: a mayor temperatura, mayor crecimiento de la planta.
     La línea roja representa la regresión ajustada al comportamiento observado en los datos.

     **Resultados del modelo:**
    - **Intercepto**: {modelo.intercept_:.2f} → Valor estimado de crecimiento cuando la temperatura es 0 °C.
    - **Coeficiente**: {modelo.coef_[0]:.2f} → Por cada 1 °C que aumenta la temperatura, se estima un aumento de {modelo.coef_[0]:.2f} cm en el crecimiento.
    - **R²**: {r2:.2f} → El {r2*100:.1f}% de la variabilidad en el crecimiento es explicado por la temperatura. Muy buen ajuste.
    - **MSE**: {mse:.2f} → Error cuadrático medio. Penaliza errores grandes. Más bajo = mejor.
    - **MAE**: {mae:.2f} → Error absoluto promedio. El modelo se equivoca en promedio ~{mae:.1f} cm.
    - **Pearson r**: {pearson_corr:.4f} → {interpret.capitalize()}. Ej: {pearson_corr:.4f} indica correlación muy fuerte positiva;
     valores cercanos a 1 implican una relación directa y fuerte.

     **Conclusión**:  
    La temperatura se comporta como un predictor altamente significativo del crecimiento vegetal en este conjunto de datos.
      Aunque la relación es fuerte, recordemos que correlación no implica causalidad.
    """
    ax2.text(0.01, 0.98, texto.strip(), fontsize=10, va='top', ha='left')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("\n Informe PDF generado exitosamente como 'informe_regresion.pdf'")