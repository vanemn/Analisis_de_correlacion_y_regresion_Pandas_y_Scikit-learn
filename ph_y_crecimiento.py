
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# 1. Cargar los datos

df = pd.read_csv("science_data1.csv")
print("Primeras filas del dataset:")
print(df.head())

# 2. Categorizar ph_suelo en Ácido, Neutro y Alcalino
print("\nCategorizando ph_suelo...")
def categorizar_ph(valor):
    if valor < 6.0:
        return "Ácido"
    elif valor <= 7.0:
        return "Neutro"
    else:
        return "Alcalino"

df['categoria_ph'] = df['ph_suelo'].apply(categorizar_ph)
print("Distribución por categoría de pH del suelo:")
print(df['categoria_ph'].value_counts())

# 3. Crear tabla de contingencia entre especie y categoría de pH
print("\nGenerando tabla de contingencia...")
tabla_contingencia = pd.crosstab(df['especie'], df['categoria_ph'])
print(tabla_contingencia)

# 4. Visualización: gráfico de barras agrupadas
print("\nMostrando gráfico de barras agrupadas...")
tabla_contingencia.plot(kind='bar', colormap='Set3')
plt.title("Distribución de Categorías de pH del Suelo por Especie")
plt.xlabel("Especie")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("barras_ph_vs_especie.png")
plt.show()

# 5. Regresión lineal simple: ph_suelo → crecimiento_planta
print("\nAplicando regresión lineal simple (ph_suelo → crecimiento_planta)...")
X = df[['ph_suelo']]
y = df['crecimiento_planta']

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

# Coeficientes
print(f"  Intercepto: {modelo.intercept_:.4f}")
print(f"  Coeficiente: {modelo.coef_[0]:.4f}")

# 6. Evaluar el modelo
print("\nEvaluando el modelo...")
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Interpretación
print("\nInterpretación:")
if r2 > 0.6:
    conclusion = "una fuerte relación lineal entre pH del suelo y crecimiento"
elif r2 > 0.3:
    conclusion = "una relación moderada"
else:
    conclusion = "una relación débil o casi nula"

print(f"El R² obtenido indica {conclusion}. Esto sugiere que el pH del suelo por sí solo no predice de forma significativa el crecimiento de las plantas.")
print("Sin embargo, puede ser relevante en combinación con otras variables.")



with PdfPages("informe_ph_y_crecimiento.pdf") as pdf:
    # Página 1: Barras + interpretación
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 horizontal
    tabla_contingencia.plot(kind='bar', colormap='Set3', ax=ax)
    ax.set_title("Distribución de Categorías de pH del Suelo por Especie")
    ax.set_xlabel("Especie")
    ax.set_ylabel("Cantidad")

    # Texto debajo del gráfico
    texto1 = (
        "Interpretación:\n"
        "- Las especies muestran distribuciones similares entre categorías de pH.\n"
        "- La categoría 'Ácido' es ligeramente más frecuente en general.\n"
        "- Esto sugiere que las especies crecen en entornos con diferentes características de acidez."
    )

    # Dibujar texto en la figura (parte inferior)
    plt.figtext(0.05, 0.02, texto1, ha="left", va="bottom", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # deja espacio inferior
    pdf.savefig()
    plt.close()

    # Página 2: Regresión pH → crecimiento + interpretación
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 horizontal
    ax.scatter(df['ph_suelo'], y, color='blue', label='Datos reales')
    ax.plot(df['ph_suelo'], y_pred, color='red', label='Regresión lineal')
    ax.set_xlabel('pH del Suelo')
    ax.set_ylabel('Crecimiento de Planta (cm)')
    ax.set_title(f'Regresión: pH del Suelo vs Crecimiento\nR² = {r2:.2f}')
    ax.legend()
    ax.grid(True)

    # Interpretación junto al gráfico
    fuerza = "fuerte" if r2 > 0.6 else "moderada" if r2 > 0.3 else "débil o casi nula"
    texto2 = (
        f"Interpretación:\n"
        f"- Intercepto: {modelo.intercept_:.2f}\n"
        f"- Coeficiente: {modelo.coef_[0]:.4f} → cada unidad de pH cambia el crecimiento en {modelo.coef_[0]:.2f} cm\n"
        f"- R²: {r2:.4f} → relación {fuerza} entre pH y crecimiento\n"
        "- El modelo lineal no explica gran parte de la variabilidad por sí solo,\n"
        "  por lo que el pH podría ser útil solo en combinación con otras variables\n"
        "  como luz, humedad o temperatura."
    )

    plt.figtext(0.05, 0.02, texto2, ha="left", va="bottom", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    pdf.savefig()
    plt.close()

print(" PDF generado exitosamente con gráfico + interpretación por página.")
