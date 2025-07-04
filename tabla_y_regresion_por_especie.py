import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages

# 1. Cargar los datos
df = pd.read_csv("science_data1.csv")
print("Primeras filas del dataset:")
print(df.head())

# 2. Categorizar crecimiento_planta en Bajo, Medio, Alto
def categorizar_crecimiento(cm):
    if cm < 20:
        return 'Bajo'
    elif cm <= 35:
        return 'Medio'
    else:
        return 'Alto'

df['categoria_crecimiento'] = df['crecimiento_planta'].apply(categorizar_crecimiento)

# 3. Tabla de contingencia especie vs categoría
tabla = pd.crosstab(df['especie'], df['categoria_crecimiento'])

# 4. Regresión lineal por especie
r2_por_especie = {}
modelos = {}

for especie in df['especie'].unique():
    subset = df[df['especie'] == especie]
    X = subset[['temperatura']]
    y = subset['crecimiento_planta']

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    modelos[especie] = {
        'intercepto': modelo.intercept_,
        'coeficiente': modelo.coef_[0],
        'r2': r2
    }
    r2_por_especie[especie] = r2

mejor_modelo = max(r2_por_especie, key=r2_por_especie.get)

# 5. Texto completo del informe 
texto = f"""
**Informe: Análisis de Crecimiento por Especie**

El gráfico de barras apiladas muestra la "Distribución de Categorías de Crecimiento por Especie", comparando las especies A, B y C en términos de categorías de crecimiento: Alto (verde azulado), Bajo (verde claro) y Medio (gris).

**Especie A:**
Presenta una distribución más equitativa entre las categorías de crecimiento, con una porción significativa en crecimiento Alto y Bajo, y una menor proporción en crecimiento Medio.

**Especie B:**
Muestra una menor proporción de crecimiento Alto en comparación con las otras especies, y una mayor proporción de crecimiento Medio.

**Especie C:**
Tiene una distribución similar a la Especie A en cuanto a crecimiento Alto y Bajo, pero con una proporción ligeramente mayor de crecimiento Medio en comparación con la Especie A.

En general, la Especie B se distingue por tener la menor cantidad de plantas con crecimiento "Alto" y la mayor cantidad con crecimiento "Medio" en comparación con las otras dos especies.

---

**¿Qué significa cada métrica del modelo?**

- **Intercepto**: Estimación del crecimiento cuando la temperatura es 0 °C. Es un punto teórico de partida y no siempre representa una situación realista.
- **Coeficiente**: Indica cuánto crece la planta por cada 1 °C adicional. Refleja la sensibilidad al cambio de temperatura.
- **R² (coeficiente de determinación)**: Mide qué porcentaje de la variabilidad del crecimiento puede ser explicado por la temperatura. Valores más cercanos a 1 indican mejor ajuste del modelo.

---

**Resultados por especie**

 Especie A  
- Intercepto: -21.82  
→ Si la temperatura fuera 0 °C, el crecimiento estimado sería -21.82 cm. Este valor no tiene sentido físico, pero es útil para trazar la recta del modelo.  
- Coeficiente: 1.92  
→ Por cada grado adicional, la planta crece en promedio 1.92 cm. Muestra alta sensibilidad a la temperatura.  
- R² = 0.85  
→ El 85% de la variación en el crecimiento puede explicarse por la temperatura. Es el mejor ajuste entre las tres especies.

 Especie B  
- Intercepto: -12.32  
→ Estimación inicial del modelo cuando la temperatura es 0 °C.  
- Coeficiente: 1.64  
→ Por cada 1 °C más, el crecimiento aumenta en promedio 1.64 cm.  
- R² = 0.79  
→ El modelo explica el 79% del crecimiento. Ajuste sólido, aunque menor que el de Especie A y C.

 Especie C  
- Intercepto: -13.53  
→ Punto de partida estimado en la recta del modelo.  
- Coeficiente: 1.74  
→ Cada grado adicional aporta, en promedio, 1.74 cm de crecimiento.  
- R² = 0.82  
→ La temperatura explica el 82% de la variabilidad. Muy buen ajuste.

---

**Conclusión final**

La especie con mejor ajuste del modelo lineal es **{mejor_modelo}**, con un R² = {r2_por_especie[mejor_modelo]:.2f}, lo que significa que el modelo puede explicar aproximadamente un {r2_por_especie[mejor_modelo]*100:.1f}% de la variabilidad en el crecimiento de esa especie a partir de la temperatura.

 **Nota importante**: Aunque un mayor R² indica que el modelo es más preciso en su predicción, esto **no implica causalidad directa**. La temperatura se relaciona con el crecimiento, pero no necesariamente lo causa.
"""

# 6. Crear PDF (2 páginas)
with PdfPages("informe_especies.pdf") as pdf:
    # Página 1: gráfico
    fig1, ax1 = plt.subplots(figsize=(11.69, 8.27))
    tabla.plot(kind='bar', stacked=True, colormap='Set2', ax=ax1)
    ax1.set_title('Distribución de Categorías de Crecimiento por Especie')
    ax1.set_xlabel('Especie')
    ax1.set_ylabel('Cantidad de Plantas')
    ax1.legend(title='Categoría de Crecimiento')
    ax1.grid(True)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close()

    # Página 2: texto
    fig2, ax2 = plt.subplots(figsize=(11.69, 8.27))
    ax2.axis('off')
    ax2.text(0.01, 0.98, texto.strip(), fontsize=10, va='top', ha='left')
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close()

print("\n Informe PDF generado exitosamente como 'informe_especies.pdf'")