#  Librerías de análisis, gráficos y modelo predictivo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # Para exportar múltiples gráficas al mismo PDF
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score

#  Cargar dataset
df = pd.read_csv("science_data1.csv")  # Carga el archivo CSV con los datos
objetivo = 'crecimiento_planta'        # Define la variable objetivo que queremos predecir

#  Selección automática de variables predictoras numéricas
X_full = df.select_dtypes(include='number').drop(columns=[objetivo])  # Selecciona todas las columnas numéricas excepto la variable objetivo
y = df[objetivo]  # Define el vector de salida (objetivo)

#  Seleccionamos automáticamente las 5 mejores variables según test estadístico F
k = min(5, X_full.shape[1])  # k será 5 o el número máximo disponible de columnas
selector = SelectKBest(score_func=f_regression, k=k)
selector.fit(X_full, y)  # Aplica el selector al conjunto de datos

#  Lista de columnas seleccionadas como predictoras relevantes
predictoras = X_full.columns[selector.get_support()].tolist()

#  Mostrar las variables seleccionadas
print(" Variables seleccionadas automáticamente:")
for v in predictoras:
    print(f"• {v}")

#  Definir subconjunto de datos con las columnas seleccionadas
X = df[predictoras]

#  Crear informe PDF con todo el análisis
with PdfPages("informe_regresion_sklearn.pdf") as pdf:

    #  Página 1: Matriz de correlaciones entre variables seleccionadas
    plt.figure(figsize=(10, 8))
    matriz = df[predictoras + [objetivo]].corr()
    sns.heatmap(matriz, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(" Matriz de Correlación")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #  Página 2+: Regresiones simples por variable
    for var in predictoras:
        X_simple = df[[var]]
        y_simple = df[objetivo]

        # Entrena modelo lineal simple
        modelo = LinearRegression().fit(X_simple, y_simple)
        y_pred = modelo.predict(X_simple)

        # Métricas de desempeño
        r2 = r2_score(y_simple, y_pred)
        mse = mean_squared_error(y_simple, y_pred)
        mae = mean_absolute_error(y_simple, y_pred)

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.scatter(X_simple, y_simple, color='skyblue', label='Datos reales')
        ax.plot(X_simple, y_pred, color='red', label='Regresión')
        ax.set_xlabel(var)
        ax.set_ylabel(objetivo)
        ax.set_title(f" Regresión: {var} → {objetivo} | R² = {r2:.2f}")
        ax.legend()

        # Interpretación textual en la parte inferior
        texto = (
            f" Resultados de la regresión simple con '{var}':\n"
            f"• Intercepto: {modelo.intercept_:.2f}\n"
            f"• Coeficiente: {modelo.coef_[0]:.4f}\n"
            f"• R²: {r2:.3f} | MAE: {mae:.2f} | MSE: {mse:.2f}\n"
            f"• Relación: {'fuerte' if r2 > 0.6 else 'moderada' if r2 > 0.3 else 'débil'}"
        )
        plt.figtext(0.05, 0.02, texto, ha='left', va='bottom', fontsize=10)
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        pdf.savefig()
        plt.close()

    # 🧮 Última página: Regresión lineal múltiple
    modelo_multi = LinearRegression().fit(X, y)  # Entrena modelo con varias variables
    y_pred_multi = modelo_multi.predict(X)

    # Métricas de desempeño
    r2_multi = r2_score(y, y_pred_multi)
    mse_multi = mean_squared_error(y, y_pred_multi)
    mae_multi = mean_absolute_error(y, y_pred_multi)

    # Evaluación con validación cruzada
    cv_scores = cross_val_score(modelo_multi, X, y, cv=5, scoring='r2')

    # Página resumen con texto
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")  # Oculta ejes para dejar espacio a texto

    texto_final = f"""
 Resumen de Regresión Múltiple

• Variables predictoras: {', '.join(predictoras)}
• Intercepto: {modelo_multi.intercept_:.2f}
• R² entrenamiento: {r2_multi:.3f} | MAE: {mae_multi:.2f} | MSE: {mse_multi:.2f}
• R² validación cruzada (5 folds): {cv_scores.mean():.3f} (range {cv_scores.min():.3f}–{cv_scores.max():.3f})

Coeficientes del modelo:"""

    # Mostrar coeficientes por variable
    for var, coef in zip(predictoras, modelo_multi.coef_):
        texto_final += f"\n  • {var}: {coef:.4f}"

    texto_final += (
        "\n\n Conclusión:\nEste modelo explica aproximadamente "
        f"el {r2_multi*100:.1f}% de la variabilidad del objetivo. "
        "Puede servir para predicción, aunque también se recomienda validar con nuevos datos y explorar modelos alternativos."
    )

    # Dibujar texto línea por línea en el PDF
    for i, linea in enumerate(texto_final.strip().split('\n')):
        fig.text(0.05, 0.95 - i * 0.035, linea.strip(), fontsize=10, ha='left')

    # Guardar página final
    pdf.savefig()
    plt.close()

#  Fin
print(" Informe completo generado: 'informe_regresion_sklearn.pdf'")