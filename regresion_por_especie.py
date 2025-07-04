import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# 1. Cargar los datos
df = pd.read_csv("science_data1.csv")
print("Primeras filas del dataset:")
print(df.head())

# 2. Calcular coeficiente de Pearson entre niveles_co2 y crecimiento_planta por especie
print("\nCalculando coeficientes de correlación de Pearson por especie...")
especies = df['especie'].unique()
pearsons = {}

for especie in especies:
    subset = df[df['especie'] == especie]
    r = np.corrcoef(subset['niveles_co2'], subset['crecimiento_planta'])[0, 1]
    pearsons[especie] = r
    print(f"  {especie}: r = {r:.4f}")

# 3. Regresión lineal simple por especie: niveles_co2 → crecimiento_planta
print("\nAplicando regresión por especie...")
metricas_por_especie = {}

for especie in especies:
    print(f"\nEspecie: {especie}")
    subset = df[df['especie'] == especie]
    X = subset[['niveles_co2']]
    y = subset['crecimiento_planta']

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    # Métricas
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"  Intercepto: {modelo.intercept_:.4f}")
    print(f"  Coeficiente: {modelo.coef_[0]:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    # Guardamos métricas
    metricas_por_especie[especie] = {'MSE': mse, 'MAE': mae, 'R2': r2}

    # 4. Generar gráfico por especie
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='blue', label='Datos reales')
    plt.plot(X, y_pred, color='red', label='Regresión')
    plt.xlabel('Niveles de CO₂ (ppm)')
    plt.ylabel('Crecimiento de Planta (cm)')
    plt.title(f'{especie} - CO₂ vs Crecimiento\nR² = {r2:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"regresion_co2_{especie}.png")
    plt.show()

# 5. Comparación final
print("\nResumen de métricas por especie:")
for especie, met in metricas_por_especie.items():
    print(f"{especie} -> MSE: {met['MSE']:.2f}, MAE: {met['MAE']:.2f}, R²: {met['R2']:.2f}")

# 6. Interpretación
print("\nInterpretación Final:")
for especie in especies:
    r_val = pearsons[especie]
    r2_val = metricas_por_especie[especie]['R2']
    fuerza = "fuerte" if abs(r_val) > 0.7 else "moderada" if abs(r_val) > 0.4 else "débil"
    print(f"  En {especie}, la correlación es {fuerza} (r = {r_val:.2f}) y el R² es {r2_val:.2f}.")
print("Recuerda: una correlación alta no implica necesariamente causalidad. Puede haber otros factores involucrados.")


#  Inicializar PDF
with PdfPages("informe_regresion_por_especie.pdf") as pdf:
    pearsons = {}
    metricas_por_especie = {}

    for especie in especies:
        subset = df[df['especie'] == especie]
        X = subset[['niveles_co2']]
        y = subset['crecimiento_planta']

        modelo = LinearRegression()
        modelo.fit(X, y)
        y_pred = modelo.predict(X)

        r = np.corrcoef(subset['niveles_co2'], subset['crecimiento_planta'])[0, 1]
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        pearsons[especie] = r
        metricas_por_especie[especie] = {'MSE': mse, 'MAE': mae, 'R2': r2}

        # Página 1: Gráfico
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='blue', label='Datos reales')
        plt.plot(X, y_pred, color='red', label='Regresión')
        plt.xlabel('Niveles de CO₂ (ppm)')
        plt.ylabel('Crecimiento de Planta (cm)')
        plt.title(f'{especie} - CO₂ vs Crecimiento\nR² = {r2:.2f}, r = {r:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Página 2: Interpretación
        plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
        plt.axis('off')
        fuerza = 'fuerte' if abs(r) > 0.7 else 'moderada' if abs(r) > 0.4 else 'débil'
        texto = f"""
 Interpretación para la especie: {especie}

• Coeficiente de correlación de Pearson: r = {r:.2f} → correlación {fuerza}
• R² = {r2:.2f} → el modelo explica el {r2*100:.1f}% de la variabilidad del crecimiento
• MSE: {mse:.2f} → error cuadrático medio
• MAE: {mae:.2f} → error absoluto medio (~{mae:.1f} cm de diferencia promedio)

 Nota: Aunque existe una correlación fuerte, no implica causalidad. Factores como luz, temperatura o humedad también podrían influir.
"""
        plt.text(0.05, 0.95, texto.strip(), fontsize=11, va='top', ha='left')
        pdf.savefig()
        plt.close()

    # Página final: resumen de métricas
    plt.figure(figsize=(11.69, 8.27))
    plt.axis('off')
    resumen = " Resumen de métricas por especie:\n\n"
    for especie, met in metricas_por_especie.items():
        r = pearsons[especie]
        fuerza = 'fuerte' if abs(r) > 0.7 else 'moderada' if abs(r) > 0.4 else 'débil'
        resumen += (
            f"• {especie}: r = {r:.2f} ({fuerza}), R² = {met['R2']:.2f}, "
            f"MSE = {met['MSE']:.2f}, MAE = {met['MAE']:.2f}\n"
        )
    resumen += "\n Recuerda: una correlación alta no implica causalidad. Puede haber otros factores involucrados."
    plt.text(0.05, 0.95, resumen.strip(), fontsize=11, va='top', ha='left')
    pdf.savefig()
    plt.close()

print(" Informe PDF generado exitosamente como 'informe_regresion_por_especie.pdf'")