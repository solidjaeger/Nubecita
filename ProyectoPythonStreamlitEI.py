import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Proyecto Estadística - FACYT", layout="wide")
st.title("📊 Análisis Estadístico: Infraestructura Core Bancario")
st.markdown("---")

# Creación de pestañas para organizar la aplicación
tab1, tab2 = st.tabs(["Parte I: ANOVA (Arquitecturas)", "Parte II: Regresión (Uso de CPU)"])

# ==========================================
# PARTE I: ANOVA UNIFACTORIAL
# ==========================================
with tab1:
    st.header("Análisis de Varianza: SwiftVen vs SwiftFast vs SwiftPay")
    
    # 1. Carga de Datos

    st.header("Configuración de Datos: Parte I")

    # 1. Definimos los datos en el formato del PDF (Ancho)
    datos_pdf = {
        'Arquitectura': ['SwiftVen', 'SwiftFast', 'SwiftPay'],
        'Muestra 1': [3.30, 3.25, 3.10],
        'Muestra 2': [3.42, 3.15, 3.25],
        'Muestra 3': [3.36, 3.30, 3.18],
        'Muestra 4': [3.34, 3.20, 3.12]
    }

    df_pdf = pd.DataFrame(datos_pdf)

    st.subheader("Panel de Control (Formato PDF)")
    # 2. Editor dinámico con columnas por muestra
    df_editado = st.data_editor(
        df_pdf, 
        num_rows="dynamic", 
        use_container_width=True
    )

    # 3. TRANSFORMACIÓN CRÍTICA: Convertimos a formato largo para el ANOVA
    # Esto hace que las columnas 'Muestra X' vuelvan a ser filas
    data_anova = df_editado.melt(
        id_vars=['Arquitectura'], 
        value_vars=['Muestra 1', 'Muestra 2', 'Muestra 3', 'Muestra 4'],
        var_name='Muestra', 
        value_name='Latencia'
    )

    # A partir de aquí, usas 'data_anova' para tus cálculos de siempre
    
    # 2. Estadística Descriptiva y Boxplot
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadística Descriptiva")
        # --- REEMPLAZA TU LÍNEA 39 CON ESTE BLOQUE ---

        st.subheader("Estadística Descriptiva")

        # 1. Obtenemos el resumen estándar (count, mean, std, etc.)
        resumen = data_anova.groupby('Arquitectura')['Latencia'].describe()

        # 2. Seleccionamos solo lo que necesitamos para el proyecto
        stats_df = resumen[['count', 'mean', 'std']].copy()

        # 3. Calculamos la varianza (S²) manualmente y la agregamos como columna
        stats_df['var'] = data_anova.groupby('Arquitectura')['Latencia'].var()

        # 4. Mostramos la tabla en Streamlit
        st.dataframe(stats_df.style.format("{:.4f}")) 

        # ---------------------------------------------    
    with col2:
        st.subheader("Visualización de Distribución")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='Arquitectura', y='Latencia', data=data_anova, ax=ax)
        st.pyplot(fig)

    # 3. Modelo ANOVA
    st.subheader("Tabla ANOVA")
    modelo_anova = ols('Latencia ~ C(Arquitectura)', data=data_anova).fit()
    tabla_anova = sm.stats.anova_lm(modelo_anova, typ=2)
    st.dataframe(tabla_anova)
    
    # Acceso correcto al P-valor usando la etiqueta de la fila
    p_valor_anova = tabla_anova.loc['C(Arquitectura)', 'PR(>F)']

    if p_valor_anova < 0.05:
        st.success(f"**Resultado:** El P-valor es {p_valor_anova:.4f} < 0.05. Se rechaza Ho. Hay diferencia significativa.")
    else:
        st.warning(f"**Resultado:** El P-valor es {p_valor_anova:.4f} > 0.05. No se rechaza Ho.")

    # 4. Validación de Supuestos
    st.subheader("Matriz Diagnóstica (Supuestos)")
    residuales = modelo_anova.resid
    
    # Normalidad (Lilliefors)
    st.write("**1. Normalidad (Lilliefors):**")
    stat_l, p_lilliefors = lilliefors(residuales)
    if p_lilliefors > 0.05: st.info(f"Cumple (p={p_lilliefors:.4f}). Los datos son normales.")
    else: st.error(f"No cumple (p={p_lilliefors:.4f}).")
    
    # Homocedasticidad (Levene como alternativa a Cochran)
    st.write("**2. Homocedasticidad (Levene):**")
    stat_lev, p_levene = stats.levene(data_anova['Latencia'][data_anova['Arquitectura']=='SwiftVen'],
                                      data_anova['Latencia'][data_anova['Arquitectura']=='SwiftFast'],
                                      data_anova['Latencia'][data_anova['Arquitectura']=='SwiftPay'])
    if p_levene > 0.05: st.info(f"Cumple (p={p_levene:.4f}). Las varianzas son iguales.")
    else: st.error(f"No cumple (p={p_levene:.4f}).")
    
    # Independencia (Prueba de Rachas)
    st.write("**3. Independencia (Rachas):**")
    stat_r, p_rachas = runstest_1samp(residuales, cutoff='median')
    if p_rachas > 0.05: st.info(f"Cumple (p={p_rachas:.4f}). Los datos son independientes.")
    else: st.error(f"No cumple (p={p_rachas:.4f}).")

    # 5. Plan B o Comparaciones Múltiples
    st.subheader("Análisis Post-Hoc / Alternativa")
    if p_lilliefors > 0.05 and p_levene > 0.05:
        st.write("Como se cumplen los supuestos, aplicamos **Tukey HSD** (estándar moderno equivalente a Duncan).")
        tukey = pairwise_tukeyhsd(endog=data_anova['Latencia'], groups=data_anova['Arquitectura'], alpha=0.05)
        # En la parte de Tukey, cambia st.text por esto:
        st.write("**Resultados de la Prueba de Rangos (Tukey):**")
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        st.dataframe(tukey_df)
    else:
        st.write("Fallo en supuestos. Aplicando **Kruskal-Wallis** (No paramétrica).")
        stat_kw, p_kw = stats.kruskal(data_anova['Latencia'][data_anova['Arquitectura']=='SwiftVen'],
                                      data_anova['Latencia'][data_anova['Arquitectura']=='SwiftFast'],
                                      data_anova['Latencia'][data_anova['Arquitectura']=='SwiftPay'])
        st.write(f"P-valor Kruskal-Wallis: {p_kw:.4f}")


# ==========================================
# PARTE II: REGRESIÓN LINEAL MÚLTIPLE
# ==========================================
with tab2:
    st.header("Modelo de Predicción de CPU")
    
    # 1. Carga de Datos (Completa según la tabla)
    data_reg = pd.DataFrame({
        'Y': [9.8, 12.6, 11.9, 13.1, 13.3, 13.5, 10.1, 13.1, 10.7, 11.0, 13.0, 11.6, 12.0, 11.4, 12.2, 12.8, 12.4, 13.2, 10.6, 7.9],
        'X1': [3.3, 4.4, 3.9, 5.9, 4.6, 5.2, 4.0, 4.7, 4.5, 3.7, 4.6, 4.7, 3.9, 4.6, 5.1, 5.0, 4.8, 5.3, 3.9, 3.4],
        'X2': [2.8, 4.9, 5.3, 2.6, 5.1, 3.2, 4.0, 4.5, 4.1, 3.6, 4.6, 3.5, 4.6, 4.0, 3.6, 4.4, 4.4, 3.5, 3.8, 3.8],
        'X3': [3.1, 3.5, 4.8, 3.1, 5.0, 3.3, 3.3, 3.5, 3.7, 3.3, 3.6, 3.5, 3.6, 3.4, 3.3, 3.6, 3.4, 3.6, 3.4, 3.4],
        'X4': [4.1, 3.9, 4.7, 3.6, 4.1, 4.3, 4.0, 3.8, 3.6, 3.6, 3.6, 3.7, 4.1, 3.6, 4.0, 3.7, 3.6, 3.7, 4.0, 3.4]
    })
    
    # 2. Construcción del Modelo
    modelo_mlr = smf.ols('Y ~ X1 + X2 + X3 + X4', data=data_reg).fit()
    
    # --- BLOQUE DE VISUALIZACIÓN UNIFICADO (REEMPLAZA LO ANTERIOR) ---
    st.subheader("Análisis de Coeficientes y Bondad de Ajuste")

    # 1. Cálculo de Intervalos de Confianza (Pregunta 5)
    intervalos = modelo_mlr.conf_int(alpha=0.05)
    intervalos.columns = ['Límite Inf. (2.5%)', 'Límite Sup. (97.5%)']

    # 2. Tabla Maestra de Coeficientes (Preguntas 5 y 6)
    df_coef_completo = pd.concat([
        pd.DataFrame({
            'Coeficiente (β)': modelo_mlr.params,
            'Error Estándar': modelo_mlr.bse,
            'Estadístico t': modelo_mlr.tvalues,
            'P-valor': modelo_mlr.pvalues
        }),
        intervalos
    ], axis=1)

    st.dataframe(df_coef_completo.style.format("{:.4f}"), use_container_width=True)

    # 3. Métricas de Bondad de Ajuste Global (Pregunta 3 y 4)
    st.write("**Estadísticos Globales del Modelo:**")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("R-squared ($R^2$)", f"{modelo_mlr.rsquared:.4f}")
    with col_m2:
        st.metric("$R^2$ Ajustado", f"{modelo_mlr.rsquared_adj:.4f}")
    with col_m3:
        st.metric("Estadístico F", f"{modelo_mlr.fvalue:.2f}")
    with col_m4:
        st.metric("P-valor (F)", f"{modelo_mlr.f_pvalue:.2e}")

    # 4. Mensaje de Validación Global
    if modelo_mlr.f_pvalue < 0.05:
        st.success(f"✅ El modelo es globalmente significativo ($P < 0.05$).")
    else:
        st.error("❌ El modelo no es significativo globalmente.")
    
    st.markdown("---")
    # --- FIN DEL BLOQUE UNIFICADO ---
    # -------------------------------------------------------
    
    # 3. Interpretación de R2 y Optimización
    st.subheader("Análisis de Significancia y Optimización")
    st.write(f"**Coeficiente de Determinación (R²):** {modelo_mlr.rsquared:.4f}. Explica el {modelo_mlr.rsquared*100:.2f}% de la variabilidad del CPU.")
    
    # Extraer p-valores de las variables
    p_valores = modelo_mlr.pvalues[1:] # Omitir intercepto
    variables_insignificantes = p_valores[p_valores > 0.05].index.tolist()
    
    if variables_insignificantes:
        st.warning(f"Variables no significativas (p > 0.05): {', '.join(variables_insignificantes)}. Se recomienda probar un modelo reducido eliminándolas.")
    else:
        st.success("Todas las variables son significativas.")

    # 4. Motor de Predicción
    st.subheader("Motor de Predicción de CPU")
    st.write("Ingrese los valores de los parámetros para predecir el uso de CPU:")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1: x1_in = st.number_input("Peticiones/s (X1)", value=5.1)
    with col_p2: x2_in = st.number_input("Tamaño trama (X2)", value=4.7)
    with col_p3: x3_in = st.number_input("Latencia Bóveda (X3)", value=4.8)
    with col_p4: x4_in = st.number_input("Consumo Memoria (X4)", value=4.0)
    
    if st.button("Predecir Uso de CPU"):
        input_data = pd.DataFrame({'X1': [x1_in], 'X2': [x2_in], 'X3': [x3_in], 'X4': [x4_in]})
        prediccion = modelo_mlr.predict(input_data)
        st.success(f"**Predicción Estimada de CPU:** {prediccion.iloc[0]:.2f}%")
