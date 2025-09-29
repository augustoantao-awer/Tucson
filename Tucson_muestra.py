import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------
# Datos históricos de ejemplo
# -------------------------
# Semanas
X = np.array([31, 32, 33, 34, 35, 36, 37]).reshape(-1, 1)

# Score acumulado hasta esa semana (ejemplo real)
y_acumulado = np.array([4.104, 4.109, 4.112, 4.116, 4.121, 4.128, 4.131])

# Cantidad de reviews por semana (ejemplo)
reviews = np.array([120, 150, 180, 200, 170, 190, 210])

# -------------------------
# Inputs del usuario
# -------------------------
st.title("📊 Simulador de Score por Semana")

score_promedio = 4.68

reviews_promedio = 30


col_11, col_22 = st.columns(2)

col_11.metric("📈 Score promedio semanal", f"{score_promedio:.2f}")
col_22.metric("📈 Cantidad de reviews promedio semanal", f"{reviews_promedio:.0f}")




nuevas_reviews = st.number_input("Cantidad de reviews en semana 38", min_value=1, value=200)
nuevo_score = st.slider("Score de la semana 38", 3.5, 5.0, 4.2, 0.01)
objetivo = st.number_input("Score objetivo acumulado", value=4.150)
ticket_promedio = st.number_input("Ticket promedio", value = 30000)

# -------------------------
# Calcular score acumulado nuevo
# -------------------------
# Reviews acumulados hasta semana 37
total_reviews_prev = reviews.sum()
total_score_prev = (y_acumulado[-1] * total_reviews_prev)

# Contribución semana 38
total_reviews_new = total_reviews_prev + nuevas_reviews
total_score_new = total_score_prev + nuevo_score * nuevas_reviews
nuevo_acumulado = total_score_new / total_reviews_new

# -------------------------
# Recalcular regresión con nuevo acumulado
# -------------------------
X_sim = np.vstack([X, [38]])
y_sim = np.append(y_acumulado, nuevo_acumulado)

# Modelo real
model_real = LinearRegression()
model_real.fit(X, y_acumulado)

# Modelo simulado
model_sim = LinearRegression()
model_sim.fit(X_sim, y_sim)

# Pendiente e intercepción
m_real, b_real = model_real.coef_[0], model_real.intercept_
m_sim, b_sim = model_sim.coef_[0], model_sim.intercept_

# Calcular en qué semana se llega al objetiv o
semana_obj_real = (((objetivo - b_real) / m_real) - 37)/ 4.345
semana_obj_sim = (((objetivo - b_sim) / m_sim)  - 37) /  4.345

perdida_real = (semana_obj_real-semana_obj_sim) *7000*0.03*ticket_promedio

# -------------------------
# Resultados
# -------------------------
st.subheader("Resultados")

col_1, col_2 = st.columns(2)
col_1.metric(f"🎯 Meses hasta completar el objetivo (real)", f"{semana_obj_real:.2f}")
col_2.metric(f"🎯 Meses hasta copmpletar el objetivo (simulada)", f"{semana_obj_sim:.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("📈 Score acumulado actual", f"{y_acumulado[-1]:.3f}")
col2.metric("🔮 Score acumulado simulado", f"{nuevo_acumulado:.3f}")
col3.metric("💰 Impacto Económico", f"${perdida_real:,.0f}")
# -------------------------
# Gráfico
# -------------------------
fig, ax = plt.subplots()
# Datos reales
ax.scatter(X, y_acumulado, color="blue", label="Datos reales")
ax.plot(X, model_real.predict(X), color="blue", linestyle="--", label="Recta real")

# Datos simulados
ax.scatter(X_sim, y_sim, color="red", label="Datos simulados")
ax.plot(X_sim, model_sim.predict(X_sim), color="red", label="Recta simulada")

# Objetivo
ax.axhline(y=objetivo, color="green", linestyle="--", label=f"Objetivo {objetivo}")

ax.legend()
ax.set_xlabel("Semana")
ax.set_ylabel("Score acumulado")
ax.set_title("Predicción: real vs simulada ( Dot ) ")
st.pyplot(fig)





