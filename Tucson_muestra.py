import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulador de Score por Semana", layout="wide")

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT9yAZJxz7svYTf6GBKkrmswJo5QGJq1KssCNAEUKLPIFC7BChzOtdsPrNTB_D0GPQ_9vofAxYx-9Ch/pub?gid=898798334&single=true&output=csv"

@st.cache_data(ttl=300)  # refresca cada 5 minutos (ajustá a gusto)
def load_data():
    df = pd.read_csv(CSV_URL)
    # Esperamos columnas: semana, score, reviews
    # Limpiar tipos y vacíos
    df = df.dropna(subset=["semana_iso", "score", "cant_reviews"])
    df["semana_iso"] = df["semana_iso"].astype(int)
    df["cant_reviews"] = df["cant_reviews"].astype(int)
    df["score"] = df["score"].astype(float)
    df = df.sort_values("semana_iso")
    # ---- calcular acumulado ponderado por reviews ----
    cum_reviews = df["cant_reviews"].cumsum()
    cum_weighted = (df["score"] * df["cant_reviews"]).cumsum()
    df["y_acumulado"] = cum_weighted / cum_reviews
    return df

df = load_data()
if df.empty:
    st.warning("No llegaron filas desde el CSV de Google Sheets.")
    st.stop()

# ====== Tu lógica original, usando df ======
X = df["semana_iso"].to_numpy().reshape(-1, 1)
y_acumulado = df["y_acumulado"].to_numpy()
reviews = df["cant_reviews"].to_numpy()

st.title("📊 Simulador de Score por Semana")

score_promedio = float(np.mean(df["score"].tail(min(4, len(df)))))
reviews_promedio = int(np.mean(reviews[-min(4, len(reviews)):]))

col_11, col_22, col_33 = st.columns(3)
col_11.metric("📈 Score promedio semanal", f"{score_promedio:.2f}")
col_22.metric("📈 Reviews promedio semanal", f"{reviews_promedio:.0f}")
col_33.metric("📈 Score acumulado actual", f"{y_acumulado[-1]:.3f}")

objetivo = st.number_input("Score objetivo acumulado", value=4.150, step=0.001, format="%.3f")

ultima_semana = int(df["semana_iso"].max())
semana_nueva = ultima_semana + 1

nuevas_reviews = st.number_input(f"Cantidad de reviews en semana {semana_nueva}", min_value=1, value=max(1, reviews_promedio))
nuevo_score = st.slider(f"Score de la semana {semana_nueva}", 3.5, 5.0, float(score_promedio if not np.isnan(score_promedio) else 4.2), 0.01)
ticket_promedio = st.number_input("Ticket promedio", value=30000, step=1000)

# Cálculos
total_reviews_prev = int(reviews.sum())
total_score_prev = float(y_acumulado[-1] * total_reviews_prev) if total_reviews_prev > 0 else 0.0

total_reviews_new = total_reviews_prev + int(nuevas_reviews)
total_score_new = total_score_prev + float(nuevo_score) * int(nuevas_reviews)
nuevo_acumulado = (total_score_new / total_reviews_new) if total_reviews_new > 0 else 0.0

X_sim = np.vstack([X, [semana_nueva]])
y_sim = np.append(y_acumulado, nuevo_acumulado)

model_real = LinearRegression().fit(X, y_acumulado)
model_sim  = LinearRegression().fit(X_sim, y_sim)

m_real, b_real = float(model_real.coef_[0]), float(model_real.intercept_)
m_sim,  b_sim  = float(model_sim.coef_[0]), float(model_sim.intercept_)

def meses_hasta_obj(m, b):
    if m == 0: return float("inf")
    return (((objetivo - b) / m) - 37) / 4.345

semana_obj_real = meses_hasta_obj(m_real, b_real)
semana_obj_sim  = meses_hasta_obj(m_sim, b_sim)

perdida_real = (semana_obj_real - semana_obj_sim) * 7000 * 0.03 * ticket_promedio

# Resultados
st.subheader("Resultados")
c1, c2 = st.columns(2)
c1.metric("🎯 Meses hasta objetivo (real)", f"{semana_obj_real:.2f}" if semana_obj_real != float("inf") else "—")
c2.metric("🎯 Meses hasta objetivo (simulada)", f"{semana_obj_sim:.2f}" if semana_obj_sim != float("inf") else "—")

c3, c4 = st.columns(2)
c3.metric("🔮 Score acumulado simulado", f"{nuevo_acumulado:.3f}")
c4.metric("💰 Impacto Económico", f"${perdida_real:,.0f}")

# Gráfico
fig, ax = plt.subplots()
ax.scatter(X, y_acumulado, label="Datos reales")
ax.plot(X, model_real.predict(X), linestyle="--", label="Recta real")
ax.scatter(X_sim, y_sim, label="Datos simulados")
ax.plot(X_sim, model_sim.predict(X_sim), label="Recta simulada")
ax.axhline(y=objetivo, linestyle="--", label=f"Objetivo {objetivo}")
ax.legend()
ax.set_xlabel("Semana")
ax.set_ylabel("Score acumulado")
ax.set_title("Predicción: real vs simulada")
st.pyplot(fig)



