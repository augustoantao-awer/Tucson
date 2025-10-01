import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulador de Score por Semana", layout="wide")

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQXEH7M3Dp2KGSJ1isnOmNt3EblHbDXO-gf2VoCCCdAspn4zD1D0YGiGNml5HnJNY9qJOJI_yCV8LCU/pub?output=csv"

#%%
@st.cache_data(ttl=300)  # refresca cada 5 minutos (ajustá a gusto)
def load_data():
    df = pd.read_csv(CSV_URL)

    # Limpiar y tipar
    df = df.dropna(subset=["semana_iso", "score", "cant_reviews"])
    df["semana_iso"] = df["semana_iso"].astype(int)
    df["cant_reviews"] = pd.to_numeric(df["cant_reviews"], errors="coerce").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.sort_values("semana_iso").reset_index(drop=True)

    # Si querés mantener tu recorte de la última fila:
    df = df.drop(df.index[-1])

    # Baseline histórico
    base_count = 3690
    base_score = 4.1
    base_weighted = base_count * base_score

    # Acumulados
    cumsum_count = df["cant_reviews"].cumsum()
    cumsum_weighted = (df["cant_reviews"] * df["score"]).cumsum()

    df["score_acumulado"] = (base_weighted + cumsum_weighted) / (base_count + cumsum_count)
    # (opcional) redondeo
    df["score_acumulado"] = df["score_acumulado"].round(4)

   

    return df

df = load_data()
base_count = 3690
base_score = 4.1
base_weighted = base_count * base_score
X = df["semana_iso"].to_numpy().reshape(-1, 1)
y_acumulado = df["score_acumulado"].to_numpy()
reviews = df["cant_reviews"].to_numpy()

st.title("📊 Simulador de Score por Semana")

score_promedio = float(df["score"].mean(skipna=True))
reviews_promedio = int(df["cant_reviews"].mean(skipna=True))

col_11, col_22, col_33 = st.columns(3)
col_11.metric("📈 Score promedio semanal", f"{score_promedio:.2f}")
col_22.metric("📈 Reviews promedio semanal", f"{reviews_promedio:.0f}")
col_33.metric("📈 Score acumulado actual", f"{y_acumulado[-1]:.3f}" if len(y_acumulado) else "—")

objetivo = st.number_input("Score objetivo acumulado", value=4.150, step=0.001, format="%.f")

ultima_semana = int(df["semana_iso"].max())
semana_nueva = ultima_semana + 1

nuevas_reviews = st.number_input(
    f"Cantidad de reviews en semana {semana_nueva}",
    min_value=1, value=max(1, reviews_promedio)
)
nuevo_score = st.slider(
    f"Score de la semana {semana_nueva}",
    3.5, 5.0, float(score_promedio if not np.isnan(score_promedio) else 4.2), 0.01
)
ticket_promedio = st.number_input("Ticket promedio", value=30000, step=1000)

# ====== Cálculo coherente con el acumulado de arriba ======
# Totales hasta la última semana (incluyendo baseline)
total_weekly_reviews = int(df["cant_reviews"].sum())
total_weekly_weighted = float((df["cant_reviews"] * df["score"]).sum())

total_reviews_prev = base_count + total_weekly_reviews
total_weighted_prev = base_weighted + total_weekly_weighted

acumulado_actual = total_weighted_prev / total_reviews_prev  # debería ≈ y_acumulado[-1]

# Agrego la nueva semana simulada
total_reviews_new = total_reviews_prev + int(nuevas_reviews)
total_weighted_new = total_weighted_prev + float(nuevo_score) * int(nuevas_reviews)
nuevo_acumulado = (total_weighted_new / total_reviews_new) if total_reviews_new > 0 else 0.0

# Series para ajustar la recta (real y simulada)
X_sim = np.vstack([X, [semana_nueva]])
y_sim = np.append(y_acumulado, nuevo_acumulado)

model_real = LinearRegression().fit(X, y_acumulado)
model_sim  = LinearRegression().fit(X_sim, y_sim)

m_real, b_real = float(model_real.coef_[0]), float(model_real.intercept_)
m_sim,  b_sim  = float(model_sim.coef_[0]), float(model_sim.intercept_)

def meses_hasta_obj(m, b, semana_actual, objetivo):
    # Si la pendiente no sube, no se alcanza el objetivo.
    if m <= 0:
        return float("inf")
    semana_obj = (objetivo - b) / m
    # meses desde la semana actual (no restes constantes mágicas)
    return max(0.0, (semana_obj - semana_actual) / 4.345)

meses_obj_real = meses_hasta_obj(m_real, b_real, ultima_semana, objetivo)
meses_obj_sim  = meses_hasta_obj(m_sim,  b_sim,  ultima_semana+1, objetivo)

# Impacto económico según tu fórmula (usa diferencia de meses)
perdida_real = (meses_obj_real - meses_obj_sim) * 7000 * 0.03 * ticket_promedio

# ====== Resultados ======
st.subheader("Resultados")
c1, c2 = st.columns(2)
c1.metric("🎯 Meses hasta objetivo (real)", f"{meses_obj_real:.2f}" if meses_obj_real != float("inf") else "—")
c2.metric("🎯 Meses hasta objetivo (simulada)", f"{meses_obj_sim:.2f}" if meses_obj_sim != float("inf") else "—")

c3, c4 = st.columns(2)
c3.metric("🔮 Score acumulado simulado", f"{nuevo_acumulado:.3f}")
c4.metric("💰 Impacto Económico", f"${perdida_real:,.0f}")

# ====== Gráfico ======
fig, ax = plt.subplots()
ax.scatter(X, y_acumulado, label="Datos reales")
ax.plot(X, model_real.predict(X), linestyle="--", label="Recta real")

ax.scatter(X_sim, y_sim, label="Datos simulados")
ax.plot(X_sim, model_sim.predict(X_sim), label="Recta simulada")

ax.axhline(y=objetivo, linestyle="--", label=f"Objetivo {objetivo}")
ax.legend()
ax.set_xlabel("Semana ISO")
ax.set_ylabel("Score acumulado")
ax.set_title("Predicción: real vs simulada - Dot")
st.pyplot(fig)



