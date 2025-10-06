import pandas as pd
import streamlit as st
import numpy as np


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


#%%
st.title("📊 Simulador de Score - Dot ")

st.subheader("         🎯 Score objetivo: 4.3       ")

score_promedio = float(df["score"].mean(skipna=True))
reviews_promedio = int(df["cant_reviews"].mean(skipna=True))

cantidad=df["cant_reviews"].sum() + 3690


ultima_semana = int(df["semana_iso"].max())
semana_nueva = ultima_semana + 1





# ====== Resultados ======
# ====== Resultados ======
st.subheader("Resultados")

# ---- Métricas y simuladores ----
col_11, col_22 = st.columns(2)
col_11.metric("📈 Score promedio semanal", f"{score_promedio:.2f}")
with col_22: 
    nuevo_score = st.number_input(
        "Score promedio (simulado)",
        min_value=3.5,
        max_value=5.0,
        value=float(score_promedio if not np.isnan(score_promedio) else 4.2),
        step=0.01
    )

col_111, col_222 = st.columns(2)
col_111.metric("📈 Cantidad de reviews promedio", f"{reviews_promedio:.0f}")
with col_222: 
    nuevas_reviews = st.number_input(
        "Cantidad de reviews (simulado)",
        min_value=1,
        value=max(1, int(reviews_promedio))
    )





objetivo = 4.3


# ---- Simulación semana a semana ----
cantidad = df["cant_reviews"].sum()  + 3690  # total acumulado hasta última semana
score_acumulado = df["score_acumulado"].iloc[-1]
semana_sim = ultima_semana

lista = []

while score_acumulado <= objetivo:
    
    score_acumulado = (score_acumulado *cantidad + nuevas_reviews * nuevo_score) / (cantidad+nuevas_reviews)
    cantidad += nuevas_reviews
    semana_sim += 1
    lista.append(score_acumulado)
    
    
semana_objetivo_simulada = (semana_sim - ultima_semana) / 4.345




score_acumulado_real = df["score_acumulado"].iloc[-1]

cantidad_real = df["cant_reviews"].sum()  + 3690  # total acumulado hasta última semana
score_acumulado_real = df["score_acumulado"].iloc[-1]
semana_sim_real = ultima_semana



while score_acumulado_real <= objetivo:
    
    score_acumulado_real = (score_acumulado_real *cantidad_real + score_promedio * reviews_promedio) / (cantidad_real +reviews_promedio)
    cantidad_real += reviews_promedio
    semana_sim_real += 1
    lista.append(score_acumulado_real)
    
    
semana_objetivo_real = (semana_sim_real - ultima_semana) / 4.345




# ---- Mostrar resultados ----
c1, c2 = st.columns(2)
c1.metric("🎯 Meses hasta objetivo (real)", f"{semana_objetivo_real:.2f}" if semana_objetivo_real != float("inf") else "—")
c2.metric("🎯 Meses hasta objetivo (simulado)", f"{semana_objetivo_simulada:.2f}" if semana_objetivo_simulada != float("inf") else "—")

ticket_promedio = st.number_input("💵 Ticket promedio", value=30000, step=1000)

Ganancia = (semana_objetivo_real - semana_objetivo_simulada) * 7000 * 0.03 * ticket_promedio

st.metric("💰 Ganancia simulada vs real", f"${Ganancia:,.0f}")





