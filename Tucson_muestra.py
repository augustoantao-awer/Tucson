import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos iniciales
X = np.array([31, 32, 33, 34, 35, 36, 37]).reshape(-1, 1)
y = np.array([4.104, 4.109, 4.112, 4.116, 4.121, 4.128, 4.131])

# Parámetros en la app
nuevo_valor = st.slider("Valor de la semana 38", 4.10, 4.20, 4.14, 0.001)
objetivo = st.number_input("Score objetivo", value=4.150)

# Recalcular con el nuevo dato
X_new = np.vstack([X, [38]])
y_new = np.append(y, nuevo_valor)

model = LinearRegression()
model.fit(X_new, y_new)

m = model.coef_[0]
b = model.intercept_
semana_obj = (objetivo - b) / m

# Mostrar resultados
st.write(f"📈 Pendiente: {m:.6f}")
st.write(f"🎯 Semana para llegar al objetivo {objetivo}: {semana_obj:.2f}")

# Gráfico
fig, ax = plt.subplots()
ax.scatter(X_new, y_new, color="blue", label="Datos reales + simulado")
ax.plot(X_new, model.predict(X_new), color="red", label="Predicción")
ax.axhline(y=objetivo, color="green", linestyle="--", label=f"Objetivo {objetivo}")
ax.legend()
st.pyplot(fig)
