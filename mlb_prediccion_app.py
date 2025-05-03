
import streamlit as st
import pandas as pd

def normalizar(valores, inverso=False):
    min_val, max_val = min(valores), max(valores)
    if max_val == min_val:
        return [0.5, 0.5]
    return [(max_val - v) / (max_val - min_val) if inverso else (v - min_val) / (max_val - min_val) for v in valores]

pesos = {
    "Promedio de bateo": 0.15,
    "Carreras por juego": 0.20,
    "OPS (slugging + OBP)": 0.20,
    "ERA del bullpen": 0.15,
    "WHIP del bullpen": 0.10,
    "Porcentaje de victorias": 0.15,
    "Errores defensivos por juego": 0.05
}

st.title("Predicción de Ganadores MLB + Apuestas con Valor")

archivo = st.file_uploader("Sube tu archivo Excel o CSV con estadísticas", type=["xlsx", "csv"])

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    partidos = df["Partido"].unique()
    resultados = []

    for partido in partidos:
        subset = df[df["Partido"] == partido]
        puntajes = {"Equipo A": 0, "Equipo B": 0}
        for _, row in subset.iterrows():
            estad = row["Estadística"]
            valores = [row["Equipo A"], row["Equipo B"]]
            inverso = "ERA" in estad or "WHIP" in estad or "Errores" in estad
            norm = normalizar(valores, inverso)
            peso = pesos.get(estad, 0.10)
            puntajes["Equipo A"] += norm[0] * peso
            puntajes["Equipo B"] += norm[1] * peso
        ganador = "Equipo A" if puntajes["Equipo A"] > puntajes["Equipo B"] else "Equipo B"
        resultados.append({
            "Partido": partido,
            "Ganador Probable": ganador,
            "Puntaje A": round(puntajes["Equipo A"], 3),
            "Puntaje B": round(puntajes["Equipo B"], 3)
        })

    df_resultado = pd.DataFrame(resultados)

    # Añadir análisis de valor esperado si hay cuotas
    if "Cuota A" in df.columns and "Cuota B" in df.columns:
        valores = []
        for i, row in df_resultado.iterrows():
            partido = row["Partido"]
            cuota_a = df[df["Partido"] == partido]["Cuota A"].dropna().values
            cuota_b = df[df["Partido"] == partido]["Cuota B"].dropna().values
            if cuota_a.size == 0 or cuota_b.size == 0:
                valores.append("Sin cuotas")
                continue
            cuota_a, cuota_b = cuota_a[0], cuota_b[0]
            prob_a_modelo = row["Puntaje A"]
            prob_b_modelo = row["Puntaje B"]
            prob_a_cuota = 1 / cuota_a
            prob_b_cuota = 1 / cuota_b
            value_a = prob_a_modelo - prob_a_cuota
            value_b = prob_b_modelo - prob_b_cuota
            if value_a > 0.05:
                valores.append("Valor en A")
            elif value_b > 0.05:
                valores.append("Valor en B")
            else:
                valores.append("Sin valor claro")
        df_resultado["Apuesta con Valor"] = valores

    st.subheader("Resultados de Predicción")
    st.dataframe(df_resultado)
