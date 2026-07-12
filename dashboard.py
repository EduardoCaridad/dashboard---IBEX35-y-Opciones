"""
Dashboard IBEX 35 — Streamlit.

Capa visual sobre el motor de datos de main.py. Ejecutar con:
    streamlit run dashboard.py

Cadencia de refresco (respeta lo que diseñamos):
  - Precios: caché de 60 s (cambian rápido).
  - Objetivos y volatilidad: caché de 6 h (cambian despacio).

Requisitos:  pip install streamlit plotly  (además de lo de main.py)
"""

import datetime
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import QuantLib as ql
from streamlit_autorefresh import st_autorefresh

# Texto object de numpy en todo el proceso (evita el segfault de string[pyarrow]).
for _opt, _val in (("future.infer_string", False), ("mode.string_storage", "python")):
    try:
        pd.set_option(_opt, _val)
    except (KeyError, ValueError):
        pass

from main import (
    IBEX35, snapshot_ibex, objetivos_cacheados, volatilidades_historicas,
    tercer_viernes, malla_direccional, resumen_afinado, revalorizacion_strike,
    revalorizacion_fecha, sin_pyarrow, TASA_SIN_RIESGO, UMBRAL_ANALISTAS,
)

st.set_page_config(page_title="IBEX 35 · Revalorización de opciones", layout="wide")


# =================== ESTILO (tema terminal financiero) ===================

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --bg:#0a0e14; --panel:#121a24; --border:#1e2a38;
  --text:#f2f6fb; --muted:#aab4c4; --green:#00c896; --red:#ff5d73;
}

.stApp{ background:var(--bg); }
.block-container{ padding-top:2.2rem; padding-bottom:3rem; max-width:1320px; }
html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
h1,h2,h3,h4{ font-family:'Space Grotesk',sans-serif; letter-spacing:-.01em; color:var(--text); }

/* Cabecera */
.eyebrow{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.30em;
  text-transform:uppercase; color:var(--green); margin-bottom:.35rem; }
.app-title{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.95rem;
  color:var(--text); line-height:1.12; margin:0 0 .55rem 0; }
.statusbar{ font-family:'JetBrains Mono',monospace; font-size:.78rem; color:var(--muted);
  border-top:1px solid var(--border); border-bottom:1px solid var(--border);
  padding:.5rem 0; margin:.2rem 0 1.2rem 0; display:flex; gap:1.6rem; flex-wrap:wrap; }
.statusbar b{ color:var(--green); font-weight:600; }
.dot{ color:var(--green); }

/* Métricas como tarjetas */
[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--border);
  border-radius:12px; padding:14px 16px; transition:border-color .15s ease; }
[data-testid="stMetric"]:hover{ border-color:var(--green); }
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; font-weight:600; }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-size:.8rem; }

/* Tarjeta P&L a medida (para colorear por signo) */
.pnl-card{ background:var(--panel); border:1px solid var(--border); border-radius:12px;
  padding:14px 16px; }
.pnl-card .lbl{ color:var(--muted); font-size:.8rem; margin-bottom:.2rem; }
.pnl-card .val{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:1.6rem;
  line-height:1.1; }
.pnl-card .sub{ color:var(--muted); font-size:.72rem; margin-top:.25rem;
  font-family:'JetBrains Mono',monospace; }

/* Barra lateral */
[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--border); }
[data-testid="stSidebar"] h2{ font-size:1.05rem; color:var(--green);
  font-family:'JetBrains Mono',monospace; letter-spacing:.05em; }

/* Pestañas */
[data-baseweb="tab-list"]{ gap:.4rem; border-bottom:1px solid var(--border); }
[data-baseweb="tab"]{ font-family:'Inter',sans-serif; font-weight:500; }

/* Capciones */
[data-testid="stCaptionContainer"] p{ color:var(--muted); }

@media (prefers-reduced-motion: reduce){
  [data-testid="stMetric"]{ transition:none; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def pnl_card(col, label, valor, color="var(--text)", sub=""):
    """Tarjeta de métrica a medida, para colorear el número por signo (verde/rojo)."""
    col.markdown(
        f'<div class="pnl-card"><div class="lbl">{label}</div>'
        f'<div class="val" style="color:{color};">{valor}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


# =================== CAPA DE DATOS (cacheada) ===================

@st.cache_data(ttl=840, show_spinner="Actualizando precios...")
def _precios():
    return snapshot_ibex(IBEX35)[["Ticker", "Ultimo", "Var_%"]]

@st.cache_data(ttl=21600, show_spinner="Cargando objetivos...")
def _objetivos():
    return objetivos_cacheados(IBEX35)[
        ["Ticker", "Objetivo", "Obj_alto", "Obj_bajo",
         "Dividendo", "N_analistas", "Recomendacion"]]

@st.cache_data(ttl=21600, show_spinner="Calculando volatilidades...")
def _vols(ventana):
    return volatilidades_historicas(IBEX35, ventana)


def construir_base(ventana):
    # Elimina cualquier string[pyarrow] (pandas 3.0) ANTES de cualquier merge:
    # es lo que provocaba el segfault al unir/operar bajo los hilos de Streamlit.
    precios = sin_pyarrow(_precios())
    objet = sin_pyarrow(_objetivos())
    vols = _vols(ventana).copy()
    vols.index = vols.index.astype(object)

    base = precios.merge(objet, on="Ticker", how="left")
    base = base.merge(vols.rename("Vol"),
                      left_on="Ticker", right_index=True, how="left")
    base["Potencial_%"] = (base["Objetivo"] / base["Ultimo"] - 1) * 100
    return base


# --- TODO el código QuantLib corre en UN ÚNICO hilo dedicado ---
# QuantLib no es thread-safe y mantiene estado global (Settings.evaluationDate).
# Streamlit reejecuta el script en hilos distintos, así que si QuantLib se toca
# desde varios hilos, al destruir sus objetos globales entre reejecuciones peta
# (segfault). Solución: un hilo trabajador vivo toda la sesión que ejecuta cada
# tarea QuantLib. Así ese estado global solo lo toca ese hilo. Nunca hay cruce.

import queue
import threading


class _QLWorker:
    def __init__(self):
        self._q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            fn, args, kwargs, salida = self._q.get()
            try:
                res = fn(*args, **kwargs)
                gc.collect()                 # limpia ciclos QuantLib EN este hilo
                salida.put((True, res))
            except Exception as e:            # noqa: BLE001
                salida.put((False, e))

    def run(self, fn, *args, **kwargs):
        salida = queue.Queue()
        self._q.put((fn, args, kwargs, salida))
        ok, val = salida.get()
        if ok:
            return val
        raise val


@st.cache_resource
def _ql_worker():
    return _QLWorker()


# Tareas QuantLib: reciben SOLO primitivos y crean todos los ql.Date por dentro,
# de modo que la creación y destrucción ocurren íntegramente en el hilo trabajador.

def _tarea_malla(datos, anio, mes, niveles, r):
    base = pd.DataFrame(list(datos),
                        columns=["Ticker", "Ultimo", "Objetivo", "Vol", "Dividendo"])
    venc = tercer_viernes(int(anio), int(mes))
    return malla_direccional(base, venc, niveles, r=r)


def _tarea_reval_strike(spot, objetivo, strike, anio, mes, vol, q, r):
    venc = tercer_viernes(int(anio), int(mes))
    return revalorizacion_strike(spot, objetivo, strike, venc, vol, q=q, r=r)


def _tarea_reval_fecha(spot, precio_sim, strike, anio_v, mes_v, sim_iso, vol, q, r):
    venc = tercer_viernes(int(anio_v), int(mes_v))
    hoy = ql.Date.todaysDate()
    y, m, d = (int(x) for x in sim_iso.split("-"))
    fecha_sim = ql.Date(d, m, y)
    return revalorizacion_fecha(spot, precio_sim, strike, venc, fecha_sim,
                                vol, q=q, r=r, hoy=hoy)


@st.cache_data(show_spinner=False)
def _malla_cached(datos, anio, mes, niveles, r):
    """datos: tuple de filas (Ticker, Ultimo, Objetivo, Vol, Dividendo)."""
    return _ql_worker().run(_tarea_malla, datos, anio, mes, niveles, r)


@st.cache_data(show_spinner=False)
def _reval_strike_cached(spot, objetivo, strike, anio, mes, vol, q, r):
    return _ql_worker().run(_tarea_reval_strike, spot, objetivo, strike,
                            anio, mes, vol, q, r)


def tercer_viernes_py(anio, mes):
    """Tercer viernes del mes en Python puro (sin QuantLib), para los límites del
    selector de fecha. Evita crear objetos QuantLib fuera del hilo trabajador."""
    d = datetime.date(anio, mes, 1)
    primer_viernes = d + datetime.timedelta(days=(4 - d.weekday()) % 7)
    return primer_viernes + datetime.timedelta(days=14)


@st.cache_data(show_spinner=False)
def _reval_fecha_cached(spot, precio_sim, strike, anio_v, mes_v, sim_iso, vol, q, r):
    """sim_iso: fecha de simulación como 'YYYY-MM-DD'."""
    return _ql_worker().run(_tarea_reval_fecha, spot, precio_sim, strike,
                            anio_v, mes_v, sim_iso, vol, q, r)


# =================== FIGURAS (lógica pura) ===================

def figura_heatmap(malla, cap):
    num = malla.drop(index="Instrumento").astype(float)
    orden = num.loc["+0%"].sort_values(ascending=False).index
    num = num[orden]
    z = num.clip(lower=0, upper=cap)                     # color capeado, texto real
    escala = [[0.0, "#0e1620"], [0.25, "#0f4d3e"],
              [0.60, "#00a37a"], [1.0, "#00e6ad"]]       # oscuro -> verde: intensidad = ganancia
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(num.columns), y=list(num.index),
        text=num.values, texttemplate="%{text:+.0f}", textfont_size=9,
        textfont_color="#d8f5ec",
        colorscale=escala, zmin=0, zmax=cap,
        colorbar=dict(title=f"Reval %<br>(cap {cap})", outlinewidth=0,
                      tickfont=dict(family="JetBrains Mono")),
        hovertemplate="%{x} · %{y}<br>Reval: %{text:+.0f}%<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#e6ecf3"),
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=None, yaxis_title="Moneyness")
    return fig


def figura_curva(malla, ticker):
    num = malla.drop(index="Instrumento").astype(float)
    serie = num[ticker]
    instr = malla.loc["Instrumento", ticker]
    color = "#00c896" if instr == "CALL" else "#ff5d73"
    relleno = "rgba(0,200,150,0.08)" if instr == "CALL" else "rgba(255,93,115,0.08)"
    fig = go.Figure(go.Scatter(
        x=list(serie.index), y=serie.values, mode="lines+markers",
        line=dict(width=3, color=color),
        marker=dict(size=9, color=color, line=dict(width=0)),
        fill="tozeroy", fillcolor=relleno))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#e6ecf3"),
        title=dict(text=f"{ticker} · revalorización de la {instr} por moneyness",
                   font=dict(family="Space Grotesk")),
        xaxis_title="Moneyness (strike relativo al precio)",
        yaxis_title="Revalorización %", height=340,
        margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="#1e2a38", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2a38", zeroline=False)
    return fig


# =================== INTERFAZ ===================

# Reejecuta el script automáticamente cada 15 min (900 000 ms) para refrescar precios.
st_autorefresh(interval=15 * 60 * 1000, key="auto_refresh_15min")
_ahora = pd.Timestamp.now(tz="Europe/Madrid")

st.markdown(
    '<div class="eyebrow">Opciones · IBEX 35 · Precio objetivo de analistas</div>'
    '<div class="app-title">Revalorización de opciones</div>'
    '<div class="statusbar">'
    f'<span><span class="dot">●</span> En vivo · retardo ~15 min</span>'
    f'<span>Actualizado <b>{_ahora:%H:%M:%S}</b></span>'
    f'<span>Fuente <b>Yahoo Finance</b></span>'
    f'<span>Modelo <b>Black-Scholes · QuantLib</b></span>'
    '</div>',
    unsafe_allow_html=True)

# --- Controles ---
with st.sidebar:
    st.header("Parámetros")
    anio = st.number_input("Año de vencimiento", 2026, 2030, 2026,
                           key="anio_venc")
    mes  = st.selectbox("Mes de vencimiento", list(range(1, 13)), index=11,
                        format_func=lambda m: f"{m:02d}", key="mes_venc")
    ventana = st.slider("Ventana volatilidad (días)", 30, 252, 120, step=10,
                        key="ventana_vol")
    rango = st.slider("Rango de moneyness (±%)", 5, 25, 10, step=5,
                      key="rango_moneyness")
    paso  = st.select_slider("Paso de moneyness (%)", [2.5, 5.0], value=5.0,
                             key="paso_moneyness")
    umbral = st.slider("Umbral de analistas (marca *)", 2, 15, UMBRAL_ANALISTAS,
                       key="umbral_analistas")
    cap = st.slider("Tope de color del heatmap (%)", 100, 1000, 300, step=50,
                    key="cap_heatmap")
    st.divider()
    if st.button("🔄 Refrescar precios ahora", key="btn_refresh"):
        _precios.clear()
        st.rerun()

# --- Cálculo ---
niveles = tuple(round(x / 100, 4)
                for x in np.arange(-rango, rango + 0.001, paso))

try:
    base = construir_base(ventana)
    datos = tuple(base[["Ticker", "Ultimo", "Objetivo", "Vol", "Dividendo"]]
                  .itertuples(index=False, name=None))
    malla = _malla_cached(datos, int(anio), int(mes), niveles, TASA_SIN_RIESGO)
except Exception as e:
    st.error(f"Error al obtener datos: {e}")
    st.stop()

# --- Métricas rápidas ---
validos = base.dropna(subset=["Potencial_%"])
c1, c2, c3 = st.columns(3)
mejor = validos.loc[validos["Potencial_%"].idxmax()]
peor  = validos.loc[validos["Potencial_%"].idxmin()]
c1.metric("Mayor potencial", mejor["Ticker"], f"{mejor['Potencial_%']:+.1f}%")
c2.metric("Menor potencial", peor["Ticker"], f"{peor['Potencial_%']:+.1f}%")
c3.metric("Sin objetivo", int(base["Objetivo"].isna().sum()))

# --- Pestañas ---
tab_res, tab_heat, tab_det = st.tabs(["📋 Resumen", "🔥 Heatmap", "🔍 Detalle por valor"])

with tab_res:
    st.caption(f"* = objetivo poco fiable (menos de {umbral} analistas)")
    vista = resumen_afinado(base, malla, umbral_analistas=umbral)
    st.dataframe(
        vista, hide_index=True, width="stretch", height=560,
        column_config={
            "Fiab":        st.column_config.TextColumn("!", width="small"),
            "Ultimo":      st.column_config.NumberColumn("Último", format="%.3f"),
            "Objetivo":    st.column_config.NumberColumn(format="%.3f"),
            "Potencial_%": st.column_config.NumberColumn("Potencial", format="%+.1f%%"),
            "Disp_%":      st.column_config.NumberColumn("Dispersión", format="%.0f%%"),
            "N_analistas": st.column_config.NumberColumn("N. anal.", format="%d"),
            "Reval_ATM_%": st.column_config.NumberColumn("Reval ATM", format="%+.0f%%"),
        })

with tab_heat:
    st.caption("Revalorización de la opción (call/put según dirección) en cada "
               "moneyness. El color está capeado; el número es real.")
    st.plotly_chart(figura_heatmap(malla, cap), width="stretch")

with tab_det:
    tickers_ok = [c for c in malla.columns]
    sel = st.selectbox("Valor", tickers_ok, key="sel_valor")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(figura_curva(malla, sel), width="stretch")
    with col_b:
        fila = base[base["Ticker"] == sel].iloc[0]
        st.metric("Precio actual", f"{fila['Ultimo']:.3f} €")
        st.metric("Objetivo", f"{fila['Objetivo']:.3f} €",
                  f"{fila['Potencial_%']:+.1f}%")
        st.metric("Volatilidad", f"{fila['Vol']:.1%}")
        st.metric("Instrumento", malla.loc["Instrumento", sel])

    st.divider()
    st.subheader("Simulador de escenario")
    st.caption("Elige un strike, un precio simulado del subyacente y la fecha en que "
               "lo alcanzaría. Se valora la opción hoy y en esa fecha (mismo "
               "vencimiento): la diferencia incluye el paso del tiempo (theta).")

    hoy_py = datetime.date.today()
    venc_py = tercer_viernes_py(int(anio), int(mes))
    venc_valido = venc_py > hoy_py

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        paso_strike = round(max(0.05, float(fila["Ultimo"]) * 0.005), 2)
        strike = st.number_input(
            "Strike (€)", min_value=0.01,
            value=round(float(fila["Ultimo"]), 2), step=paso_strike, format="%.2f",
            key=f"strike_calc_{sel}",
            help="El paso se ajusta al precio (0,5%). También puedes escribir el valor.")
    with cc2:
        precio_sim = st.number_input(
            "Precio simulado (€)", min_value=0.01,
            value=round(float(fila["Objetivo"]), 2), step=paso_strike, format="%.2f",
            key=f"precio_sim_{sel}",
            help="Por defecto, el precio objetivo. Puedes probar cualquier valor.")
    with cc3:
        fecha_sim = st.date_input(
            "Fecha de simulación",
            value=hoy_py, min_value=hoy_py,
            max_value=venc_py if venc_valido else hoy_py,
            key=f"fecha_sim_{sel}",
            help=f"Entre hoy y el vencimiento ({venc_py:%d/%m/%Y}).")

    if not venc_valido:
        st.warning("El vencimiento elegido es hoy o está en el pasado. "
                   "Elige un mes/año de vencimiento futuro en la barra lateral.")
    else:
        res = _reval_fecha_cached(
            float(fila["Ultimo"]), float(precio_sim), float(strike),
            int(anio), int(mes), fecha_sim.isoformat(), float(fila["Vol"]),
            0.0 if pd.isna(fila["Dividendo"]) else float(fila["Dividendo"]),
            TASA_SIN_RIESGO)

        m1, m2, m3 = st.columns(3)
        pnl_card(m1, f"Valor {res['instrumento']} hoy",
                 f"{res['valor_hoy']:.4f} €",
                 sub=f"Precio actual {fila['Ultimo']:.3f} €")
        pnl_card(m2, f"Valor el {fecha_sim:%d/%m}",
                 f"{res['valor_sim']:.4f} €",
                 sub=f"{res['dias_sim_a_venc']} días al vencimiento")
        gana = res["reval_%"] >= 0
        pnl_card(m3, "Revalorización" if gana else "Pérdida",
                 f"{res['reval_%']:+.0f}%",
                 color="var(--green)" if gana else "var(--red)",
                 sub=f"Moneyness {res['moneyness_%']:+.1f}% · "
                     f"{res['dias_hasta_sim']} días a la fecha")
