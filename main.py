"""
Dashboard IBEX 35 — cotizaciones, precios objetivo y revalorización de opciones.

Todo se alimenta de Yahoo Finance (yfinance). El valor de las opciones se calcula
con QuantLib bajo el escenario B: "¿cuánto valdría la opción si la acción saltara
al precio objetivo de los analistas, hoy, con la misma volatilidad y vencimiento?"

Estructura:
  1. Configuración
  2. Cotizaciones actuales      (snapshot_ibex)
  3. Objetivos + dividendo      (objetivos_ibex + caché versionado)
  4. Volatilidad histórica      (volatilidades_historicas)
  5. Pricing de opciones        (tercer_viernes, valor_bs)
  6. Malla de revalorización    (malla_revalorizacion, malla_direccional)
  7. Resumen afinado            (resumen_afinado)
  8. Benchmarking               (DORMIDO)
  9. Main

Requisitos:  pip install yfinance pandas numpy QuantLib pyarrow
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import QuantLib as ql

# Evita las cadenas de texto respaldadas por pyarrow (string[pyarrow]). Ese tipo
# provoca segfaults al operar (merge, drop...) bajo los hilos de Streamlit. Con
# esto, pandas usa texto object de numpy en todo el proceso. Debe ir ANTES de
# crear cualquier DataFrame; main.py se importa el primero, así aplica a todo.
for _opt, _val in (("future.infer_string", False), ("mode.string_storage", "python")):
    try:
        pd.set_option(_opt, _val)
    except (KeyError, ValueError):
        pass


# ======================= 1. CONFIGURACIÓN =======================

IBEX35 = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "IDR.MC", "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC",
    "NTGY.MC", "PUIG.MC", "RED.MC", "REP.MC", "ROVI.MC", "SAB.MC",
    "SAN.MC", "SCYR.MC", "SLR.MC", "TEF.MC", "UNI.MC",
]

CACHE_FILE   = Path(__file__).parent / "cache_objetivos.pkl"
CACHE_HORAS  = 6                                   # validez del caché de objetivos

VENTANA_VOL     = 120                               # días para la volatilidad histórica
NIVELES_MALLA   = (-0.10, -0.05, 0.0, 0.05, 0.10)   # moneyness (strike relativo al spot)
TASA_SIN_RIESGO = 0.03                              # r (euro, aproximado)
UMBRAL_ANALISTAS = 5                                # menos de N analistas -> poco fiable
MES_VENCIMIENTO = (2026, 12)                        # (año, mes) del 3er viernes

EJECUTAR_BENCHMARK = False                          # ← True para medir tiempos

# columnas que el caché DEBE tener; si faltan, se regenera (evita KeyError por formato viejo)
COLUMNAS_ESPERADAS = {"Ticker", "Objetivo", "Obj_alto", "Obj_bajo",
                      "N_analistas", "Recomendacion", "Dividendo"}


# ==================== 2. COTIZACIONES ACTUALES ====================

def snapshot_ibex(tickers: list[str]) -> pd.DataFrame:
    """Último precio, cierre anterior y variación %. 1 petición. Precio CRUDO."""
    datos = yf.download(tickers, period="5d", interval="1m",
                        group_by="ticker", progress=False, auto_adjust=False,
                        threads=False)   # sin hilos internos: evita segfault bajo Streamlit

    if not isinstance(datos.columns, pd.MultiIndex):
        datos.columns = pd.MultiIndex.from_product([[tickers[0]], datos.columns])

    filas = []
    for t in tickers:
        cierres = datos[t]["Close"].dropna()
        sesiones = cierres.groupby(cierres.index.date).last()

        if len(sesiones) < 2:
            filas.append({"Ticker": t, "Ultimo": float("nan"),
                          "Cierre_ant": float("nan"), "Var_%": float("nan"),
                          "Hora": None})
            continue

        ultimo, anterior = sesiones.iloc[-1], sesiones.iloc[-2]
        filas.append({"Ticker": t, "Ultimo": ultimo, "Cierre_ant": anterior,
                      "Var_%": (ultimo / anterior - 1) * 100,
                      "Hora": cierres.index[-1].strftime("%H:%M")})

    return pd.DataFrame(filas)


# =============== 3. OBJETIVOS + DIVIDENDO (cacheado) ===============

def _norm_div(y) -> float:
    """yfinance devuelve el dividend yield de forma inconsistente: unas veces como
    fracción (0.04) y otras como porcentaje (4.0). Normalizamos a fracción."""
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return 0.0
    return y / 100 if y > 1 else y


def _datos_de_uno(ticker: str) -> dict:
    """Pide el .info de UN ticker y extrae objetivo, recomendación y dividendo.
    Nunca lanza excepción."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "Ticker":        ticker,
        "Objetivo":      info.get("targetMeanPrice"),
        "Obj_alto":      info.get("targetHighPrice"),
        "Obj_bajo":      info.get("targetLowPrice"),
        "N_analistas":   info.get("numberOfAnalystOpinions"),
        "Recomendacion": info.get("recommendationKey"),
        "Dividendo":     _norm_div(info.get("dividendYield")),
    }


def objetivos_ibex(tickers: list[str], max_hilos: int = 8) -> pd.DataFrame:
    """Objetivos + dividendo en paralelo (I/O-bound)."""
    filas = []
    with ThreadPoolExecutor(max_workers=max_hilos) as executor:
        futuros = {executor.submit(_datos_de_uno, t): t for t in tickers}
        for futuro in as_completed(futuros):
            filas.append(futuro.result())

    orden = {t: i for i, t in enumerate(tickers)}
    return pd.DataFrame(filas).sort_values(
        "Ticker", key=lambda col: col.map(orden)
    ).reset_index(drop=True)


def sin_pyarrow(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte etiquetas de columna, índice y columnas de texto a object de numpy,
    eliminando cualquier string[pyarrow] (por defecto en pandas 3.0). Ese tipo
    provoca segfaults al operar (merge, drop...) bajo los hilos de Streamlit."""
    df = df.copy()
    df.columns = df.columns.astype(object)
    if not (pd.api.types.is_numeric_dtype(df.index) or
            pd.api.types.is_datetime64_any_dtype(df.index)):
        df.index = df.index.astype(object)
    for c in df.columns:
        col = df[c]
        if not (pd.api.types.is_numeric_dtype(col) or
                pd.api.types.is_datetime64_any_dtype(col) or
                pd.api.types.is_bool_dtype(col)):
            df[c] = col.astype(object)
    return df


def objetivos_cacheados(tickers: list[str], forzar: bool = False) -> pd.DataFrame:
    if not forzar and CACHE_FILE.exists():
        edad_horas = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if edad_horas < CACHE_HORAS:
            df = sin_pyarrow(pd.read_pickle(CACHE_FILE))   # neutraliza caché viejo
            if COLUMNAS_ESPERADAS.issubset(df.columns):
                print(f"Caché de objetivos válido ({edad_horas:.1f} h). Reutilizando.")
                return df
            print("Caché con formato antiguo. Regenerando...")

    print("Pidiendo objetivos a Yahoo...")
    df = sin_pyarrow(objetivos_ibex(tickers))
    df.to_pickle(CACHE_FILE)
    return df


# =================== 4. VOLATILIDAD HISTÓRICA ===================

def volatilidades_historicas(tickers: list[str], ventana: int = VENTANA_VOL) -> pd.Series:
    """Volatilidad anualizada por valor, de 1 año de cierres diarios.
    auto_adjust=True: para volatilidad queremos precios AJUSTADOS (un split crearía
    un salto falso que dispararía la vol). Es la decisión opuesta al snapshot."""
    datos = yf.download(tickers, period="1y", interval="1d",
                        group_by="ticker", progress=False, auto_adjust=True,
                        threads=False)   # sin hilos internos: evita segfault bajo Streamlit

    if not isinstance(datos.columns, pd.MultiIndex):
        datos.columns = pd.MultiIndex.from_product([[tickers[0]], datos.columns])

    vols = {}
    for t in tickers:
        cierres = datos[t]["Close"].dropna()
        rend = np.log(cierres / cierres.shift(1)).dropna()
        if len(rend) < 20:
            vols[t] = float("nan")
        else:
            vols[t] = rend.iloc[-ventana:].std() * np.sqrt(252)
    return pd.Series(vols, name="Vol")


# =================== 5. PRICING DE OPCIONES ===================

def tercer_viernes(anio: int, mes: int) -> ql.Date:
    """Tercer viernes del mes: vencimiento estándar de MEFF."""
    return ql.Date.nthWeekday(3, ql.Friday, mes, anio)


def valor_bs(spot: float, strike: float, venc: ql.Date, vol: float,
             r: float = TASA_SIN_RIESGO, q: float = 0.0,
             tipo=ql.Option.Call, hoy: ql.Date = None) -> float:
    """Precio Black-Scholes-Merton de una opción europea, HOY.
    Escenario B: pasa spot = precio_objetivo para el valor hipotético."""
    hoy = hoy or ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = hoy

    opcion = ql.VanillaOption(ql.PlainVanillaPayoff(tipo, strike),
                              ql.EuropeanExercise(venc))
    dc, cal = ql.Actual365Fixed(), ql.TARGET()
    proceso = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(hoy, q, dc)),   # dividendos
        ql.YieldTermStructureHandle(ql.FlatForward(hoy, r, dc)),   # tipo sin riesgo
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(hoy, cal, vol, dc)))
    opcion.setPricingEngine(ql.AnalyticEuropeanEngine(proceso))
    return opcion.NPV()


def revalorizacion_strike(spot: float, objetivo: float, strike: float,
                          venc: ql.Date, vol: float, q: float = 0.0,
                          r: float = TASA_SIN_RIESGO, hoy: ql.Date = None) -> dict:
    """Revalorización de la opción para un strike EXACTO en euros (no relativo).
    Elige call/put según la dirección del potencial. Devuelve un dict con el
    instrumento, el moneyness resultante y los dos valores + la revalorización."""
    tipo = ql.Option.Call if objetivo >= spot else ql.Option.Put
    v_hoy = valor_bs(spot,     strike, venc, vol, r, q, tipo, hoy)
    v_esc = valor_bs(objetivo, strike, venc, vol, r, q, tipo, hoy)
    reval = (v_esc / v_hoy - 1) * 100 if v_hoy > 1e-6 else float("nan")
    return {"instrumento": "CALL" if tipo == ql.Option.Call else "PUT",
            "moneyness_%": (strike / spot - 1) * 100,
            "valor_hoy": v_hoy, "valor_escenario": v_esc, "reval_%": reval}


def revalorizacion_fecha(spot: float, precio_sim: float, strike: float,
                         venc: ql.Date, fecha_sim: ql.Date, vol: float,
                         q: float = 0.0, r: float = TASA_SIN_RIESGO,
                         hoy: ql.Date = None) -> dict:
    """Valora la opción HOY (spot actual) y en una FECHA FUTURA (fecha_sim) suponiendo
    que el subyacente estará en precio_sim, con el MISMO vencimiento. Como en fecha_sim
    queda menos tiempo, captura la pérdida de valor temporal (theta). La revalorización
    puede ser negativa (pérdida). Elige call/put según la dirección de precio_sim."""
    hoy = hoy or ql.Date.todaysDate()
    tipo = ql.Option.Call if precio_sim >= spot else ql.Option.Put
    v_hoy = valor_bs(spot, strike, venc, vol, r, q, tipo, hoy=hoy)

    if fecha_sim >= venc:                       # a vencimiento: valor = intrínseco puro
        v_sim = (max(precio_sim - strike, 0.0) if tipo == ql.Option.Call
                 else max(strike - precio_sim, 0.0))
    else:
        v_sim = valor_bs(precio_sim, strike, venc, vol, r, q, tipo, hoy=fecha_sim)

    reval = (v_sim / v_hoy - 1) * 100 if v_hoy > 1e-6 else float("nan")
    return {"instrumento": "CALL" if tipo == ql.Option.Call else "PUT",
            "moneyness_%": (strike / spot - 1) * 100,
            "valor_hoy": v_hoy, "valor_sim": v_sim, "reval_%": reval,
            "dias_hasta_sim": int(fecha_sim - hoy),
            "dias_sim_a_venc": max(int(venc - fecha_sim), 0)}


# ================= 6. MALLA DE REVALORIZACIÓN =================

def malla_revalorizacion(base: pd.DataFrame, venc: ql.Date,
                         niveles=NIVELES_MALLA, tipo=ql.Option.Call,
                         r: float = TASA_SIN_RIESGO, hoy: ql.Date = None) -> pd.DataFrame:
    """Malla 2D: filas = moneyness, columnas = tickers, celdas = revalorización %.
    base necesita columnas Ticker, Ultimo (spot), Objetivo, Vol, Dividendo."""
    matriz = {}
    for row in base.itertuples():
        spot, obj, vol = row.Ultimo, row.Objetivo, row.Vol
        q = 0.0 if pd.isna(row.Dividendo) else row.Dividendo
        if any(pd.isna(x) for x in (spot, obj, vol)):
            continue
        col = {}
        for m in niveles:
            strike = spot * (1 + m)
            v_hoy = valor_bs(spot, strike, venc, vol, r, q, tipo, hoy)
            v_esc = valor_bs(obj,  strike, venc, vol, r, q, tipo, hoy)
            col[f"{m:+.0%}"] = (v_esc / v_hoy - 1) * 100 if v_hoy > 1e-6 else float("nan")
        matriz[row.Ticker] = pd.Series(col)
    return pd.DataFrame(matriz)


def malla_direccional(base: pd.DataFrame, venc: ql.Date,
                      niveles=NIVELES_MALLA, r: float = TASA_SIN_RIESGO,
                      hoy: ql.Date = None) -> pd.DataFrame:
    """Como malla_revalorizacion, pero por cada valor elige el instrumento que
    corresponde al signo del potencial: CALL si objetivo>=spot, PUT si no.
    Añade una fila 'Instrumento' informativa."""
    matriz, instrumento = {}, {}
    for row in base.itertuples():
        spot, obj, vol = row.Ultimo, row.Objetivo, row.Vol
        q = 0.0 if pd.isna(row.Dividendo) else row.Dividendo
        if any(pd.isna(x) for x in (spot, obj, vol)):
            continue
        tipo = ql.Option.Call if obj >= spot else ql.Option.Put
        instrumento[row.Ticker] = "CALL" if tipo == ql.Option.Call else "PUT"
        col = {}
        for m in niveles:
            strike = spot * (1 + m)
            v_hoy = valor_bs(spot, strike, venc, vol, r, q, tipo, hoy)
            v_esc = valor_bs(obj,  strike, venc, vol, r, q, tipo, hoy)
            col[f"{m:+.0%}"] = (v_esc / v_hoy - 1) * 100 if v_hoy > 1e-6 else float("nan")
        matriz[row.Ticker] = pd.Series(col)

    df = pd.DataFrame(matriz)
    df.loc["Instrumento"] = pd.Series(instrumento)
    return df


# ================= 7. RESUMEN AFINADO =================

def resumen_afinado(base: pd.DataFrame, malla: pd.DataFrame,
                    umbral_analistas: int = UMBRAL_ANALISTAS) -> pd.DataFrame:
    """Tabla por valor con potencial, dispersión del objetivo, nº de analistas,
    instrumento y revalorización ATM de la opción. Marca con '*' los objetivos
    poco fiables (menos de `umbral_analistas` analistas).
    Recibe la malla ya calculada para no repetir el pricing."""
    reval_atm   = pd.to_numeric(malla.loc["+0%"], errors="coerce")
    instrumento = malla.loc["Instrumento"]

    df = base.copy()
    df["Disp_%"]      = (df["Obj_alto"] - df["Obj_bajo"]) / df["Objetivo"] * 100
    df["Reval_ATM_%"] = df["Ticker"].map(reval_atm)
    df["Instr"]       = df["Ticker"].map(instrumento)
    df["Fiab"]        = df["N_analistas"].apply(
        lambda n: "*" if pd.notna(n) and n < umbral_analistas else "")

    cols = ["Fiab", "Ticker", "Ultimo", "Objetivo", "Potencial_%", "Disp_%",
            "N_analistas", "Instr", "Reval_ATM_%", "Recomendacion"]
    return df[cols].sort_values("Potencial_%", ascending=False)


# ================= 8. BENCHMARKING (DORMIDO) =================
# No se ejecuta en el arranque normal. Pon EJECUTAR_BENCHMARK = True para medir.

def _medir(nombre, tickers, hilos):
    inicio = time.perf_counter()
    df = objetivos_ibex(tickers, max_hilos=hilos)
    seg = time.perf_counter() - inicio
    huecos = df["Objetivo"].isna().sum()
    print(f"{nombre:20s} {seg:6.2f} s   ({huecos} huecos)")
    return seg


def ejecutar_benchmark(tickers):
    print("Comparando secuencial vs concurrente...\n")
    t_seq = _medir("Secuencial (1 hilo)", tickers, hilos=1)
    t_con = _medir("Concurrente (8)",     tickers, hilos=8)
    print(f"\nAceleración: {t_seq / t_con:.1f}× más rápido\n")


# ========================= 9. MAIN =========================

def construir_base(tickers: list[str]) -> pd.DataFrame:
    """Ensambla spot + objetivo + dividendo + volatilidad en una sola tabla."""
    precios = snapshot_ibex(tickers)[["Ticker", "Ultimo", "Var_%"]]
    objet   = objetivos_cacheados(tickers)[
        ["Ticker", "Objetivo", "Obj_alto", "Obj_bajo",
         "Dividendo", "N_analistas", "Recomendacion"]]
    vols    = volatilidades_historicas(tickers)

    base = precios.merge(objet, on="Ticker", how="left")
    base = base.merge(vols.rename("Vol"), left_on="Ticker", right_index=True, how="left")
    base["Potencial_%"] = (base["Objetivo"] / base["Ultimo"] - 1) * 100
    return base


def main():
    if EJECUTAR_BENCHMARK:
        ejecutar_benchmark(IBEX35)

    base = construir_base(IBEX35)
    venc = tercer_viernes(*MES_VENCIMIENTO)

    # La malla se calcula UNA vez y se reutiliza para el resumen y para imprimirla.
    malla = malla_direccional(base, venc)

    # --- Tabla resumen por valor ---
    print("\n=== RESUMEN POR VALOR ===")
    print("(* = objetivo poco fiable, menos de "
          f"{UMBRAL_ANALISTAS} analistas)\n")
    vista = resumen_afinado(base, malla)
    print(vista.to_string(
        index=False, na_rep="—",
        formatters={
            "Ultimo":      "{:8.3f}".format,
            "Objetivo":    "{:8.3f}".format,
            "Potencial_%": "{:+6.1f}%".format,
            "Disp_%":      "{:6.1f}%".format,
            "N_analistas": "{:.0f}".format,
            "Reval_ATM_%": "{:+6.0f}%".format,
        }))

    # --- Malla de revalorización de opciones ---
    print(f"\n=== MALLA DE REVALORIZACIÓN (venc. {venc}) ===")
    print("Instrumento elegido por dirección del potencial (call/put):\n")
    cols_ord = pd.to_numeric(malla.loc["+0%"], errors="coerce")\
                 .sort_values(ascending=False).index
    print(malla[cols_ord].to_string(
        float_format=lambda x: f"{x:+.0f}"))

    faltan = base["Objetivo"].isna().sum()
    if faltan:
        print(f"\n{faltan} valores sin objetivo (omitidos en la malla).")


if __name__ == "__main__":
    main()