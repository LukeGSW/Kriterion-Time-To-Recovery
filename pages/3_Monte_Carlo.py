"""
pages/3_Monte_Carlo.py — Simulazione Monte Carlo TTR | Kriterion Quant

Dato un drawdown in corso (o ipotetico), stima la distribuzione empirica
dei tempi di recovery attesi via bootstrap sugli episodi storici.
Usa i dati calcolati nella pagina principale (session_state) o
permette di specificare un nuovo ticker.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_fetcher import fetch_full_history
from src.analytics    import (
    calculate_ttr_episodes,
    compute_summary_stats,
    fit_kaplan_meier,
    simulate_ttr_montecarlo,
)
from src.charts import build_montecarlo_chart, COLORS

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================

st.set_page_config(
    page_title="Monte Carlo TTR | Kriterion Quant",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# API KEY
# ============================================================

try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    st.error("❌ EODHD_API_KEY non trovata. Configura i secrets.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎲 Monte Carlo TTR")
    st.caption("Kriterion Quant — Simulazione Forward-Looking")
    st.divider()

    # Controlla se ci sono dati dalla pagina principale
    has_session_data = (
        "ticker"       in st.session_state and
        "episodes_df"  in st.session_state and
        "price_series" in st.session_state and
        not st.session_state["episodes_df"].empty
    )

    if has_session_data:
        session_ticker = st.session_state["ticker"]
        st.success(f"✅ Dati disponibili da:\n`{session_ticker}`")
        use_session = st.radio(
            "Fonte dati",
            options=[f"Usa {session_ticker} (già caricato)", "Nuovo ticker"],
            index=0,
        )
        load_new = use_session.startswith("Nuovo")
    else:
        st.info("ℹ️ Nessun dato in sessione.\nSpecifica un ticker da analizzare.")
        load_new = True

    if load_new:
        new_ticker = st.text_input("Ticker EODHD", value="SPY.US").strip().upper()
        min_depth_slider = st.slider(
            "Profondità minima (%)",
            min_value=-50, max_value=0, value=-5, step=1,
        )
        min_depth_pct = min_depth_slider / 100
        load_btn = st.button("📡 Carica Dati", type="primary", use_container_width=True)
    else:
        new_ticker = None
        load_btn   = False

    st.divider()

    st.subheader("⚙️ Parametri Simulazione")
    current_depth_slider = st.slider(
        "Drawdown corrente / ipotetico (%)",
        min_value=-80, max_value=-1, value=-15, step=1,
        help=(
            "Profondità del drawdown corrente o di uno scenario ipotetico "
            "di cui vuoi stimare il TTR atteso. "
            "La simulazione usa gli episodi storici con profondità ≥ a questo valore."
        ),
    )
    current_depth_pct = current_depth_slider / 100

    n_simulations = st.select_slider(
        "N° simulazioni bootstrap",
        options=[1_000, 5_000, 10_000, 50_000],
        value=10_000,
        help="Più simulazioni = distribuzione più precisa, ma calcolo più lento.",
    )

    st.divider()

    with st.expander("📖 Sintassi Ticker EODHD"):
        st.markdown("""
**ETF USA:** `SPY.US` · `QQQ.US`
**Crypto:** `BTC-USD.CC` · `ETH-USD.CC`
**Indici:** `GSPC.INDX` · `NDX.INDX`
        """)

# ============================================================
# HEADER
# ============================================================

st.title("🎲 Monte Carlo — Distribuzione del TTR Atteso")
st.markdown("""
Data una profondità di drawdown corrente (o ipotetica), questa sezione
stima via **bootstrap empirico** la distribuzione dei tempi di recovery attesi.

**Metodologia:** si selezionano gli episodi storici con profondità almeno uguale
a quella in input (comparabili o peggiori), poi si campiona con reinserimento
dai loro TTR per N simulazioni. Il risultato è una distribuzione empirica
dei possibili tempi di recupero — senza assumere distribuzioni parametriche,
il che la rende robusta per asset con code grasse (crypto, indici in crisi).

> **Uso operativo:** fornisce una risposta quantitativa alla domanda
> *"Quanto potrebbe durare questo drawdown, se paragonabile a quelli storici?"*
""")
st.divider()

# ============================================================
# CARICAMENTO DATI
# ============================================================

# Decide quali dati usare
if has_session_data and not load_new:
    ticker       = st.session_state["ticker"]
    episodes_df  = st.session_state["episodes_df"]
    price_series = st.session_state["price_series"]
    summary_stats = st.session_state.get("summary_stats", {})
    data_loaded  = True

elif load_new and load_btn and new_ticker:
    with st.spinner(f"⏳ Scaricando storia completa di `{new_ticker}`..."):
        try:
            df = fetch_full_history(new_ticker, EODHD_API_KEY)
        except Exception as e:
            st.error(f"❌ Errore caricamento `{new_ticker}`: {e}")
            st.stop()

    if df.empty:
        st.error(f"❌ Nessun dato per `{new_ticker}`. Verifica il ticker.")
        st.stop()

    price_series = df["adjusted_close"].dropna()
    episodes_df  = calculate_ttr_episodes(price_series, min_depth_pct)
    summary_stats = compute_summary_stats(episodes_df, price_series) if not episodes_df.empty else {}
    ticker       = new_ticker
    data_loaded  = True

    if episodes_df.empty:
        st.warning(
            f"⚠️ Nessun episodio trovato per `{new_ticker}` "
            f"con soglia {min_depth_slider}%. Abbassa la soglia."
        )
        st.stop()

else:
    if load_new and not load_btn:
        st.info("👈 Specifica il ticker nella sidebar e premi **Carica Dati**.")
    elif not has_session_data and not load_new:
        st.info("👈 Vai alla pagina principale, analizza un asset, poi torna qui.")
    st.stop()

# ============================================================
# CONTESTO DATI DISPONIBILI
# ============================================================

n_tot  = len(episodes_df)
n_comp = len(episodes_df[episodes_df["depth_pct"] <= current_depth_pct])

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Asset Analizzato", ticker)
col_info2.metric("Episodi Totali", f"{n_tot}")
col_info3.metric("Episodi Comparabili", f"{n_comp}",
                 help=f"Episodi con profondità ≥ {current_depth_slider}%")
col_info4.metric("Drawdown Simulato", f"{current_depth_slider}%")

if n_comp < 5:
    st.warning(
        f"⚠️ Solo **{n_comp}** episodi comparabili (profondità ≥ {current_depth_slider}%). "
        "La simulazione usa l'intero campione storico come fallback. "
        "I risultati sono meno specifici per questa profondità."
    )

st.divider()

# ============================================================
# SIMULAZIONE MONTE CARLO
# ============================================================

with st.spinner(f"🎲 Eseguendo {n_simulations:,} simulazioni bootstrap..."):
    mc_results = simulate_ttr_montecarlo(
        episodes_df=episodes_df,
        current_depth_pct=current_depth_pct,
        n_simulations=n_simulations,
    )

if not mc_results:
    st.error("❌ Simulazione fallita. Dati insufficienti.")
    st.stop()

pct     = mc_results["percentiles"]
prob    = mc_results["prob_recovery_by_days"]
fallbk  = mc_results["used_fallback"]

if fallbk:
    st.caption(
        "ℹ️ _Fallback attivo_: episodi comparabili insufficienti. "
        "Usato l'intero campione storico per la simulazione."
    )

# ============================================================
# SEZIONE 1: KPI SIMULAZIONE
# ============================================================

st.subheader("📊 Stima Distribuzione TTR Atteso")
st.markdown(f"""
Sulla base di **{mc_results['n_comparable']:,} episodi comparabili**
(profondità ≥ {current_depth_slider}%) e **{n_simulations:,} simulazioni bootstrap**:
""")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("TTR Mediano (P50)", f"{pct['p50']} gg",
          help="Il 50% dei drawdown simili si risolve entro questo numero di giorni.")
c2.metric("TTR P25", f"{pct['p25']} gg",
          help="Scenario ottimistico: il 25% dei recovery avviene entro questa stima.")
c3.metric("TTR P75", f"{pct['p75']} gg",
          help="Il 75% dei drawdown si risolve entro questo numero di giorni.")
c4.metric("TTR P90", f"{pct['p90']} gg",
          help="Solo il 10% dei drawdown dura più di questo valore.")
c5.metric("Media Simulata", f"{pct['media']:.0f} gg",
          help="Media delle simulazioni (distorta verso l'alto dai casi estremi).")

st.divider()

# ============================================================
# SEZIONE 2: GRAFICO MONTE CARLO
# ============================================================

st.subheader("📈 Distribuzione Simulata e Curva di Recovery Cumulativa")
st.markdown(f"""
**Pannello superiore — Istogramma:** distribuzione dei TTR simulati.
Le linee verticali marcano i percentili chiave: P25 (verde), P50 (arancio), P75 (rosso), P90 (ciano).

**Pannello inferiore — Curva CDF:** probabilità cumulativa di recovery entro N giorni.
Leggi: *"c'è il X% di probabilità che il drawdown da {current_depth_slider}% si risolva entro Y giorni"*.
""")

fig_mc = build_montecarlo_chart(mc_results)
st.plotly_chart(fig_mc, use_container_width=True)

st.divider()

# ============================================================
# SEZIONE 3: TABELLA PROBABILITÀ
# ============================================================

st.subheader("📋 Probabilità di Recovery entro N Giorni di Borsa")
st.markdown("""
Tabella di riferimento rapido per la valutazione operativa del rischio.
Ogni riga indica la probabilità empirica (basata sui dati storici dell'asset)
di aver recuperato il picco entro quel numero di giorni di borsa.
""")

# Costruzione tabella leggibile
checkpoint_labels = {
    "10":  "10 gg (~2 settimane)",
    "21":  "21 gg (~1 mese)",
    "63":  "63 gg (~3 mesi)",
    "126": "126 gg (~6 mesi)",
    "252": "252 gg (~1 anno)",
    "504": "504 gg (~2 anni)",
}

prob_rows = []
for key, label in checkpoint_labels.items():
    p = prob.get(key, 0)
    prob_rows.append({
        "Orizzonte Temporale":    label,
        "Prob. Recovery (%)":     round(p, 1),
        "Prob. Ancora in DD (%)": round(100 - p, 1),
    })

prob_df = pd.DataFrame(prob_rows)
styled_prob = (
    prob_df.style
    .background_gradient(cmap="RdYlGn", subset=["Prob. Recovery (%)"])
    .background_gradient(cmap="RdYlGn_r", subset=["Prob. Ancora in DD (%)"])
    .format({
        "Prob. Recovery (%)":     "{:.1f}%",
        "Prob. Ancora in DD (%)": "{:.1f}%",
    })
    .set_properties(**{"text-align": "center"})
)
st.dataframe(styled_prob, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# SEZIONE 4: EPISODI STORICI COMPARABILI (RIFERIMENTO)
# ============================================================

st.subheader("🔍 Episodi Storici Usati come Riferimento")
st.markdown(f"""
Questi sono gli episodi storici di `{ticker}` con profondità **≥ {current_depth_slider}%**
che hanno alimentato la simulazione. Ogni episodio rappresenta un precedente storico
rilevante per il drawdown in analisi.
""")

comparable_eps = episodes_df[episodes_df["depth_pct"] <= current_depth_pct].copy()

if comparable_eps.empty:
    st.info(
        f"ℹ️ Nessun episodio storico con profondità ≥ {current_depth_slider}%. "
        "La simulazione ha usato l'intero campione."
    )
else:
    # Formattazione display
    disp = comparable_eps.copy()
    disp["peak_date"]     = disp["peak_date"].dt.strftime("%Y-%m-%d")
    disp["trough_date"]   = disp["trough_date"].dt.strftime("%Y-%m-%d")
    disp["recovery_date"] = disp["recovery_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "⏳ In corso"
    )
    disp["depth_pct"]   = (disp["depth_pct"] * 100).round(2).astype(str) + "%"
    disp["is_censored"] = disp["is_censored"].map({True: "⏳ Sì", False: "✅ No"})
    disp = disp.rename(columns={
        "peak_date":     "Data Picco",
        "trough_date":   "Data Minimo",
        "recovery_date": "Data Recovery",
        "depth_pct":     "Profondità",
        "ttr_days":      "TTR (gg)",
        "is_censored":   "Censurato",
        "depth_bin":     "Classe",
    })

    st.dataframe(
        disp[["Data Picco", "Data Minimo", "Data Recovery",
              "Profondità", "TTR (gg)", "Censurato", "Classe"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"**{len(comparable_eps)}** episodi comparabili su {n_tot} totali.")

st.divider()

# ============================================================
# NOTE METODOLOGICHE
# ============================================================

with st.expander("ℹ️ Note Metodologiche e Limitazioni"):
    st.markdown(f"""
    **Metodologia Bootstrap Empirico (Stratificato)**

    1. **Selezione pool:** episodi storici con `depth_pct ≤ {current_depth_slider/100:.2f}`.
       Se < 5 episodi, fallback all'intero campione.
    2. **Bootstrap:** campionamento con reinserimento (`n = {n_simulations:,}`).
       Ogni simulazione estrae un TTR dal pool storico.
    3. **Distribuzione empirica:** i percentili e la CDF sono calcolati
       direttamente dai {n_simulations:,} valori simulati.

    **Assunzioni e Limitazioni:**
    - La distribuzione storica del TTR è rappresentativa del futuro.
    - I drawdown comparabili per profondità lo sono anche per contesto macroeconomico.
    - Campioni piccoli (< 10 episodi comparabili) producono stime meno affidabili.
    - I drawdown censurati (ancora aperti) contribuiscono al pool
      con il loro TTR parziale: questo tende a *sottostimare* il TTR mediano.
    - **Non è una previsione:** è una distribuzione di possibilità basata
      sui precedenti storici dello specifico asset.
    """)

st.caption(
    f"Kriterion Quant · Monte Carlo TTR · "
    f"Asset: {ticker} · "
    f"Simulazioni: {n_simulations:,} · "
    f"Drawdown simulato: {current_depth_slider}%"
)
