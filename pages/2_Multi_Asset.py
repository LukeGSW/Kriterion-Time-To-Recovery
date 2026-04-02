"""
pages/2_Multi_Asset.py — Comparazione Multi-Asset | Kriterion Quant

Confronto parallelo della resilienza di più asset finanziari:
heatmap TTR Mediano, tabella comparativa risk metrics, KM curves overlay.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_fetcher import fetch_full_history
from src.analytics import (
    calculate_ttr_episodes,
    compute_summary_stats,
    compute_conditional_analysis,
    fit_kaplan_meier,
    DEPTH_LABELS,
)
from src.charts import (
    build_multi_asset_heatmap,
    build_comparative_bar,
    build_kaplan_meier_chart,
)

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================

st.set_page_config(
    page_title="Multi-Asset Comparison | Kriterion Quant",
    page_icon="🌐",
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

DEFAULT_TICKERS = [
    "SPY.US",
    "QQQ.US",
    "GLD.US",
    "BTC-USD.CC",
    "ETH-USD.CC",
    "IWM.US",
]

with st.sidebar:
    st.title("🌐 Multi-Asset")
    st.caption("Kriterion Quant — Comparazione Resilienza")
    st.divider()

    st.subheader("📋 Tickers da Confrontare")
    st.markdown("Un ticker per riga (formato EODHD).")

    tickers_text = st.text_area(
        "Lista Tickers",
        value="\n".join(DEFAULT_TICKERS),
        height=160,
        help="Un ticker per riga. Usa la sintassi EODHD (es. SPY.US, BTC-USD.CC).",
    )
    tickers_list = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]

    st.divider()

    min_depth_slider = st.slider(
        "Profondità minima (%)",
        min_value=-50, max_value=0, value=-5, step=1,
    )
    min_depth_pct = min_depth_slider / 100

    st.divider()

    with st.expander("📖 Sintassi Ticker EODHD"):
        st.markdown("""
**ETF USA:** `SPY.US` · `QQQ.US` · `GLD.US`
**Crypto:** `BTC-USD.CC` · `ETH-USD.CC`
**Indici:** `GSPC.INDX` · `DAX.INDX`
**Forex:** `EURUSD.FOREX`
**Futures:** `GC.COMM` · `CL.COMM`
        """)

    st.divider()
    run_btn = st.button("▶️ Avvia Analisi Multi-Asset", type="primary", use_container_width=True)
    st.caption("⚠️ L'analisi scarica dati per ogni ticker: attendi il completamento.")

# ============================================================
# HEADER
# ============================================================

st.title("🌐 Comparazione Multi-Asset — Resilienza e TTR")
st.markdown("""
Questa sezione confronta la **resilienza storica** di più asset finanziari
in parallelo: quanta profondità di drawdown hanno storicamente subìto
e quanto tempo hanno impiegato per recuperare.

La **heatmap TTR** è uno strumento sinottico immediato: le celle verdi indicano
recovery veloci, le rosse recovery lunghi. Permette di quantificare
i benefici della diversificazione in termini di resilienza al rischio.

> **Come si usa:** inserisci i ticker nella sidebar (uno per riga) e premi **Avvia Analisi**.
""")
st.divider()

# ============================================================
# ANALISI (solo al click del bottone)
# ============================================================

if not run_btn:
    st.info("👈 Configura la lista ticker nella sidebar e premi **Avvia Analisi**.")
    st.stop()

if not tickers_list:
    st.warning("⚠️ Inserisci almeno un ticker nella sidebar.")
    st.stop()

if len(tickers_list) > 12:
    st.warning("⚠️ Massimo 12 ticker per analisi. Usa solo i primi 12.")
    tickers_list = tickers_list[:12]

# ============================================================
# FETCH + CALCOLO PER OGNI ASSET
# ============================================================

progress_bar = st.progress(0, text="Inizializzazione...")
results      = {}   # ticker → dict con tutti i dati

for i, tk in enumerate(tickers_list):
    progress_bar.progress(
        (i) / len(tickers_list),
        text=f"📡 Scaricando {tk} ({i+1}/{len(tickers_list)})...",
    )
    try:
        df = fetch_full_history(tk, EODHD_API_KEY)
        if df.empty:
            st.warning(f"⚠️ Nessun dato per `{tk}`. Salto.")
            continue

        price = df["adjusted_close"].dropna()
        if len(price) < 50:
            st.warning(f"⚠️ Dati insufficienti per `{tk}` ({len(price)} barre). Salto.")
            continue

        episodes  = calculate_ttr_episodes(price, min_depth_pct)
        if episodes.empty:
            st.warning(f"⚠️ Nessun episodio trovato per `{tk}` (soglia: {min_depth_slider}%). Salto.")
            continue

        stats     = compute_summary_stats(episodes, price)
        cond      = compute_conditional_analysis(episodes)
        kmf       = fit_kaplan_meier(episodes, label=tk)

        results[tk] = {
            "price":     price,
            "episodes":  episodes,
            "stats":     {**stats, "ticker": tk},
            "cond":      cond,
            "kmf":       kmf,
        }

    except Exception as e:
        st.warning(f"⚠️ Errore per `{tk}`: {e}. Salto.")
        continue

progress_bar.progress(1.0, text="✅ Analisi completata!")

if not results:
    st.error("❌ Nessun asset analizzato con successo. Verifica i ticker e la connessione.")
    st.stop()

valid_tickers = list(results.keys())
st.success(f"✅ Analizzati **{len(valid_tickers)}** asset: {', '.join(f'`{t}`' for t in valid_tickers)}")
st.divider()

# ============================================================
# SEZIONE 1: HEATMAP TTR MEDIANO
# ============================================================

st.subheader("🌡️ Heatmap Comparativa — TTR Mediano per Asset e Classe")
st.markdown("""
Ogni cella mostra il TTR mediano (in giorni di borsa) per quell'asset in quella
classe di profondità. **Verde** = recovery storicamente rapido, **Rosso** = recovery lento.
Le celle vuote indicano che quell'asset non ha mai subìto drawdown in quella classe.

Questa heatmap permette di rispondere visivamente a domande come:
"Quale asset si riprende più velocemente da un drawdown del -20%?"
""")

# Costruisci pivot table
heatmap_rows = []
for tk, data in results.items():
    cond = data["cond"]
    if cond.empty:
        continue
    for _, row in cond.iterrows():
        heatmap_rows.append({
            "Asset":   tk,
            "Classe":  str(row.get("Classe", "")),
            "TTR_Med": row.get("TTR_Mediano", np.nan),
        })

if heatmap_rows:
    hm_df   = pd.DataFrame(heatmap_rows)
    # Ordine classi per gravità crescente
    cat_order = [c for c in DEPTH_LABELS if c in hm_df["Classe"].values]
    hm_df["Classe"] = pd.Categorical(hm_df["Classe"], categories=cat_order, ordered=True)
    pivot   = hm_df.pivot_table(index="Asset", columns="Classe", values="TTR_Med", aggfunc="mean")
    pivot   = pivot.reindex(columns=[c for c in cat_order if c in pivot.columns])

    fig_hm = build_multi_asset_heatmap(pivot)
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("ℹ️ Dati insufficienti per la heatmap.")

st.divider()

# ============================================================
# SEZIONE 2: TABELLA COMPARATIVA RISK METRICS
# ============================================================

st.subheader("📊 Tabella Comparativa Risk Metrics")
st.markdown("""
Confronto diretto delle principali metriche di resilienza e rischio.
Ordina la tabella per qualsiasi colonna per identificare immediatamente
l'asset con la migliore o peggiore performance su quella dimensione.
""")

stats_list = [data["stats"] for data in results.values()]
metrics_rows = []
for s in stats_list:
    metrics_rows.append({
        "Asset":          s.get("ticker", "?"),
        "Storia (anni)":  s.get("anni_storia", 0),
        "N° Episodi":     s.get("n_episodi", 0),
        "TTR Mediano (gg)": s.get("ttr_mediano_gg", 0),
        "TTR P90 (gg)":   s.get("ttr_p90_gg", 0),
        "Max DD (%)":     s.get("profondita_max_pct", 0),
        "CAGR (%)":       s.get("cagr_pct", 0),
        "Calmar Ratio":   s.get("calmar_ratio", None),
        "Ulcer Index":    s.get("ulcer_index", 0),
        "Pain Index":     s.get("pain_index", 0),
        "Recovery Factor": s.get("recovery_factor", None),
    })

metrics_df = pd.DataFrame(metrics_rows)

styled_metrics = (
    metrics_df.style
    .background_gradient(cmap="RdYlGn_r", subset=["TTR Mediano (gg)", "TTR P90 (gg)",
                                                    "Ulcer Index", "Pain Index"])
    .background_gradient(cmap="RdYlGn", subset=["CAGR (%)", "Calmar Ratio",
                                                  "Recovery Factor"])
    .format({
        "Storia (anni)":    "{:.1f}",
        "TTR Mediano (gg)": "{:.0f}",
        "TTR P90 (gg)":     "{:.0f}",
        "Max DD (%)":       "{:.1f}%",
        "CAGR (%)":         "{:.2f}%",
        "Calmar Ratio":     lambda x: f"{x:.2f}" if x is not None else "N/A",
        "Ulcer Index":      "{:.2f}%",
        "Pain Index":       "{:.2f}%",
        "Recovery Factor":  lambda x: f"{x:.2f}" if x is not None else "N/A",
    })
    .set_properties(**{"text-align": "center"})
)
st.dataframe(styled_metrics, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# SEZIONE 3: BAR CHART COMPARATIVO
# ============================================================

st.subheader("📊 Confronto Visivo Metriche Chiave")

col_m1, col_m2 = st.columns(2)

with col_m1:
    fig_bar_ttr = build_comparative_bar(
        stats_list,
        metric="ttr_mediano_gg",
        y_label="TTR Mediano (giorni)",
    )
    st.plotly_chart(fig_bar_ttr, use_container_width=True)

with col_m2:
    fig_bar_ulcer = build_comparative_bar(
        stats_list,
        metric="ulcer_index",
        y_label="Ulcer Index (%)",
    )
    st.plotly_chart(fig_bar_ulcer, use_container_width=True)

col_m3, col_m4 = st.columns(2)

with col_m3:
    fig_bar_calmar = build_comparative_bar(
        stats_list,
        metric="calmar_ratio",
        y_label="Calmar Ratio",
    )
    st.plotly_chart(fig_bar_calmar, use_container_width=True)

with col_m4:
    fig_bar_dd = build_comparative_bar(
        stats_list,
        metric="profondita_max_pct",
        y_label="Max Drawdown (%)",
    )
    st.plotly_chart(fig_bar_dd, use_container_width=True)

st.divider()

# ============================================================
# SEZIONE 4: KAPLAN-MEIER OVERLAY
# ============================================================

st.subheader("📐 Funzioni di Sopravvivenza — Overlay Multi-Asset")
st.markdown("""
Sovrappone le curve di Kaplan-Meier di tutti gli asset analizzati.
Un asset la cui curva rimane più alta per più tempo ha storicamente
drawdown più *persistenti*: anche se non necessariamente più profondi,
il mercato impiega più tempo a restituire i livelli precedenti.

Questo grafico è particolarmente utile per confrontare asset dello stesso
universo (es. ETF settoriali, crypto vs indici) in termini di velocità
di recupero strutturale.
""")

kmf_list_all = [data["kmf"] for data in results.values() if data["kmf"] is not None]
if kmf_list_all:
    fig_km_all = build_kaplan_meier_chart(kmf_list_all, show_ci=False)
    st.plotly_chart(fig_km_all, use_container_width=True)
else:
    st.info("ℹ️ Dati insufficienti per le curve Kaplan-Meier.")

st.divider()
st.caption(
    f"Kriterion Quant · Multi-Asset TTR · "
    f"Asset: {', '.join(valid_tickers)} · "
    f"Fonte: EODHD"
)
