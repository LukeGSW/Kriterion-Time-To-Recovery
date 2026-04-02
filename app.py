"""
app.py — TTR Drawdown Dashboard | Kriterion Quant
Pagina principale: analisi singolo asset (tutta la storia disponibile).

Struttura:
    Sidebar → parametri analisi e legenda ticker EODHD
    Main    → KPI metrics | Equity + DD | Kaplan-Meier | Analisi condizionale
              | Regime | Tabella episodi | Export JSON
"""

import json
import streamlit as st
import pandas as pd
import numpy as np

from src.data_fetcher  import fetch_full_history
from src.analytics     import (
    calculate_ttr_episodes,
    compute_summary_stats,
    compute_conditional_analysis,
    fit_kaplan_meier,
    fit_regime_km,
    build_export_json,
)
from src.charts import (
    build_equity_drawdown_chart,
    build_drawdown_series_chart,
    build_kaplan_meier_chart,
    build_ttr_boxplot,
)

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================

st.set_page_config(
    page_title="TTR Drawdown Dashboard | Kriterion Quant",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# API KEY
# ============================================================

try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    st.error(
        "❌ **EODHD_API_KEY non trovata.** "
        "Aggiungila in `.streamlit/secrets.toml` (locale) "
        "o in Settings → Secrets (Streamlit Cloud)."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📉 TTR Dashboard")
    st.caption("Kriterion Quant — Drawdown & Time to Recovery")
    st.divider()

    # --- Ticker principale ---
    st.subheader("⚙️ Parametri Analisi")
    ticker = st.text_input(
        "Ticker EODHD",
        value="SPY.US",
        help="Inserisci il simbolo nel formato EODHD (vedi legenda sotto).",
    ).strip().upper()

    # --- Filtro profondità minima ---
    min_depth_slider = st.slider(
        "Profondità minima drawdown (%)",
        min_value=-50, max_value=0, value=-5, step=1,
        help="Ignora episodi con profondità superiore (meno profondi) alla soglia.",
    )
    min_depth_pct = min_depth_slider / 100  # conversione in decimale (es. -0.05)

    st.divider()

    # --- Parametri regime ---
    st.subheader("🔁 Analisi di Regime")
    regime_type = st.radio(
        "Tipo di regime",
        options=["VIX (Volatilità)", "SMA200 (Trend)"],
        index=0,
        help="Scegli come dividere gli episodi in regimi di mercato.",
    )

    use_sma200 = regime_type == "SMA200 (Trend)"

    vix_ticker    = "VIX.INDX"
    vix_threshold = 20.0
    if not use_sma200:
        vix_ticker = st.text_input(
            "Ticker VIX (EODHD)",
            value="VIX.INDX",
            help="Ticker EODHD per il VIX. Default: VIX.INDX",
        ).strip().upper()
        vix_threshold = st.slider(
            "Soglia VIX",
            min_value=10.0, max_value=40.0, value=20.0, step=0.5,
            help="Episodi con VIX ≥ soglia → regime 'Alta Volatilità'.",
        )

    st.divider()

    # --- Legenda sintassi ticker EODHD ---
    with st.expander("📖 Sintassi Ticker EODHD"):
        st.markdown("""
**Azioni & ETF USA** → suffisso `.US`
`SPY.US` · `QQQ.US` · `AAPL.US` · `TSLA.US`

**Azioni Europa** → codice exchange
`ENI.MI` · `BMW.XETRA` · `BP.LSE` · `AIR.PA`

**Crypto** → suffisso `.CC`
`BTC-USD.CC` · `ETH-USD.CC` · `SOL-USD.CC`

**Forex** → suffisso `.FOREX`
`EURUSD.FOREX` · `GBPUSD.FOREX` · `USDJPY.FOREX`

**Indici** → suffisso `.INDX`
`GSPC.INDX` (S&P 500) · `DJI.INDX` (Dow)
`NDX.INDX` (Nasdaq 100) · `VIX.INDX` · `DAX.INDX`

**Futures/Commodities** → suffisso `.COMM`
`GC.COMM` (Oro) · `CL.COMM` (WTI Oil)
`ES.COMM` (E-mini S&P) · `ZN.COMM` (T-Note 10Y)

**Obbligazioni (ETF)** → come azioni USA
`TLT.US` · `HYG.US` · `LQD.US`
        """)

    st.divider()
    st.caption("📡 Fonte dati: EODHD Historical Data")
    st.caption("🕐 Cache dati: 60 minuti")

# ============================================================
# HEADER PAGINA
# ============================================================

st.title("📉 Drawdown & Time to Recovery — Analisi Singolo Asset")
st.markdown("""
Questo studio quantifica la **resilienza storica** di un asset finanziario:
quante volte ha subìto drawdown significativi, quanto profondi, e quanto tempo
ha impiegato per recuperare il massimo precedente (*Time to Recovery, TTR*).

L'analisi usa tutta la storia disponibile per il ticker selezionato,
senza filtri temporali, per massimizzare la significatività statistica.

> **Come si usa:** imposta il ticker nella sidebar, regola la soglia di profondità
> minima per filtrare i rumori, poi scorri le sezioni dell'analisi.
""")
st.divider()

# ============================================================
# FETCH DATI ASSET PRINCIPALE
# ============================================================

if not ticker:
    st.warning("⚠️ Inserisci un ticker nella sidebar per avviare l'analisi.")
    st.stop()

with st.spinner(f"⏳ Scaricando tutta la storia disponibile per **{ticker}**..."):
    try:
        df = fetch_full_history(ticker, EODHD_API_KEY)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dati per `{ticker}`: {e}")
        st.stop()

if df.empty:
    st.error(
        f"❌ Nessun dato trovato per `{ticker}`. "
        "Verifica la sintassi del ticker (vedi legenda nella sidebar)."
    )
    st.stop()

price_series = df["adjusted_close"].dropna()

# Controllo minimo dati
if len(price_series) < 50:
    st.warning(
        f"⚠️ Solo {len(price_series)} osservazioni disponibili per `{ticker}`. "
        "L'analisi potrebbe non essere statisticamente significativa."
    )

# ============================================================
# CALCOLO EPISODI E STATISTICHE
# ============================================================

with st.spinner("🔢 Calcolando episodi di drawdown e statistiche..."):
    episodes_df    = calculate_ttr_episodes(price_series, min_depth_pct)
    summary_stats  = compute_summary_stats(episodes_df, price_series) if not episodes_df.empty else {}
    conditional_df = compute_conditional_analysis(episodes_df)
    kmf_overall    = fit_kaplan_meier(episodes_df, label=ticker) if not episodes_df.empty else None

# Salva in session_state per le pagine Monte Carlo
st.session_state["ticker"]       = ticker
st.session_state["episodes_df"]  = episodes_df
st.session_state["price_series"] = price_series
st.session_state["summary_stats"] = summary_stats

if episodes_df.empty:
    st.warning(
        f"⚠️ Nessun episodio di drawdown trovato per `{ticker}` "
        f"con la soglia minima impostata ({min_depth_slider}%). "
        "Prova ad abbassare la soglia."
    )
    st.stop()

n_eps = len(episodes_df)
data_start = str(price_series.index[0].date())
data_end   = str(price_series.index[-1].date())

# ============================================================
# SEZIONE 1: KPI METRICS
# ============================================================

st.subheader(f"📊 KPI di Resilienza — {ticker}")
st.markdown(f"""
Riepilogo quantitativo su **{summary_stats.get('anni_storia', 0):.1f} anni di storia**
({data_start} → {data_end}).
Le metriche avanzate completano il quadro: l'**Ulcer Index** penalizza sia la profondità
che la *durata* dei drawdown; il **Calmar Ratio** confronta il rendimento annualizzato
con il rischio di perdita massima; il **Recovery Factor** indica quante volte il profitto
netto compensa il drawdown massimo storico.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("N° Episodi", f"{summary_stats.get('n_episodi', 0)}")
c2.metric("TTR Mediano", f"{summary_stats.get('ttr_mediano_gg', 0):.0f} gg")
c3.metric("Max Drawdown", f"{summary_stats.get('profondita_max_pct', 0):.1f}%")
c4.metric("TTR Max Storico", f"{summary_stats.get('ttr_max_gg', 0)} gg")

c5, c6, c7, c8 = st.columns(4)
calmar_val = summary_stats.get("calmar_ratio")
ulcer_val  = summary_stats.get("ulcer_index", 0)
pain_val   = summary_stats.get("pain_index", 0)
recfac_val = summary_stats.get("recovery_factor")

c5.metric("Calmar Ratio",
          f"{calmar_val:.2f}" if calmar_val is not None else "N/A",
          help="CAGR / |Max Drawdown|. Più alto = migliore risk/return.")
c6.metric("Ulcer Index",
          f"{ulcer_val:.2f}%",
          help="Misura del 'dolore' da drawdown (profondità × durata). Più basso = meglio.")
c7.metric("Pain Index",
          f"{pain_val:.2f}%",
          help="Media del drawdown in % nel tempo. Più basso = meno stress cronico.")
c8.metric("Recovery Factor",
          f"{recfac_val:.2f}" if recfac_val is not None else "N/A",
          help="Profitto netto / Max Drawdown assoluto. > 1 = il guadagno supera la perdita massima.")

st.divider()

# ============================================================
# SEZIONE 2: EQUITY LINE + DRAWDOWN
# ============================================================

st.subheader("📈 Equity Line con Periodi di Drawdown")
st.markdown("""
L'equity line adjusted mostra il prezzo rettificato per dividendi e split.
Le **aree rosse** evidenziano ogni episodio di drawdown rilevato:
l'intensità del colore è proporzionale alla profondità.
Ogni area è annotata con la profondità percentuale del trough.
""")

fig_eq = build_equity_drawdown_chart(price_series, episodes_df, ticker)
st.plotly_chart(fig_eq, use_container_width=True)

# Drawdown % continuo
st.markdown("""
Il grafico sottostante mostra il **drawdown percentuale continuo** rispetto all'High-Water Mark (HWM):
in ogni punto, la distanza dal massimo storico precedente. Utile per visualizzare la durata
complessiva dei periodi "sott'acqua", anche quando i singoli episodi si susseguono.
""")
fig_dd = build_drawdown_series_chart(price_series, ticker)
st.plotly_chart(fig_dd, use_container_width=True)

with st.expander("ℹ️ Metodologia"):
    st.markdown("""
    - **High-Water Mark (HWM):** massimo cumulato del prezzo adjusted in ogni istante.
    - **Drawdown:** `(prezzo / HWM) - 1` — sempre ≤ 0.
    - **Inizio episodio:** primo giorno in cui il prezzo scende sotto l'HWM precedente.
    - **Fine episodio (recovery):** primo giorno in cui il prezzo torna ≥ al picco iniziale.
    - **Censurato:** episodio ancora aperto alla fine della serie storica disponibile.
    """)

st.divider()

# ============================================================
# SEZIONE 3: KAPLAN-MEIER
# ============================================================

st.subheader("📐 Funzione di Sopravvivenza dei Drawdown (Kaplan-Meier)")
st.markdown("""
La curva di Kaplan-Meier stima la probabilità che un drawdown duri **più di t giorni**
di borsa. È l'equivalente della funzione di sopravvivenza in analisi medica,
adattata alla resilienza finanziaria.

**Come si legge:** se la curva è al livello 0.20 a quota t=252,
significa che storicamente il **20%** dei drawdown di `{ticker}` ha impiegato
più di un anno lavorativo per risolversi.

L'area semitrasparente rappresenta l'intervallo di confidenza al 95% della stima,
che si allarga progressivamente per TTR lunghi (dove i dati sono più rari).
""".format(ticker=ticker))

if kmf_overall is not None:
    fig_km = build_kaplan_meier_chart([kmf_overall], ticker)
    st.plotly_chart(fig_km, use_container_width=True)
else:
    st.info("ℹ️ Dati insufficienti per il calcolo Kaplan-Meier (minimo 2 episodi).")

st.divider()

# ============================================================
# SEZIONE 4: ANALISI CONDIZIONALE
# ============================================================

st.subheader("📋 TTR Mediano per Classe di Profondità")
st.markdown("""
Questa tabella quantifica la relazione **non lineare** tra profondità e durata
del drawdown. Un calo del -30% non richiede il triplo del tempo di un -10%:
in molti asset la relazione è super-lineare, soprattutto per drawdown
legati a shock sistemici (crisi 2008, COVID, crypto bear markets).

L'IQR (distanza interquartilica) misura la dispersione del TTR all'interno
di ogni classe: un IQR ampio indica alta imprevedibilità nei tempi di recovery.
""")

if not conditional_df.empty:
    # Formattazione tabella
    display_cond = conditional_df.copy()
    display_cond["TTR_Mediano"] = display_cond["TTR_Mediano"].round(0).astype(int)
    display_cond["TTR_Media"]   = display_cond["TTR_Media"].round(0).astype(int)
    display_cond["IQR"]         = display_cond["IQR"].round(0).astype(int)
    display_cond["TTR_Max"]     = display_cond["TTR_Max"].astype(int)

    styled = (
        display_cond
        .style
        .background_gradient(cmap="RdYlGn_r", subset=["TTR_Mediano"])
        .format({
            "TTR_Mediano": "{} gg",
            "TTR_Media":   "{} gg",
            "IQR":         "{} gg",
            "TTR_Max":     "{} gg",
            "% Completati": "{:.1f}%",
        })
        .set_properties(**{"text-align": "center"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ Dati insufficienti per l'analisi condizionale.")

# Boxplot
st.markdown("**Distribuzione del TTR per classe** (scala logaritmica):")
fig_box = build_ttr_boxplot(episodes_df, ticker)
st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ============================================================
# SEZIONE 5: ANALISI DI REGIME
# ============================================================

st.subheader("🔁 Analisi di Regime")

if use_sma200:
    st.markdown(f"""
    Confronto delle curve di sopravvivenza nei due regimi di trend:
    **Sopra SMA200** (mercato in trend rialzista al momento del picco)
    vs **Sotto SMA200** (mercato in fase correttiva o bear).

    Se la curva "Sotto SMA200" scende molto più lentamente, `{ticker}` impiega
    significativamente più tempo a recuperare quando il drawdown avviene in un
    contesto di trend ribassista — dato operativamente rilevante per la gestione
    del rischio in portafoglio.
    """)
    vix_series_for_regime = None
else:
    st.markdown(f"""
    Confronto delle curve di sopravvivenza nei due regimi di volatilità implicita:
    **VIX < {vix_threshold:.0f}** (mercato calmo) vs **VIX ≥ {vix_threshold:.0f}** (mercato stressato).

    Se la curva "Alta Volatilità" scende più lentamente, `{ticker}` soffre di
    recovery più lenti durante i periodi di paura di mercato — un'indicazione che
    i drawdown in contesti di stress sono strutturalmente diversi da quelli ordinari.
    """)
    # Fetch VIX per analisi regime
    with st.spinner(f"⏳ Scaricando dati VIX ({vix_ticker})..."):
        try:
            vix_df      = fetch_full_history(vix_ticker, EODHD_API_KEY)
            vix_series_for_regime = vix_df["adjusted_close"].dropna() \
                                    if not vix_df.empty else None
            if vix_series_for_regime is None:
                # Fallback: prova con 'close'
                vix_series_for_regime = vix_df["close"].dropna() \
                                        if not vix_df.empty and "close" in vix_df.columns \
                                        else None
        except Exception:
            vix_series_for_regime = None

    if vix_series_for_regime is None:
        st.warning(
            f"⚠️ Impossibile scaricare i dati VIX da `{vix_ticker}`. "
            "Verifica il ticker oppure passa al regime SMA200."
        )

kmf_regimes = fit_regime_km(
    episodes_df,
    vix_series=vix_series_for_regime if not use_sma200 else None,
    vix_threshold=vix_threshold,
    use_sma200=use_sma200,
    price_series=price_series,
)

if kmf_regimes:
    # Aggiungi la curva Overall per confronto
    all_kmf = [kmf_overall] + kmf_regimes if kmf_overall else kmf_regimes
    fig_regime = build_kaplan_meier_chart(all_kmf, ticker)
    st.plotly_chart(fig_regime, use_container_width=True)
else:
    st.info(
        "ℹ️ Dati insufficienti per l'analisi di regime. "
        "Servono almeno 3 episodi per ciascun regime."
    )

st.divider()

# ============================================================
# SEZIONE 6: LOG EPISODI (TABELLA FILTRATA)
# ============================================================

st.subheader("📋 Log Completo degli Episodi di Drawdown")
st.markdown("""
Tabella completa di tutti gli episodi rilevati con i parametri correnti.
Usa i **filtri** per isolare gli episodi più rilevanti.
Il **diario clinico** del rischio storico dell'asset.
""")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    show_censored = st.checkbox("Mostra episodi censurati (in corso)", value=True)
with col_f2:
    depth_classes = list(episodes_df["depth_bin"].cat.categories)
    selected_classes = st.multiselect(
        "Filtra per classe di profondità",
        options=depth_classes,
        default=depth_classes,
    )
with col_f3:
    min_ttr_filter = st.number_input("TTR minimo (giorni)", min_value=0, value=0, step=1)

# Applica filtri
filtered_eps = episodes_df.copy()
if not show_censored:
    filtered_eps = filtered_eps[~filtered_eps["is_censored"]]
if selected_classes:
    filtered_eps = filtered_eps[filtered_eps["depth_bin"].isin(selected_classes)]
filtered_eps = filtered_eps[filtered_eps["ttr_days"] >= min_ttr_filter]

# Formattazione display
display_eps = filtered_eps.copy()
display_eps["peak_date"]     = display_eps["peak_date"].dt.strftime("%Y-%m-%d")
display_eps["trough_date"]   = display_eps["trough_date"].dt.strftime("%Y-%m-%d")
display_eps["recovery_date"] = display_eps["recovery_date"].apply(
    lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "⏳ In corso"
)
display_eps["depth_pct"]  = (display_eps["depth_pct"] * 100).round(2).astype(str) + "%"
display_eps["is_censored"] = display_eps["is_censored"].map({True: "⏳ Sì", False: "✅ No"})
display_eps = display_eps.rename(columns={
    "peak_date":     "Data Picco",
    "trough_date":   "Data Minimo",
    "recovery_date": "Data Recovery",
    "depth_pct":     "Profondità (%)",
    "ttr_days":      "TTR (giorni)",
    "is_censored":   "Censurato",
    "depth_bin":     "Classe",
})

st.dataframe(
    display_eps[["Data Picco", "Data Minimo", "Data Recovery",
                 "Profondità (%)", "TTR (giorni)", "Censurato", "Classe"]],
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Episodi visualizzati: **{len(filtered_eps)}** su {n_eps} totali.")

st.divider()

# ============================================================
# SEZIONE 7: EXPORT JSON
# ============================================================

st.subheader("⬇️ Export Studio — JSON per Analisi LLM")
st.markdown("""
Scarica l'intero studio in formato JSON strutturato, ottimizzato per
essere analizzato da un modello linguistico (ChatGPT, Claude, Gemini, ecc.).

Il file include: statistiche aggregate, log degli episodi, funzione di sopravvivenza
Kaplan-Meier e analisi condizionale per classe. Non include i raw OHLCV
per mantenere il file compatto e focalizzato sulle metriche di rischio.
""")

if kmf_overall is not None:
    json_str = build_export_json(
        ticker=ticker,
        price_series=price_series,
        episodes_df=episodes_df,
        summary_stats=summary_stats,
        conditional_df=conditional_df,
        kmf=kmf_overall,
    )

    st.download_button(
        label=f"📥 Scarica {ticker}_TTR_Study.json",
        data=json_str,
        file_name=f"{ticker}_TTR_Study.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("👁️ Anteprima JSON (primi 2000 caratteri)"):
        st.code(json_str[:2000] + "\n...", language="json")
else:
    st.info("ℹ️ Export non disponibile: dati insufficienti per l'analisi completa.")

st.divider()
st.caption(
    "Kriterion Quant · TTR Drawdown Dashboard · "
    f"Dati: EODHD · Ticker: {ticker} · "
    f"Storia: {data_start} → {data_end} · "
    f"{summary_stats.get('n_osservazioni', 0):,} osservazioni"
)
