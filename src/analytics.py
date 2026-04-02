"""
analytics.py — Calcoli quantitativi core: TTR, Kaplan-Meier, risk metrics, Monte Carlo.

Tutte le funzioni sono pure (input/output espliciti, nessuno stato globale)
e indipendenti da Streamlit, per facilitare test e riuso.
"""

import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


# ============================================================
# CLASSIFICAZIONE PROFONDITÀ DRAWDOWN
# ============================================================

DEPTH_BINS   = [-np.inf, -0.50, -0.30, -0.20, -0.10, -0.05, 0.0]
DEPTH_LABELS = ["< -50%", "[-50%,-30%)", "[-30%,-20%)", "[-20%,-10%)", "[-10%,-5%)", "[-5%,0%)"]


# ============================================================
# RILEVAMENTO EPISODI DI DRAWDOWN E TTR
# ============================================================

def calculate_ttr_episodes(
    price_series: pd.Series,
    min_depth_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Identifica tutti gli episodi di drawdown e calcola il Time to Recovery (TTR).

    Algoritmo High-Water Mark (HWM):
    - Un episodio inizia quando il prezzo scende sotto il precedente HWM.
    - L'episodio termina (recovered) quando il prezzo torna >= al picco iniziale.
    - Se l'episodio non si è chiuso entro l'ultimo dato, è marcato 'censurato'.

    Il trattamento dei dati censurati con Kaplan-Meier garantisce stime non
    distorte (nessun survivorship bias): anche i drawdown non risolti contribuiscono
    all'analisi.

    Args:
        price_series:   Serie temporale dei prezzi adjusted_close (DatetimeIndex)
        min_depth_pct:  Soglia di filtro: scarta episodi meno profondi di questo
                        valore (es. -0.05 per ignorare drawdown < 5%).
                        Deve essere <= 0. Default 0.0 (nessun filtro).

    Returns:
        DataFrame con colonne:
          peak_date, trough_date, recovery_date (NaT se censurato),
          depth_pct (valore negativo), ttr_days, is_censored, depth_bin
    """
    if price_series.empty or len(price_series) < 2:
        return pd.DataFrame()

    price_series = price_series.dropna().sort_index()
    hwm      = price_series.cummax()
    drawdown = (price_series / hwm) - 1  # sempre <= 0

    episodes   = []
    in_drawdown = False
    episode     = {}

    for current_date, dd_val in drawdown.items():

        if not in_drawdown and dd_val < 0:
            # === INIZIO EPISODIO ===
            # Trova l'HWM precedente più recente (ultimo punto con dd == 0)
            prev_idx = drawdown.index[drawdown.index < current_date]
            if prev_idx.empty:
                continue
            prev_dd = drawdown.loc[prev_idx]
            prev_zero = prev_dd[prev_dd == 0]
            if prev_zero.empty:
                continue

            peak_date = prev_zero.index.max()
            in_drawdown = True
            episode = {
                "peak_date":   peak_date,
                "peak_value":  price_series.loc[peak_date],
                "trough_date": current_date,
                "depth_pct":   dd_val,
            }

        elif in_drawdown and "peak_value" in episode:
            # === DURANTE L'EPISODIO ===
            # Aggiorna il minimo se troviamo un punto più basso
            if dd_val < episode["depth_pct"]:
                episode["depth_pct"]   = dd_val
                episode["trough_date"] = current_date

            # Verifica recovery completo (prezzo >= picco iniziale)
            if price_series[current_date] >= episode["peak_value"]:
                in_drawdown = False
                episode["recovery_date"] = current_date
                # TTR = numero di barre da picco a recovery (escluso il picco)
                segment = price_series.loc[episode["peak_date"]:current_date]
                episode["ttr_days"]    = len(segment) - 1
                episode["is_censored"] = False
                episodes.append(episode)
                episode = {}

    # === EPISODIO CENSURATO (ancora aperto a fine serie) ===
    if in_drawdown and "peak_value" in episode:
        segment = price_series.loc[episode["peak_date"]:]
        episode.update({
            "recovery_date": pd.NaT,
            "ttr_days":      len(segment) - 1,
            "is_censored":   True,
        })
        episodes.append(episode)

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes)[[
        "peak_date", "trough_date", "recovery_date",
        "depth_pct", "ttr_days", "is_censored",
    ]]

    # Classificazione in fasce di profondità
    df["depth_bin"] = pd.cut(
        df["depth_pct"],
        bins=DEPTH_BINS,
        labels=DEPTH_LABELS,
        right=False,
    )

    # Filtro profondità minima (min_depth_pct <= 0)
    if min_depth_pct < 0:
        df = df[df["depth_pct"] <= min_depth_pct]

    df = df.reset_index(drop=True)
    df["ttr_days"] = df["ttr_days"].astype(int)

    return df


# ============================================================
# KAPLAN-MEIER
# ============================================================

def fit_kaplan_meier(episodes_df: pd.DataFrame, label: str = "Overall") -> KaplanMeierFitter:
    """
    Stima la funzione di sopravvivenza dei drawdown con il metodo di Kaplan-Meier.

    Il 'tempo' è il TTR in giorni di borsa. L''evento' è il recupero completo.
    I drawdown censurati (ancora aperti) entrano nell'analisi senza bias:
    contribuiscono all'informazione finché sono osservati, poi vengono rimossi.

    Interpretazione: S(t) = probabilità che un drawdown duri PIÙ di t giorni.
    Se S(252) = 0.15, c'è il 15% di probabilità che un drawdown non si sia
    risolto dopo un anno di borsa.

    Args:
        episodes_df: Output di calculate_ttr_episodes()
        label:       Etichetta curva (per la legenda del grafico)

    Returns:
        KaplanMeierFitter già fittato
    """
    if episodes_df.empty or len(episodes_df) < 2:
        return None

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=episodes_df["ttr_days"],
        event_observed=~episodes_df["is_censored"],
        label=label,
    )
    return kmf


def fit_regime_km(
    episodes_df: pd.DataFrame,
    vix_series: pd.Series = None,
    vix_threshold: float = 20.0,
    use_sma200: bool = False,
    price_series: pd.Series = None,
) -> list:
    """
    Fitta curve Kaplan-Meier separate per regimi di mercato.

    Regime VIX: split su soglia VIX configurabile (alta vs bassa volatilità).
    Regime SMA200: split su prezzo > SMA200 (trend bull) vs < SMA200 (trend bear).

    Per il regime VIX, il valore del VIX viene campionato alla data di picco
    di ogni episodio (using pd.merge_asof per gestire date di borsa diverse).

    Args:
        episodes_df:   Episodi drawdown con colonna 'peak_date'
        vix_series:    Serie storica VIX (index DatetimeIndex, valori numerici)
        vix_threshold: Soglia VIX per lo split (default 20)
        use_sma200:    Se True, usa SMA200 invece del VIX
        price_series:  Serie prezzi originale (richiesta se use_sma200=True)

    Returns:
        Lista di KaplanMeierFitter (uno per regime), vuota se dati insufficienti
    """
    if episodes_df.empty or len(episodes_df) < 4:
        return []

    eps = episodes_df.copy().sort_values("peak_date")

    if use_sma200 and price_series is not None:
        # === REGIME SMA200 ===
        sma200 = price_series.rolling(200, min_periods=100).mean()

        def _get_regime_sma(peak_date):
            try:
                p = price_series.asof(peak_date)
                s = sma200.asof(peak_date)
                if pd.isna(p) or pd.isna(s):
                    return None
                return "Sopra SMA200 (Bull)" if p >= s else "Sotto SMA200 (Bear)"
            except Exception:
                return None

        eps["regime"] = eps["peak_date"].apply(_get_regime_sma)

    elif vix_series is not None and not vix_series.empty:
        # === REGIME VIX ===
        vix_df = vix_series.to_frame(name="vix_close").dropna()
        vix_df.index = pd.to_datetime(vix_df.index)

        merged = pd.merge_asof(
            eps,
            vix_df,
            left_on="peak_date",
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("5D"),
        )

        def _get_regime_vix(row):
            v = row.get("vix_close", np.nan)
            if pd.isna(v):
                return None
            return f"VIX ≥ {vix_threshold:.0f} (Alta Vol.)" if v >= vix_threshold \
                   else f"VIX < {vix_threshold:.0f} (Bassa Vol.)"

        merged["regime"] = merged.apply(_get_regime_vix, axis=1)
        eps = merged

    else:
        return []

    # Fitta KM per ogni regime con almeno 3 episodi
    kmf_list = []
    for regime in eps["regime"].dropna().unique():
        subset = eps[eps["regime"] == regime]
        if len(subset) < 3:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset["ttr_days"], ~subset["is_censored"], label=regime)
        kmf_list.append(kmf)

    return kmf_list


# ============================================================
# STATISTICHE RIEPILOGO E RISK METRICS
# ============================================================

def compute_summary_stats(
    episodes_df: pd.DataFrame,
    price_series: pd.Series,
) -> dict:
    """
    Calcola le statistiche di riepilogo TTR e i risk metrics avanzati.

    Risk metrics inclusi:
    - Calmar Ratio: CAGR / |max drawdown| — misura il rendimento per unità
      di drawdown massimo. Valori > 1 indicano buona efficienza risk/return.
    - Ulcer Index: sqrt(media dei drawdown^2) — penalizza sia la profondità
      che la persistenza dei drawdown. Sviluppato da Peter Martin (1987).
    - Pain Index: media semplice dei drawdown (in modulo) sul periodo —
      misura il "costo psicologico" medio dell'investimento.
    - Recovery Factor: profitto netto / max drawdown assoluto — quante volte
      il profitto compensa il rischio storico massimo.

    Args:
        episodes_df:  Output di calculate_ttr_episodes()
        price_series: Serie prezzi adjusted_close originale (non filtrata)

    Returns:
        Dict con tutte le statistiche, serializzabile in JSON
    """
    if episodes_df.empty:
        return {}

    # === TTR STATISTICS ===
    ttr     = episodes_df["ttr_days"]
    depths  = episodes_df["depth_pct"]

    n_years      = max(len(price_series) / 252, 0.001)
    total_return = price_series.iloc[-1] / price_series.iloc[0] - 1
    cagr         = (1 + total_return) ** (1 / n_years) - 1

    # === DRAWDOWN SERIES (continua, per Ulcer e Pain) ===
    hwm       = price_series.cummax()
    dd_series = (price_series / hwm) - 1   # sempre <= 0

    # Ulcer Index: sqrt(media(dd^2)) — più alto = più "dolore" prolungato
    ulcer_index = float(np.sqrt((dd_series ** 2).mean()) * 100)

    # Pain Index: media del drawdown in modulo
    pain_index = float((-dd_series).mean() * 100)

    # Calmar Ratio: CAGR / max drawdown (positivo)
    max_dd_pct = abs(depths.min())
    calmar     = float(cagr / max_dd_pct) if max_dd_pct > 0 else np.nan

    # Recovery Factor: profitto netto / max drawdown assoluto in unità di prezzo
    net_profit  = float(price_series.iloc[-1] - price_series.iloc[0])
    # Max drawdown assoluto: hwm * |depth_pct| al momento del trough
    trough_prices = episodes_df.apply(
        lambda r: price_series.asof(r["trough_date"]) if pd.notna(r["trough_date"]) else np.nan,
        axis=1,
    )
    peak_prices = episodes_df.apply(
        lambda r: price_series.asof(r["peak_date"]) if pd.notna(r["peak_date"]) else np.nan,
        axis=1,
    )
    max_dd_abs   = float((peak_prices - trough_prices).max())
    recovery_fac = float(net_profit / max_dd_abs) if max_dd_abs > 0 else np.nan

    return {
        # Periodo
        "data_inizio":       str(price_series.index[0].date()),
        "data_fine":         str(price_series.index[-1].date()),
        "n_osservazioni":    int(len(price_series)),
        "anni_storia":       round(n_years, 1),
        # TTR metrics
        "n_episodi":         int(len(episodes_df)),
        "n_censurati":       int(episodes_df["is_censored"].sum()),
        "pct_censurati":     round(episodes_df["is_censored"].mean() * 100, 1),
        "ttr_mediano_gg":    round(float(ttr.median()), 1),
        "ttr_media_gg":      round(float(ttr.mean()), 1),
        "ttr_p75_gg":        round(float(ttr.quantile(0.75)), 1),
        "ttr_p90_gg":        round(float(ttr.quantile(0.90)), 1),
        "ttr_max_gg":        int(ttr.max()),
        "profondita_mediana_pct": round(float(depths.median() * 100), 2),
        "profondita_max_pct":     round(float(depths.min() * 100), 2),
        # Performance
        "cagr_pct":          round(cagr * 100, 2),
        "total_return_pct":  round(total_return * 100, 2),
        # Risk metrics avanzati
        "calmar_ratio":      round(calmar, 3) if not np.isnan(calmar) else None,
        "ulcer_index":       round(ulcer_index, 3),
        "pain_index":        round(pain_index, 3),
        "recovery_factor":   round(recovery_fac, 3) if not np.isnan(recovery_fac) else None,
    }


def compute_conditional_analysis(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisi condizionale: TTR mediano per classe di profondità del drawdown.

    Quantifica la relazione non lineare tra magnitudine e durata del drawdown.
    Un drawdown del -20% non richiede necessariamente il doppio del tempo
    di recupero rispetto a uno del -10%: in molti asset la relazione è
    super-lineare (soprattutto oltre certi livelli di stress di mercato).

    Args:
        episodes_df: Output di calculate_ttr_episodes() con colonna 'depth_bin'

    Returns:
        DataFrame con per ogni classe: N_Episodi, % Completati, TTR_Mediano,
        TTR_Media, IQR, TTR_Max
    """
    if "depth_bin" not in episodes_df.columns or episodes_df.empty:
        return pd.DataFrame()

    # Conteggio completati per classe
    completed = (
        episodes_df[~episodes_df["is_censored"]]
        .groupby("depth_bin", observed=False)
        .size()
        .reset_index(name="Completati")
    )

    result = (
        episodes_df
        .groupby("depth_bin", observed=False)["ttr_days"]
        .agg(
            N_Episodi="count",
            TTR_Mediano="median",
            TTR_Media="mean",
            IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
            TTR_Max="max",
        )
        .reset_index()
        .rename(columns={"depth_bin": "Classe"})
        .merge(
            completed.rename(columns={"depth_bin": "Classe"}),
            on="Classe",
            how="left",
        )
    )

    result["Completati"] = result["Completati"].fillna(0).astype(int)
    result["% Completati"] = (result["Completati"] / result["N_Episodi"] * 100).round(1)

    # Mantieni solo classi con almeno 1 episodio
    return result[result["N_Episodi"] > 0].reset_index(drop=True)


# ============================================================
# MONTE CARLO TTR
# ============================================================

def simulate_ttr_montecarlo(
    episodes_df: pd.DataFrame,
    current_depth_pct: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Simulazione Monte Carlo della distribuzione del TTR atteso dato un drawdown corrente.

    Metodologia: bootstrap stratificato.
    1. Si selezionano gli episodi storici con profondità >= |current_depth_pct|
       (almeno altrettanto profondi del drawdown in corso).
    2. Se meno di 5 episodi comparabili, si usa l'intero storico come fallback.
    3. Si campiona con reinserimento dai TTR di quegli episodi per N_SIM iterazioni.
    4. Si calcola la distribuzione percentilare dei TTR simulati.

    Questa approccio è empirico e non assume una distribuzione parametrica,
    il che lo rende robusto per asset con distribuzioni fat-tailed (crypto, indici
    in periodi di stress).

    Args:
        episodes_df:       Episodi storici (output di calculate_ttr_episodes)
        current_depth_pct: Drawdown corrente (valore negativo, es. -0.15 per -15%)
        n_simulations:     Numero di simulazioni bootstrap (default 10.000)
        seed:              Seed per riproducibilità

    Returns:
        Dict con: simulated_ttrs (lista), percentiles (dict), prob_recovery_by_days (dict),
                  n_comparable, used_fallback (bool)
    """
    if episodes_df.empty:
        return {}

    rng = np.random.default_rng(seed)

    # Episodi storici con profondità >= quella corrente (almeno altrettanto gravi)
    comparable   = episodes_df[episodes_df["depth_pct"] <= current_depth_pct]
    used_fallback = False

    if len(comparable) < 5:
        # Fallback: usa tutto il campione storico
        comparable    = episodes_df.copy()
        used_fallback = True

    ttr_pool  = comparable["ttr_days"].values
    simulated = rng.choice(ttr_pool, size=n_simulations, replace=True).astype(float)

    # Percentili chiave
    percentiles = {
        "p10":   int(np.percentile(simulated, 10)),
        "p25":   int(np.percentile(simulated, 25)),
        "p50":   int(np.percentile(simulated, 50)),
        "p75":   int(np.percentile(simulated, 75)),
        "p90":   int(np.percentile(simulated, 90)),
        "p95":   int(np.percentile(simulated, 95)),
        "media": round(float(np.mean(simulated)), 1),
    }

    # Probabilità di recovery entro N giorni di borsa
    checkpoints = [10, 21, 63, 126, 252, 504]
    prob_by_days = {
        str(d): round(float((simulated <= d).mean() * 100), 1)
        for d in checkpoints
    }

    return {
        "simulated_ttrs":        simulated.tolist(),
        "percentiles":           percentiles,
        "prob_recovery_by_days": prob_by_days,
        "n_comparable":          int(len(comparable)),
        "used_fallback":         used_fallback,
        "current_depth_pct":     round(current_depth_pct * 100, 2),
    }


# ============================================================
# EXPORT JSON (per analisi LLM esterna)
# ============================================================

def build_export_json(
    ticker: str,
    price_series: pd.Series,
    episodes_df: pd.DataFrame,
    summary_stats: dict,
    conditional_df: pd.DataFrame,
    kmf: KaplanMeierFitter,
) -> str:
    """
    Serializza l'intero studio TTR in un JSON strutturato per analisi LLM esterna.

    Il JSON include tutti i dati quantitativi in formato leggibile da un modello
    linguistico: statistiche aggregate, episodi completi, sopravvivenza KM,
    analisi condizionale. Intenzionalmente esclude i raw OHLCV per mantenere
    il file compatto e focalizzato sulle metriche di rischio.

    Args:
        ticker:         Simbolo analizzato
        price_series:   Serie prezzi (per metadati)
        episodes_df:    Episodi drawdown completi
        summary_stats:  Output di compute_summary_stats()
        conditional_df: Output di compute_conditional_analysis()
        kmf:            Modello Kaplan-Meier fittato

    Returns:
        Stringa JSON indentata
    """
    # Episodi: conversione date in stringhe per JSON
    episodes_records = []
    for _, row in episodes_df.iterrows():
        rec = {
            "peak_date":      str(row["peak_date"].date()) if pd.notna(row["peak_date"]) else None,
            "trough_date":    str(row["trough_date"].date()) if pd.notna(row["trough_date"]) else None,
            "recovery_date":  str(row["recovery_date"].date()) if pd.notna(row["recovery_date"]) else None,
            "depth_pct":      round(float(row["depth_pct"]) * 100, 2),
            "ttr_days":       int(row["ttr_days"]),
            "is_censored":    bool(row["is_censored"]),
            "depth_class":    str(row["depth_bin"]) if pd.notna(row["depth_bin"]) else None,
        }
        episodes_records.append(rec)

    # Sopravvivenza KM: campioni ogni 20 giorni fino al max TTR
    km_data = []
    if kmf is not None:
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        for t in sf.index:
            km_data.append({
                "days":            int(t),
                "survival_prob":   round(float(sf.iloc[sf.index.get_loc(t), 0]), 4),
                "ci_lower":        round(float(ci.iloc[ci.index.get_loc(t), 0]), 4),
                "ci_upper":        round(float(ci.iloc[ci.index.get_loc(t), 1]), 4),
            })

    # Analisi condizionale
    cond_records = []
    if not conditional_df.empty:
        for _, row in conditional_df.iterrows():
            cond_records.append({
                "depth_class":   str(row.get("Classe", "")),
                "n_episodi":     int(row.get("N_Episodi", 0)),
                "pct_completati": float(row.get("% Completati", 0)),
                "ttr_mediano":   round(float(row.get("TTR_Mediano", 0)), 1),
                "ttr_media":     round(float(row.get("TTR_Media", 0)), 1),
                "iqr":           round(float(row.get("IQR", 0)), 1),
                "ttr_max":       int(row.get("TTR_Max", 0)),
            })

    payload = {
        "meta": {
            "ticker":          ticker,
            "generato_il":     pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "fonte_dati":      "EODHD Historical Data API",
            "descrizione":     (
                "Studio quantitativo completo di Drawdown e Time to Recovery (TTR). "
                "Contiene: statistiche aggregate, log episodi, funzione di sopravvivenza "
                "Kaplan-Meier e analisi condizionale per classe di profondità."
            ),
        },
        "statistiche_riepilogo": summary_stats,
        "analisi_condizionale":  cond_records,
        "kaplan_meier_survival": km_data,
        "episodi_drawdown":      episodes_records,
    }

    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
