"""
charts.py — Funzioni per la creazione di grafici Plotly dark-theme standardizzati.

Ogni funzione restituisce un go.Figure pronto per st.plotly_chart().
Tutte condividono la palette COLORS e il layout _base_layout().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter


# ============================================================
# PALETTE E LAYOUT BASE
# ============================================================

COLORS = {
    "primary":    "#2196F3",   # blu — equity line, linea principale
    "secondary":  "#FF9800",   # arancio — linea secondaria / segnali
    "positive":   "#4CAF50",   # verde — recovery, bull regime
    "negative":   "#F44336",   # rosso — drawdown, bear regime
    "neutral":    "#9E9E9E",   # grigio — riferimenti, zero line
    "background": "#1E1E2E",   # sfondo scuro
    "surface":    "#2A2A3E",   # card/pannelli
    "text":       "#E0E0E0",   # testo principale
    "accent":     "#AB47BC",   # viola — indicatori speciali / MC
    "accent2":    "#00BCD4",   # ciano — terza serie
    "grid":       "#333355",   # colore griglia
}

# Palette per sequenze di asset multipli
MULTI_COLORS = [
    "#2196F3", "#FF9800", "#4CAF50", "#F44336",
    "#AB47BC", "#00BCD4", "#FFC107", "#E91E63",
]


def _base_layout(
    title: str,
    x_title: str = "",
    y_title: str = "",
    y2_title: str = "",
    height: int = 450,
) -> dict:
    """Layout Plotly condiviso e professionale per tutti i grafici."""
    base = dict(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"]), x=0.01),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, color=COLORS["text"],
            tickfont=dict(color=COLORS["text"]),
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, color=COLORS["text"],
            tickfont=dict(color=COLORS["text"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(color=COLORS["text"]),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=55, b=50),
        height=height,
    )
    if y2_title:
        base["yaxis2"] = dict(
            title=y2_title,
            overlaying="y", side="right",
            showgrid=False,
            color=COLORS["text"],
            tickfont=dict(color=COLORS["text"]),
        )
    return base


# ============================================================
# GRAFICO 1: EQUITY LINE + EPISODI DI DRAWDOWN
# ============================================================

def build_equity_drawdown_chart(
    price_series: pd.Series,
    episodes_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """
    Equity line con evidenziazione colorata degli episodi di drawdown.

    Ogni episodio è rappresentato da un rettangolo rosso semitrasparente,
    con opacità proporzionale alla profondità del drawdown. Al hover,
    mostra profondità e TTR dell'episodio.

    Args:
        price_series: Serie prezzi adjusted_close
        episodes_df:  Episodi drawdown da calculate_ttr_episodes()
        ticker:       Simbolo per il titolo

    Returns:
        go.Figure interattivo con equity line e drawdown shading
    """
    fig = go.Figure()

    # Equity line
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        name="Adjusted Close",
        line=dict(color=COLORS["primary"], width=1.5),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))

    # Shading per ogni episodio
    for _, row in episodes_df.iterrows():
        depth        = row["depth_pct"]
        peak_date    = row["peak_date"]
        # Fine del rettangolo: recovery_date se disponibile, altrimenti ultimo dato
        end_date = row["recovery_date"] if pd.notna(row["recovery_date"]) \
                   else price_series.index[-1]
        opacity      = min(0.08 + abs(depth) * 0.55, 0.55)
        ttr_label    = f"{int(row['ttr_days'])}gg" + (" ⏳" if row["is_censored"] else "")
        hover_text   = f"Peak: {str(peak_date.date())}<br>DD: {depth:.1%}<br>TTR: {ttr_label}"

        fig.add_vrect(
            x0=peak_date, x1=end_date,
            fillcolor=COLORS["negative"],
            opacity=opacity,
            line_width=0,
            annotation_text=f"{depth:.0%}",
            annotation_position="top left",
            annotation_font_size=8,
            annotation_font_color=COLORS["text"],
        )

    fig.update_layout(**_base_layout(
        f"{ticker} — Equity Line e Periodi di Drawdown",
        x_title="Data",
        y_title="Prezzo Adjusted",
        height=480,
    ))
    return fig


# ============================================================
# GRAFICO 2: DRAWDOWN PERCENTUALE NEL TEMPO
# ============================================================

def build_drawdown_series_chart(
    price_series: pd.Series,
    ticker: str,
) -> go.Figure:
    """
    Grafico del drawdown percentuale continuo rispetto all'High-Water Mark.

    Visualizza in ogni istante di quanto si è sotto il massimo storico.
    L'area riempita permette di cogliere immediatamente la profondità e
    la durata complessiva dei periodi di stress.

    Args:
        price_series: Serie prezzi adjusted_close
        ticker:       Simbolo per il titolo

    Returns:
        go.Figure con area chart del drawdown %
    """
    hwm      = price_series.cummax()
    dd_pct   = (price_series / hwm - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_pct.index,
        y=dd_pct.values,
        name="Drawdown %",
        fill="tozeroy",
        fillcolor="rgba(244, 67, 54, 0.25)",
        line=dict(color=COLORS["negative"], width=1.2),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8, line_dash="dot")

    fig.update_layout(**_base_layout(
        f"{ticker} — Drawdown % rispetto al Massimo Storico (HWM)",
        x_title="Data",
        y_title="Drawdown (%)",
        height=300,
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


# ============================================================
# GRAFICO 3: FUNZIONE DI SOPRAVVIVENZA KAPLAN-MEIER
# ============================================================

def build_kaplan_meier_chart(
    kmf_list: list,
    ticker: str = "",
    show_ci: bool = True,
) -> go.Figure:
    """
    Grafico della funzione di sopravvivenza Kaplan-Meier.

    S(t) = probabilità che un drawdown duri PIÙ di t giorni di borsa.
    Esempio di lettura: se S(252) = 0.12, il 12% dei drawdown storici
    non si è risolto entro un anno di mercato.

    Può ricevere una lista di KMF (per confronto regimi) o un singolo.
    L'intervallo di confidenza al 95% (area semitrasparente) quantifica
    l'incertezza della stima, maggiore quando ci sono pochi episodi.

    Args:
        kmf_list:  Lista di KaplanMeierFitter già fittati
        ticker:    Simbolo per il titolo
        show_ci:   Se True mostra l'intervallo di confidenza 95%

    Returns:
        go.Figure con curve di sopravvivenza
    """
    if not kmf_list:
        return go.Figure()

    fig  = go.Figure()
    # Linee di riferimento orizzontali
    for level in [0.75, 0.50, 0.25]:
        fig.add_hline(
            y=level, line_dash="dot",
            line_color=COLORS["neutral"], line_width=0.7,
            annotation_text=f"{level:.0%}",
            annotation_position="right",
            annotation_font_color=COLORS["neutral"],
            annotation_font_size=10,
        )

    for i, kmf in enumerate(kmf_list):
        color = MULTI_COLORS[i % len(MULTI_COLORS)]
        sf    = kmf.survival_function_
        ci    = kmf.confidence_interval_
        label = kmf.label if hasattr(kmf, "label") else f"Serie {i+1}"

        # Confidence interval (area)
        if show_ci and ci is not None and not ci.empty:
            lower_col = ci.columns[0]
            upper_col = ci.columns[1]
            fig.add_trace(go.Scatter(
                x=list(ci.index) + list(ci.index[::-1]),
                y=list(ci[upper_col]) + list(ci[lower_col][::-1]),
                fill="toself",
                fillcolor=f"rgba({_hex_to_rgb(color)}, 0.12)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Curva di sopravvivenza (step)
        fig.add_trace(go.Scatter(
            x=sf.index,
            y=sf.iloc[:, 0],
            name=label,
            mode="lines",
            line=dict(color=color, width=2, shape="hv"),  # shape="hv" = step function
            hovertemplate="Giorno %{x:.0f}: S(t)=%{y:.3f}<extra>" + label + "</extra>",
        ))

    title = f"{ticker} — Funzione di Sopravvivenza dei Drawdown (Kaplan-Meier)" \
            if ticker else "Funzione di Sopravvivenza dei Drawdown (Kaplan-Meier)"
    fig.update_layout(**_base_layout(title, x_title="Giorni di Borsa (TTR)", y_title="P(drawdown > t)"))
    fig.update_yaxes(range=[-0.02, 1.05])
    return fig


# ============================================================
# GRAFICO 4: BOXPLOT TTR PER CLASSE DI PROFONDITÀ
# ============================================================

def build_ttr_boxplot(
    episodes_df: pd.DataFrame,
    ticker: str = "",
) -> go.Figure:
    """
    Boxplot del Time to Recovery suddiviso per classe di profondità del drawdown.

    Ogni box mostra la distribuzione del TTR per una fascia di profondità.
    L'asse y è in scala logaritmica per gestire la forte asimmetria positiva
    della distribuzione (pochi drawdown molto lunghi allungano la coda destra).

    Un box allargato indica alta incertezza sui tempi di recovery;
    un box stretto ma posizionato in alto indica drawdown storicamente lenti
    a risolversi ma con comportamento abbastanza prevedibile.

    Args:
        episodes_df: Output di calculate_ttr_episodes()
        ticker:      Simbolo per il titolo

    Returns:
        go.Figure con boxplot per classe di profondità (log scale)
    """
    if episodes_df.empty or "depth_bin" not in episodes_df.columns:
        return go.Figure()

    fig = go.Figure()

    bins_present = [b for b in episodes_df["depth_bin"].cat.categories
                    if b in episodes_df["depth_bin"].values]

    for i, depth_class in enumerate(bins_present):
        subset = episodes_df[episodes_df["depth_bin"] == depth_class]["ttr_days"]
        if subset.empty:
            continue

        color = MULTI_COLORS[i % len(MULTI_COLORS)]
        fig.add_trace(go.Box(
            y=subset.values,
            name=str(depth_class),
            marker_color=color,
            line_color=color,
            fillcolor=f"rgba({_hex_to_rgb(color)}, 0.30)",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.6,
            hovertemplate="Classe: " + str(depth_class) +
                          "<br>TTR: %{y:.0f} gg<extra></extra>",
        ))

    title = f"{ticker} — Distribuzione TTR per Classe di Profondità (scala log)" \
            if ticker else "Distribuzione TTR per Classe di Profondità"
    fig.update_layout(**_base_layout(title, y_title="Time to Recovery (giorni)"))
    fig.update_yaxes(type="log", tickformat=".0f")
    return fig


# ============================================================
# GRAFICO 5: MULTI-ASSET HEATMAP TTR MEDIANO
# ============================================================

def build_multi_asset_heatmap(heatmap_pivot: pd.DataFrame) -> go.Figure:
    """
    Heatmap comparativa del TTR mediano per asset e classe di profondità.

    Colori: rosso = recovery lento, verde = recovery veloce (RdYlGn inverso).
    Permette confronto visivo immediato della resilienza di diversi asset:
    oro, indici, crypto, settori, etc.

    Args:
        heatmap_pivot: DataFrame pivot (index=Asset, columns=Classe, values=TTR_Mediano)

    Returns:
        go.Figure con heatmap annotata
    """
    if heatmap_pivot.empty:
        return go.Figure()

    # Valori per annotazioni (interi leggibili)
    z_vals  = heatmap_pivot.values
    text    = [[f"{v:.0f}gg" if not np.isnan(v) else "—"
                for v in row] for row in z_vals]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=list(heatmap_pivot.columns),
        y=list(heatmap_pivot.index),
        text=text,
        texttemplate="%{text}",
        textfont=dict(color=COLORS["text"], size=11),
        colorscale=[
            [0.0,  "#4CAF50"],   # verde — recovery veloce
            [0.5,  "#FFC107"],   # giallo — medio
            [1.0,  "#F44336"],   # rosso — recovery lento
        ],
        colorbar=dict(
            title=dict(text="TTR Mediano (gg)", font=dict(color=COLORS["text"])),
            tickfont=dict(color=COLORS["text"]),
        ),
        hoverongaps=False,
        hovertemplate="Asset: %{y}<br>Classe: %{x}<br>TTR Mediano: %{z:.0f} gg<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        "Heatmap Comparativa — TTR Mediano per Asset e Classe di Profondità",
        x_title="Classe di Profondità",
        y_title="Asset",
        height=max(300, len(heatmap_pivot) * 55 + 100),
    ))
    return fig


# ============================================================
# GRAFICO 6: MONTE CARLO — ISTOGRAMMA + CURVA RECOVERY CUMULATIVA
# ============================================================

def build_montecarlo_chart(mc_results: dict) -> go.Figure:
    """
    Grafico Monte Carlo: istogramma simulazioni TTR + curva probabilità recovery cumulativa.

    Il pannello superiore mostra la distribuzione delle simulazioni bootstrap.
    Le linee verticali indicano i percentili chiave (p25, p50, p75, p90).
    Il pannello inferiore mostra la probabilità cumulativa di recovery entro
    N giorni, leggibile come: "c'è il 60% di probabilità di uscire dal drawdown
    entro 90 giorni di borsa".

    Args:
        mc_results: Output di simulate_ttr_montecarlo()

    Returns:
        go.Figure con subplot doppio (istogramma + CDF)
    """
    if not mc_results or "simulated_ttrs" not in mc_results:
        return go.Figure()

    sims = np.array(mc_results["simulated_ttrs"])
    pct  = mc_results["percentiles"]
    depth_str = f"{mc_results.get('current_depth_pct', 0):.1f}%"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Distribuzione TTR Simulata (Bootstrap, n={len(sims):,})",
            "Probabilità Cumulativa di Recovery entro N giorni",
        ),
        vertical_spacing=0.14,
        shared_xaxes=False,
    )

    # --- Pannello 1: Istogramma ---
    fig.add_trace(go.Histogram(
        x=sims,
        nbinsx=60,
        name="Simulazioni",
        marker_color=COLORS["accent"],
        opacity=0.75,
        hovertemplate="TTR: %{x:.0f} gg<br>Freq: %{y}<extra></extra>",
    ), row=1, col=1)

    pct_lines = [
        ("p25", COLORS["positive"],   "P25"),
        ("p50", COLORS["secondary"],  "P50 (Mediana)"),
        ("p75", COLORS["negative"],   "P75"),
        ("p90", COLORS["accent2"],    "P90"),
    ]
    for key, color, name in pct_lines:
        v = pct.get(key, None)
        if v is not None:
            fig.add_vline(x=v, line_dash="dash", line_color=color, line_width=1.5,
                          annotation_text=f"{name}: {v}gg",
                          annotation_position="top right",
                          annotation_font_color=color,
                          annotation_font_size=10,
                          row=1, col=1)

    # --- Pannello 2: CDF (probabilità recovery cumulativa) ---
    max_t   = int(np.percentile(sims, 97))
    t_vals  = np.arange(1, max_t + 1)
    cdf     = np.array([(sims <= t).mean() * 100 for t in t_vals])

    fig.add_trace(go.Scatter(
        x=t_vals, y=cdf,
        name="P(recovery ≤ t)",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.15)",
        hovertemplate="Giorno %{x:.0f}: %{y:.1f}% probabilità recovery<extra></extra>",
    ), row=2, col=1)

    # Linee guida orizzontali
    for level in [25, 50, 75, 90]:
        fig.add_hline(y=level, line_dash="dot", line_color=COLORS["neutral"],
                      line_width=0.7, row=2, col=1)

    # Formattazione
    fig.update_layout(
        title=dict(
            text=f"Monte Carlo TTR — Drawdown Corrente: {depth_str}",
            font=dict(size=15, color=COLORS["text"]), x=0.01,
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=COLORS["grid"]),
        height=620,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], color=COLORS["text"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], color=COLORS["text"])
    fig.update_xaxes(title_text="Time to Recovery (giorni di borsa)", row=1, col=1)
    fig.update_yaxes(title_text="Frequenza", row=1, col=1)
    fig.update_xaxes(title_text="Giorni di Borsa", row=2, col=1)
    fig.update_yaxes(title_text="Prob. Recovery (%)", ticksuffix="%", row=2, col=1)

    return fig


# ============================================================
# GRAFICO 7: CONFRONTO RISK METRICS MULTI-ASSET (BAR)
# ============================================================

def build_comparative_bar(
    stats_list: list,
    metric: str = "ttr_mediano_gg",
    y_label: str = "TTR Mediano (giorni)",
) -> go.Figure:
    """
    Bar chart comparativo per una metrica di rischio su più asset.

    Args:
        stats_list: Lista di dict (output di compute_summary_stats) con chiave 'ticker'
        metric:     Chiave della metrica da confrontare
        y_label:    Etichetta asse y

    Returns:
        go.Figure con bar chart orizzontale
    """
    if not stats_list:
        return go.Figure()

    tickers = [s.get("ticker", f"Asset {i}") for i, s in enumerate(stats_list)]
    values  = [s.get(metric, 0) for s in stats_list]

    # Colori: verde se basso (recovery veloce), rosso se alto
    max_v   = max(v for v in values if v is not None and not np.isnan(v)) or 1
    colors  = [
        f"rgba({_hex_to_rgb(_lerp_color(v / max_v))}, 0.8)"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=tickers,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" if v is not None else "N/A" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        f"Confronto Multi-Asset — {y_label}",
        y_title=y_label,
        height=380,
    ))
    return fig


# ============================================================
# UTILITY COLORI
# ============================================================

def _hex_to_rgb(hex_color: str) -> str:
    """Converte colore hex in stringa 'R, G, B' per rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"


def _lerp_color(t: float) -> str:
    """
    Interpolazione lineare verde → rosso in base a t ∈ [0, 1].
    t=0 → verde (#4CAF50), t=1 → rosso (#F44336).
    """
    t = max(0.0, min(1.0, t))
    g_r, g_g, g_b = 0x4C, 0xAF, 0x50
    r_r, r_g, r_b = 0xF4, 0x43, 0x36
    rr = int(g_r + (r_r - g_r) * t)
    gg = int(g_g + (r_g - g_g) * t)
    bb = int(g_b + (r_b - g_b) * t)
    return f"#{rr:02X}{gg:02X}{bb:02X}"
