# TTR Drawdown Dashboard — Kriterion Quant

Dashboard Streamlit per l'analisi quantitativa del **Drawdown** e del **Time to Recovery (TTR)**
di asset finanziari. Dati storici completi via EODHD API.

---

## Funzionalità

### Pagina 1 — Analisi Singolo Asset
- Equity line con evidenziazione episodi di drawdown
- Drawdown % continuo rispetto all'HWM
- Funzione di sopravvivenza Kaplan-Meier
- Analisi condizionale TTR per classe di profondità
- Analisi di regime (VIX o SMA200, configurabile)
- Tabella episodi filtrata interattivamente
- Export JSON per analisi LLM

### Pagina 2 — Comparazione Multi-Asset
- Heatmap TTR Mediano per asset × classe di profondità
- Tabella comparativa risk metrics (Calmar, Ulcer, Pain, Recovery Factor)
- Bar chart comparativi
- Overlay curve Kaplan-Meier

### Pagina 3 — Monte Carlo TTR
- Simulazione bootstrap empirica (N=10.000 default)
- Dato un drawdown corrente: distribuzione TTR atteso
- Istogramma + CDF cumulativa
- Tabella probabilità recovery per orizzonte temporale
- Log episodi storici comparabili

---

## Installazione e Avvio Locale

```bash
# 1. Clona il repository
git clone https://github.com/tuo-utente/ttr-drawdown-dashboard.git
cd ttr-drawdown-dashboard

# 2. Crea e attiva un virtualenv (opzionale ma consigliato)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Configura la API key EODHD
# Crea il file .streamlit/secrets.toml con:
# EODHD_API_KEY = "la-tua-chiave"

# 5. Avvia la dashboard
streamlit run app.py
```

---

## Deploy su Streamlit Cloud

1. Push del repository su GitHub (assicurati che `.streamlit/secrets.toml` sia in `.gitignore`)
2. Vai su [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Seleziona il repository e `app.py` come entry point
4. In **Advanced settings → Secrets**, incolla:
   ```toml
   EODHD_API_KEY = "la-tua-chiave-eodhd"
   ```
5. Clicca **Deploy**

---

## Sintassi Ticker EODHD

| Tipo Asset | Suffisso | Esempi |
|-----------|---------|--------|
| Azioni & ETF USA | `.US` | `SPY.US`, `AAPL.US`, `QQQ.US` |
| Azioni Europa | codice exchange | `ENI.MI`, `BMW.XETRA`, `BP.LSE` |
| Crypto | `.CC` | `BTC-USD.CC`, `ETH-USD.CC`, `SOL-USD.CC` |
| Forex | `.FOREX` | `EURUSD.FOREX`, `GBPUSD.FOREX` |
| Indici | `.INDX` | `GSPC.INDX` (S&P 500), `NDX.INDX` (Nasdaq 100), `VIX.INDX`, `GDAXI.INDX` (DAX), `FCHI.INDX` (CAC 40), `N225.INDX` (Nikkei), `STOXX50E.INDX` (EuroStoxx 50) |
| Futures/Commodities | `.COMM` | `GC.COMM` (Oro), `CL.COMM` (WTI), `ES.COMM` |

---

## Struttura Repository

```
ttr-drawdown-dashboard/
├── app.py                        # Pagina 1: Singolo Asset
├── pages/
│   ├── 2_Multi_Asset.py          # Pagina 2: Comparazione Multi-Asset
│   └── 3_Monte_Carlo.py          # Pagina 3: Simulazione Monte Carlo
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py           # Fetch EODHD con caching
│   ├── analytics.py              # TTR, KM, risk metrics, Monte Carlo
│   └── charts.py                 # Grafici Plotly dark-theme
├── .streamlit/
│   ├── config.toml               # Tema dark Streamlit
│   └── secrets.toml              # API key (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dipendenze Principali

| Package | Versione | Uso |
|---------|---------|-----|
| streamlit | ≥ 1.32 | Framework UI |
| plotly | ≥ 5.20 | Grafici interattivi dark-theme |
| lifelines | ≥ 0.27 | Kaplan-Meier survival analysis |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.26 | Calcoli numerici, bootstrap |
| requests | ≥ 2.31 | Chiamate EODHD API |

---

*Kriterion Quant — Finanza Quantitativa Applicata*
