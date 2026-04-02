"""
data_fetcher.py — Fetch e caching dati EODHD.

Tutte le funzioni di fetch usano @st.cache_data (TTL 1 ora) per evitare
chiamate API ridondanti. La storia scaricata è sempre completa (dal dato
più antico disponibile a oggi), senza filtri di data configurabili dall'utente.
"""

import time
import requests
import pandas as pd
import streamlit as st
from functools import wraps


# ============================================================
# UTILITY: RETRY DECORATOR
# ============================================================

def _retry(max_retries: int = 3, delay: float = 2.0):
    """
    Decorator per retry automatico su errori HTTP transitori (429, 5xx).
    Non ritenta su errori 4xx non transitori (401 Unauthorized, 404 Not Found).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response is not None else 0
                    # Retry solo su errori server (5xx) e rate limit (429)
                    if code in (429, 500, 502, 503) and attempt < max_retries:
                        wait = delay * attempt
                        time.sleep(wait)
                    else:
                        raise
                except requests.exceptions.RequestException:
                    if attempt < max_retries:
                        time.sleep(delay * attempt)
                    else:
                        raise
        return wrapper
    return decorator


# ============================================================
# FETCH STORIA COMPLETA — EODHD
# ============================================================

@_retry(max_retries=3, delay=2.0)
def _raw_fetch(ticker: str, api_key: str) -> list:
    """
    Chiamata grezza all'API EODHD EOD.
    Usa 'from=1970-01-01' per garantire tutta la storia disponibile.
    Restituisce la lista JSON grezza.
    """
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "api_token": api_key,
        "from":      "1970-01-01",   # massima storia disponibile
        "period":    "d",
        "fmt":       "json",
        "order":     "a",            # ascending (dal più vecchio al più recente)
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_history(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Scarica tutta la storia EOD disponibile per un ticker EODHD.
    Nessun filtro temporale: restituisce tutti i dati dall'IPO/listing ad oggi.

    Il campo di prezzo usato per l'analisi è 'adjusted_close' (aggiustato
    per dividendi e split). Se non disponibile, si usa 'close' come fallback.

    Args:
        ticker:  Simbolo EODHD (es. 'SPY.US', 'BTC-USD.CC', 'VIX.INDX')
        api_key: Chiave API EODHD

    Returns:
        DataFrame con DatetimeIndex e colonne:
        open, high, low, close, volume, adjusted_close
        Vuoto se nessun dato trovato o ticker non valido.
    """
    data = _raw_fetch(ticker, api_key)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Standardizzazione colonne (EODHD restituisce 'adjusted_close')
    rename_map = {"adj_close": "adjusted_close"}
    df = df.rename(columns=rename_map)

    # Selezione colonne standard
    cols_wanted = ["open", "high", "low", "close", "volume", "adjusted_close"]
    present = [c for c in cols_wanted if c in df.columns]
    df = df[present].copy()

    # Conversione numerica (elimina eventuali stringhe o NaN)
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fallback: se adjusted_close manca, usa close
    if "adjusted_close" not in df.columns and "close" in df.columns:
        df["adjusted_close"] = df["close"]

    # Rimuovi righe con adjusted_close mancante (dati corrotti)
    df = df.dropna(subset=["adjusted_close"])

    # Rimuovi prezzi <= 0 (valori anomali)
    df = df[df["adjusted_close"] > 0]

    return df
