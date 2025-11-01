from __future__ import annotations
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from core.key import API_KEY

AV_DAILY_URL = "https://www.alphavantage.co/query"

def _get_api_key(explicit_key: str | None = None) -> str:
    key = explicit_key or os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise ValueError(
            "Clé API Alpha Vantage introuvable. "
            "Passe-la via l'argument `api_key=` ou la variable d'environnement ALPHAVANTAGE_API_KEY."
        )
    return key

def _alpha_vantage_daily_adjusted(symbol: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    """
    Télécharge TIME_SERIES_DAILY_ADJUSTED (toutes dates) puis renvoie un DataFrame indexé par date.
    Colonnes standard d'Alpha Vantage (string keys) -> renommées en propres.
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,  # 'compact' ~ 100 derniers jours, 'full' tout l'historique
        "datatype": "json",
    }
    r = requests.get(AV_DAILY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage: {data['Error Message']}")
    if "Note" in data:
        # Note typique quand limite de débit atteinte
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    ts = data.get("Time Series (Daily)")
    if not ts:
        raise ValueError("Réponse Alpha Vantage invalide: 'Time Series (Daily)' manquant.")

    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index).tz_localize(None).sort_values()
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
            "8. split coefficient": "split_coeff",
        }
    )
    # convertir en float
    for c in ["open", "high", "low", "close", "adj_close", "dividend", "split_coeff"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df

# core/data.py

def _alpha_vantage_daily(symbol: str, api_key: str, adjusted: bool = True, outputsize: str = "full") -> pd.DataFrame:
    """
    Tente de télécharger Daily (ajusté si adjusted=True), sinon lève une erreur claire.
    """
    fn = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    params = {
        "function": fn,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,  # 'compact' recommandé pour éviter le rate limit
        "datatype": "json",
    }
    r = requests.get(AV_DAILY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Messages d'erreur fréquents (plan/quotas/symbole)
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage Error: {data['Error Message']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage Information: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")

    # Clé de time series (diffère selon endpoint)
    ts = data.get("Time Series (Daily)") or data.get("Time Series (Daily) Adjusted")
    if not ts:
        # aide au debug: liste les clés renvoyées
        keys = ", ".join(list(data.keys())[:6])
        raise ValueError(f"Réponse Alpha Vantage invalide: time series manquante. Clés vues: {keys}")

    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index).tz_localize(None).sort_values()

    # renommage tolérant (ajusté ou non)
    ren = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend",
        "8. split coefficient": "split_coeff",
    }
    df = df.rename(columns=ren)

    # s'il n'y a pas 'adj_close' (endpoint brut), on le remplit avec 'close'
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # cast
    for c in ["open", "high", "low", "close", "adj_close", "dividend", "split_coeff"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df


def fetch_prices_alpha_vantage(symbol: str, months: int = 12, api_key: str | None = None) -> pd.Series:
    """
    Renvoie une Série des *Adjusted Close* si dispo, sinon 'close'.
    Fallback: DAILY si DAILY_ADJUSTED indisponible (plan/quotas).
    """
    months = int(max(6, min(24, months)))
    key = _get_api_key(api_key)

    try:
        df = _alpha_vantage_daily(symbol, key, adjusted=True, outputsize="compact")
    except (RuntimeError, ValueError):
        # fallback sur daily brut (souvent dispo en gratuit)
        df = _alpha_vantage_daily(symbol, key, adjusted=False, outputsize="compact")

    if df.empty or "adj_close" not in df.columns:
        raise ValueError(f"Aucun prix exploitable pour {symbol} via Alpha Vantage.")

    end = df.index.max()
    # ~ mois → fenêtre glissante
    start = end - timedelta(days=int(months * 30.4375))
    s = df.loc[df.index >= start, "adj_close"].dropna()
    if s.empty:
        raise ValueError(f"Pas de données suffisantes pour {symbol} sur {months} mois.")
    s.name = symbol
    return s


def parse_uploaded_csv(file, price_col: str | None = None, date_col: str | None = None) -> pd.Series:
    """
    Lecture d'un CSV utilisateur. Détection heuristique des colonnes Date / Prix si non fournie.
    """
    df = pd.read_csv(file)
    if date_col is None:
        cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = cands[0] if cands else df.columns[0]
    if price_col is None:
        cands = [c for c in df.columns if any(k in c.lower() for k in ["adj", "close", "price", "px"])]
        price_col = cands[0] if cands else df.columns[-1]
    s = pd.Series(df[price_col].values, index=pd.to_datetime(df[date_col]), name=price_col).sort_index()
    s = s.dropna()
    s.index = s.index.tz_localize(None)
    return s
