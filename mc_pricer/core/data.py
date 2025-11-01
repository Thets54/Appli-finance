from __future__ import annotations
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_prices_yf(ticker: str, months: int = 12) -> pd.Series:
    """
    Télécharge des cours 'Adj Close' pour `ticker` sur `months` derniers mois.
    Renvoie une série pandas indexée par date (UTC naïf).
    """
    months = max(6, min(24, months))
    end = datetime.utcnow()
    start = end - timedelta(days=int(months * 30.4375))
    df = yf.download(ticker, start=start.date(), end=end.date(), auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Aucun historique pour {ticker}.")
    px = df["Adj Close"].dropna().rename(ticker)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px

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
