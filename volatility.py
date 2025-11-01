import numpy as np
import pandas as pd

TRADING_DAYS = 252

def realized_vol(prices: pd.Series, method: str = "simple", lambda_ewma: float = 0.94) -> float:
    """
    Annualise la volatilité (retours log).
    - simple: std des log-returns * sqrt(252)
    - ewma: RiskMetrics EWMA, lambda ~ 0.94 par défaut
    """
    logret = np.log(prices).diff().dropna()
    if len(logret) < 20:
        raise ValueError("Historique insuffisant pour estimer la volatilité.")
    if method == "simple":
        return float(logret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    elif method == "ewma":
        w = (1 - lambda_ewma) * (lambda_ewma ** np.arange(len(logret)-1, -1, -1))
        w /= w.sum()
        mu = float((w * logret.values).sum())
        var = float((w * (logret.values - mu) ** 2).sum())
        return float(np.sqrt(var) * np.sqrt(TRADING_DAYS))
    else:
        raise ValueError("Méthode vol inconnue.")
