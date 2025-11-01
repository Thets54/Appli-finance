from __future__ import annotations
import numpy as np

def european_call(paths: np.ndarray, K: float) -> np.ndarray:
    ST = paths[:, -1]
    return np.maximum(ST - K, 0.0)

def european_put(paths: np.ndarray, K: float) -> np.ndarray:
    ST = paths[:, -1]
    return np.maximum(K - ST, 0.0)

def digital_call(paths: np.ndarray, K: float, payout: float = 1.0) -> np.ndarray:
    ST = paths[:, -1]
    return (ST > K).astype(float) * payout

def digital_put(paths: np.ndarray, K: float, payout: float = 1.0) -> np.ndarray:
    ST = paths[:, -1]
    return (ST < K).astype(float) * payout

def asian_arith_call(paths: np.ndarray, K: float) -> np.ndarray:
    A = paths.mean(axis=1)
    return np.maximum(A - K, 0.0)

def asian_arith_put(paths: np.ndarray, K: float) -> np.ndarray:
    A = paths.mean(axis=1)
    return np.maximum(K - A, 0.0)


def stability_range_digital(paths: np.ndarray, B_low: float, B_high: float, payout: float = 1.0) -> np.ndarray:
    """
    Paye `payout` si le chemin reste strictement entre (B_low, B_high) sur toute la période.
    Monitoring discret sur la grille simulée. Par défaut, toucher une barrière knock-out.
    Pour accepter le toucher, remplace les comparaisons strictes par <= / >=.
    """
    inside = (paths > B_low) & (paths < B_high)   # toucher = knock-out
    ok = inside.all(axis=1)
    return ok.astype(float) * payout

PAYOFFS = {
    "European Call": lambda P, K, **_: european_call(P, K),
    "European Put":  lambda P, K, **_: european_put(P, K),
    "Digital Call":  lambda P, K, payout=1.0, **_: digital_call(P, K, payout=payout),
    "Digital Put":   lambda P, K, payout=1.0, **_: digital_put(P, K, payout=payout),
    "Asian (arith) Call": lambda P, K, **_: asian_arith_call(P, K),
    "Asian (arith) Put":  lambda P, K, **_: asian_arith_put(P, K),
    "Stability (Range) Digital": lambda P, K, B_low=None, B_high=None, payout=1.0, **_: (stability_range_digital(P, B_low, B_high, payout=payout)),
}
