from __future__ import annotations
import numpy as np
from .models import GBM, MertonJumps
from .payoffs import PAYOFFS

MODELS = {
    "GBM (Black-Scholes)": GBM(),
    "Merton (GBM + sauts)": MertonJumps(),  # paramètres éditables dans l'UI
}

def price_mc(
    s0: float, r: float, q: float, sigma: float, T_years: float, n_steps: int, n_paths: int,
    model_key: str, payoff_key: str, K: float, seed: int | None = None, **payoff_kwargs
):
    model = MODELS[model_key]
    paths = model.simulate(s0, r, q, sigma, T_years, n_steps, n_paths, seed=seed)
    payoff = PAYOFFS[payoff_key](paths, K, **payoff_kwargs)
    disc = np.exp(-r * T_years)
    price = float(disc * payoff.mean())
    stderr = float(disc * payoff.std(ddof=1) / np.sqrt(len(payoff)))
    return price, stderr, paths, payoff
