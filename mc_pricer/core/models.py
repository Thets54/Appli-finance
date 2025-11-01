from __future__ import annotations
import numpy as np

class BaseModel:
    def simulate(self, s0: float, r: float, q: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int | None = None):
        raise NotImplementedError

class GBM(BaseModel):
    """
    dS_t = (r - q) S_t dt + sigma S_t dW_t  (mesure risque-neutre)
    """
    def simulate(self, s0, r, q, sigma, T, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        diff = sigma * np.sqrt(dt)
        z = rng.standard_normal((n_paths, n_steps))
        # antithétiques
        z = np.vstack([z, -z])
        paths = np.empty((z.shape[0], n_steps + 1), dtype=float)
        paths[:, 0] = s0
        np.exp(drift + diff * z, out=z)  # réutilise z pour les facteurs multiplicatifs
        np.cumprod(z, axis=1, out=z)
        paths[:, 1:] = s0 * z
        return paths

class MertonJumps(BaseModel):
    """
    GBM + sauts de Poisson (Merton).
    S_t * J pour chaque saut, avec ln J ~ N(mj, sj^2).
    """
    def __init__(self, jump_intensity=0.2, jump_mu=-0.05, jump_sigma=0.10):
        self.lmbda = jump_intensity
        self.mj = jump_mu
        self.sj = jump_sigma

    def simulate(self, s0, r, q, sigma, T, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        # Ajustement du drift pour préserver la martingale sous Q
        k = np.exp(self.mj + 0.5 * self.sj**2) - 1.0
        drift = (r - q - 0.5 * sigma**2 - self.lmbda * k) * dt
        diff = sigma * np.sqrt(dt)

        z = rng.standard_normal((n_paths, n_steps))
        z = np.vstack([z, -z])

        # sauts
        n_tot = z.shape[0]
        n_steps_arr = np.full((n_tot, n_steps), self.lmbda * dt)
        n_poiss = rng.poisson(n_steps_arr)
        # tailles de sauts, produit cumulatif
        J = np.ones_like(z)
        mask = n_poiss > 0
        # Pour efficacité, on réalise les sauts via log-somme approx: exp(sum ln J)
        # Ici, on échantillonne un nombre de sauts et agrège.
        for i in np.where(mask.any(axis=1))[0]:
            for t in np.where(mask[i])[0]:
                k_t = n_poiss[i, t]
                if k_t > 0:
                    lnJ = rng.normal(self.mj, self.sj, size=k_t).sum()
                    J[i, t] = np.exp(lnJ)

        incr = np.exp(drift + diff * z) * J
        paths = np.empty((n_tot, n_steps + 1), dtype=float)
        paths[:, 0] = s0
        np.cumprod(incr, axis=1, out=incr)
        paths[:, 1:] = s0 * incr
        return paths
