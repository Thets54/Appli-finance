import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data import fetch_prices_yf, parse_uploaded_csv
from core.volatility import realized_vol
from core.pricer import price_mc, MODELS
from core.payoffs import PAYOFFS

st.set_page_config(page_title="MC Derivatives Pricer", layout="wide")

st.title("🧪 Monte Carlo – Sous-jacent & Dérivés")
st.caption("V1 – App personnelle (flexible & extensible)")

with st.sidebar:
    st.header("Paramètres généraux")
    src = st.radio("Source de données", ["Yahoo Finance", "Upload CSV"], horizontal=True)

    if src == "Yahoo Finance":
        ticker = st.text_input("Ticker (ex: AAPL, MSFT, AIR.PA)", "AAPL")
        hist_months = st.slider("Fenêtre historique (mois)", 6, 24, 12)
    else:
        up = st.file_uploader("CSV (Date, Price)", type=["csv"])
        date_col = st.text_input("Nom colonne date (optionnel)", "")
        price_col = st.text_input("Nom colonne prix (optionnel)", "")
        hist_months = 12  # non utilisé

    st.markdown("---")
    st.subheader("Estimation de volatilité")
    vol_method = st.selectbox("Méthode", ["simple", "ewma"])
    lam = st.slider("λ EWMA (si EWMA)", 0.80, 0.99, 0.94, step=0.01)

    st.markdown("---")
    st.subheader("Marché / Horizon")
    r = st.number_input("Taux sans risque r (annualisé)", value=0.02, step=0.005, format="%.4f")
    q = st.number_input("Dividende q (annualisé)", value=0.00, step=0.005, format="%.4f")
    T_years = st.number_input("Échéance (années)", value=0.5, min_value=1/252, step=0.25, format="%.4f")
    n_steps = st.slider("Pas de temps", 20, 1000, 252)
    n_paths = st.slider("Nombre de trajectoires", 1_000, 200_000, 20_000, step=1_000)
    seed = st.number_input("Seed (optionnel)", value=42, step=1)

    st.markdown("---")
    st.subheader("Modèle")
    model_key = st.selectbox("Choix du modèle", list(MODELS.keys()))
    if model_key == "Merton (GBM + sauts)":
        st.caption("Ajuste les paramètres dans core/models.py au besoin.")

    st.markdown("---")
    st.subheader("Payoff")
    payoff_key = st.selectbox("Type de payoff", list(PAYOFFS.keys()))
    K = st.number_input("Strike K", value=100.0, step=1.0)
    payout = st.number_input("Payout (Digital)", value=1.0, step=0.5)

    st.markdown("---")
    run = st.button("Lancer la simulation / pricing")

# Chargement des prix & estimation vol
prices = None
if src == "Yahoo Finance":
    try:
        prices = fetch_prices_yf(ticker, months=hist_months)
    except Exception as e:
        st.error(f"Erreur download YF: {e}")
else:
    if up is not None:
        try:
            prices = parse_uploaded_csv(up, price_col=price_col or None, date_col=date_col or None)
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")

if prices is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historique")
        st.line_chart(prices)
    with col2:
        try:
            sigma = realized_vol(prices, method=vol_method, lambda_ewma=lam)
            st.metric("Vol annualisée", f"{sigma:.2%}")
            s0 = float(prices.iloc[-1])
            st.metric("Spot (S₀)", f"{s0:,.2f}")
        except Exception as e:
            st.error(f"Volatilité: {e}")
            sigma = None
            s0 = None

    if run and sigma is not None:
        # Monte Carlo pricing
        payoff_kwargs = {}
        if "Digital" in payoff_key:
            payoff_kwargs["payout"] = payout

        price, stderr, paths, payoff = price_mc(
            s0=s0, r=r, q=q, sigma=sigma, T_years=T_years, n_steps=n_steps,
            n_paths=n_paths, model_key=model_key, payoff_key=payoff_key, K=K, seed=int(seed), **payoff_kwargs
        )

        st.success(f"Prix (MC): {price:,.4f}   ·   IC ~ ± {1.96*stderr:,.4f} (95%)")

        # Graph 1: Trajectoires
        st.subheader("Trajectoires simulées")
        n_show = min(200, paths.shape[0])
        st.line_chart(pd.DataFrame(paths[:n_show, :]).T)

        # Graph 2: Distribution terminale
        st.subheader("Distribution du payoff")
        hist, edges = np.histogram(payoff, bins=50)
        hist_df = pd.DataFrame({"count": hist}, index=pd.Index([0.5*(edges[i]+edges[i+1]) for i in range(len(hist))], name="payoff"))
        st.bar_chart(hist_df)

        # Export
        with st.expander("Exporter les résultats"):
            df_paths = pd.DataFrame(paths.T)
            df_paths.index.name = "step"
            st.download_button("Télécharger les trajectoires (CSV)", df_paths.to_csv().encode(), file_name="paths.csv")
            st.download_button("Télécharger les paramètres (TXT)", 
                               f"s0={s0}\nr={r}\nq={q}\nsigma={sigma}\nT={T_years}\nsteps={n_steps}\npaths={n_paths}\nmodel={model_key}\npayoff={payoff_key}\nK={K}\n".encode(),
                               file_name="params.txt")
