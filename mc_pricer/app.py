import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from core.data import fetch_prices_alpha_vantage, parse_uploaded_csv
from core.volatility import realized_vol
from core.pricer import price_mc, MODELS
from core.payoffs import PAYOFFS

st.set_page_config(page_title="MC Derivatives Pricer", layout="wide")

st.title("🧪 Monte Carlo – Sous-jacent & Dérivés")
st.caption("V1 – App personnelle (flexible & extensible)")

with st.sidebar:
    st.header("Paramètres généraux")
    src = st.radio("Source de données", ["Alpha Vantage", "Upload CSV"], horizontal=True)

    if src == "Alpha Vantage":
        ticker = st.text_input("Ticker (ex: AAPL, MSFT, AIR.PA)", "AAPL")
        hist_months = st.slider("Fenêtre historique (mois)", 6, 24, 12)
        api_key = st.text_input("API Key Alpha Vantage (optionnel)", value="", type="password")
        st.caption("Si vide, j’utiliserai la variable d’environnement ALPHAVANTAGE_API_KEY.")
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

    # La date d'échéance sera choisie plus bas, après chargement des prix (pour connaître la date de valeur).
    # On affiche juste une info ici :
    st.caption("La maturité est choisie par date dans la zone centrale (après chargement des prix).")

    st.markdown("---")
    st.subheader("Modèle")
    model_key = st.selectbox("Choix du modèle", list(MODELS.keys()))
    if model_key == "Merton (GBM + sauts)":
        st.caption("Ajuste les paramètres dans core/models.py au besoin.")

    st.markdown("---")
    st.subheader("Payoff")
    payoff_key = st.selectbox("Type de payoff", list(PAYOFFS.keys()))

    # Paramètres spécifiques aux payoffs
    if payoff_key == "Stability (Range) Digital":
        B_low = st.number_input("Barrière inférieure Bₗ", value=80.0, step=1.0)
        B_high = st.number_input("Barrière supérieure Bᵤ", value=120.0, step=1.0)
        payout = st.number_input("Payout (Digital)", value=10.0, step=1.0)
        K = 0.0  # non utilisé par ce payoff (signature uniforme)
        st.caption("Convention: toucher = knock-out ; monitoring discret aux pas de simulation.")
    else:
        K = st.number_input("Strike K", value=100.0, step=1.0)
        payout = st.number_input("Payout (Digital)", value=1.0, step=0.5)

    st.markdown("---")
    n_steps = st.slider("Pas de temps", 20, 1000, 252)
    n_paths = st.slider("Nombre de trajectoires", 1_000, 200_000, 20_000, step=1_000)
    seed = st.number_input("Seed (optionnel)", value=42, step=1)

    st.markdown("---")
    run = st.button("Lancer la simulation / pricing")

# =========================
# Chargement des prix
# =========================
prices = None
if src == "Alpha Vantage":
    try:
        prices = fetch_prices_alpha_vantage(ticker, months=hist_months, api_key=(api_key or None))
    except Exception as e:
        st.error(f"Erreur Alpha Vantage: {e}")
else:
    if 'up' in locals() and up is not None:
        try:
            prices = parse_uploaded_csv(up, price_col=price_col or None, date_col=date_col or None)
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")

# =========================
# Affichage + vol + choix d'échéance par DATE
# =========================
valuation_date = None
T_years = None

if prices is not None and not prices.empty:
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

    # Choix de maturité par date (à partir de la dernière date disponible)
    valuation_date = pd.to_datetime(prices.index[-1]).date()
    st.markdown("---")
    st.subheader("Échéance (par date)")
    default_maturity = max(valuation_date, date.today())
    maturity_date = st.date_input(
        "Date d'échéance (inclus)",
        value=default_maturity,
        min_value=valuation_date,
        help="La date de valeur est la dernière date de l'historique chargé."
    )

    # Conversion date -> années (base ACT/365.25 simple)
    delta_days = (pd.to_datetime(maturity_date) - pd.to_datetime(valuation_date)).days
    T_years = max(delta_days / 365.25, 1/252)  # garde une borne min (~1 jour de marché)

    st.caption(f"Date de valeur: {valuation_date} · Échéance: {maturity_date} · T ≈ {T_years:.4f} an(s)")

    # =========================
    # Lancement MC
    # =========================
    if run and sigma is not None and T_years is not None:
        # Validation basique des bornes pour Stability
        if payoff_key == "Stability (Range) Digital" and B_low >= B_high:
            st.error("Borne invalide: B_inf doit être strictement < B_sup.")
        else:
            payoff_kwargs = {}
            if "Digital" in payoff_key:
                payoff_kwargs["payout"] = payout
            if payoff_key == "Stability (Range) Digital":
                payoff_kwargs["B_low"] = B_low
                payoff_kwargs["B_high"] = B_high

            price, stderr, paths, payoff = price_mc(
                s0=s0, r=r, q=q, sigma=sigma, T_years=T_years, n_steps=n_steps,
                n_paths=n_paths, model_key=model_key, payoff_key=payoff_key, K=K, seed=int(seed), **payoff_kwargs
            )

            st.success(f"Prix (MC): {price:,.4f}   ·   IC ~ ± {1.96*stderr:,.4f} (95%)")

            # Trajectoires
            st.subheader("Trajectoires simulées")
            n_show = min(200, paths.shape[0])
            st.line_chart(pd.DataFrame(paths[:n_show, :]).T)

            # Distribution payoff
            st.subheader("Distribution du payoff")
            hist, edges = np.histogram(payoff, bins=50)
            centers = [0.5*(edges[i]+edges[i+1]) for i in range(len(hist))]
            hist_df = pd.DataFrame({"count": hist}, index=pd.Index(centers, name="payoff"))
            st.bar_chart(hist_df)

            # Export
            with st.expander("Exporter les résultats"):
                df_paths = pd.DataFrame(paths.T)
                df_paths.index.name = "step"
                st.download_button("Télécharger les trajectoires (CSV)", df_paths.to_csv().encode(), file_name="paths.csv")

                params_txt = (
                    f"s0={s0}\nr={r}\nq={q}\nsigma={sigma}\n"
                    f"valuation_date={valuation_date}\nmaturity_date={maturity_date}\nT={T_years}\n"
                    f"steps={n_steps}\npaths={n_paths}\nmodel={model_key}\npayoff={payoff_key}\nK={K}\n"
                )
                if payoff_key == "Stability (Range) Digital":
                    params_txt += f"B_low={B_low}\nB_high={B_high}\npayout={payout}\n"
                elif "Digital" in payoff_key:
                    params_txt += f"payout={payout}\n"

                st.download_button("Télécharger les paramètres (TXT)", params_txt.encode(), file_name="params.txt")
