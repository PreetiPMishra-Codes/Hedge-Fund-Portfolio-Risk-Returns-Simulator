# 📊 Hedge-Fund-Portfolio-Risk-Returns-Simulator

An institutional-grade portfolio risk analysis engine and simulator built from scratch using vectorized mathematical computations. This application bypasses slow iterative loops in favor of linear algebra matrix operations to ingest historical market data, calculate portfolio return dynamics, and model worst-case risk scenarios. 

Designed specifically as a lightweight, interactive utility for quantitative evaluation and trend projection.
A quantitative portfolio analysis tool that fetches real historical stock data,
computes risk/return metrics using vectorized NumPy/Pandas operations, and
projects short-term trends with a lightweight ML model — packaged in an
interactive Streamlit dashboard.

## Status
🚧 In progress — built incrementally as part of a learning project.

---

## 🚀 Core Architecture & Features

*   **Automated Data Ingestion:** Pipelines 3–5 years of historical, split-and-dividend-adjusted asset prices via the `yfinance` API with clean time-series forward-filling handling.
*   **Vectorized Portfolio Math:** Utilizes NumPy matrix multiplication ($W^T \cdot \Sigma \cdot W$) to compute historical variance, annualized volatility, and expected returns across dynamic asset allocations instantly.
*   **Risk Quantification Engine:** Deploys a Historical Simulation Value-at-Risk (VaR) framework to compute explicit 95% and 99% portfolio daily capital loss expectations.
*   **Trend Projection (ML Layer):** Evaluates non-neural time-series trajectories using an optimized Scikit-Learn linear regression model to project risk trendlines 30 days forward.
*   **Interactive Dashboard:** Packaged inside a responsive, browser-based Streamlit user interface featuring dynamic weight allocation adjustment sliders and reactive metric visualization.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy (Linear Algebra / Vectorization Engine) |
| **Modeling & ML** | Scikit-Learn (Ridge / Linear Regression) |
| **Frontend / UI** | Streamlit Web Framework |
