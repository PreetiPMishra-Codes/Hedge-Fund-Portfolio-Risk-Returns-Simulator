# Hedge-Fund-Portfolio-Risk-Returns-Simulator
An institutional-grade portfolio risk analysis engine and simulator built from scratch using vectorized mathematical computations. This application bypasses slow iterative loops in favor of linear algebra matrix operations to ingest historical market data, calculate portfolio return dynamics, and model worst-case risk scenarios.Designed specifically as a lightweight, interactive utility for quantitative evaluation and trend projection.
# 🚀 Core Architecture & FeaturesAutomated Data Ingestion:
Pipelines 3–5 years of historical, split-and-dividend-adjusted asset prices via yfinance API with clean time-series forward-filling handling.
Vectorized Portfolio Math: Utilizes NumPy matrix multiplication ($W^T \cdot \Sigma \cdot W$) to compute historical variance, annualized volatility, and expected returns across dynamic asset allocations instantly.
Risk Quantification Engine: Deploys a Historical Simulation Value-at-Risk (VaR) framework to compute explicit 95% and 99% portfolio daily capital loss expectations.
Trend Projection (ML Layer): Evaluates non-neural time-series trajectories using an optimized Scikit-Learn linear regression model to project risk trendlines 30 days forward.
Interactive Dashboard: Packaged inside a responsive, browser-based Streamlit user interface featuring dynamic weight allocation adjustment sliders and reactive metric visualization.
# 🛠️ Technical StackLanguage: 
Python 3.x
Data Manipulation: Pandas,NumPy (Linear Algebra / Vectorization Engine)
Modeling: Scikit-Learn (Ridge/Linear Regression)
Frontend/UI: Streamlit Web Framework
