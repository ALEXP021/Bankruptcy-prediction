# Corporate Bankruptcy Prediction (V4FinBench)

A full-stack Machine Learning application for predicting corporate bankruptcy across **6 time horizons (1–6 years)**. Built with **XGBoost** and deployed as a **Streamlit** web app with a **FastAPI** backend.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 📄 **Full technical documentation (PL):** [docs/Dokumentacja_Modelu.md]([docs/Dokumentacja_Modelu.md](https://github.com/ALEXP021/Bankruptcy-prediction/blob/main/Dokumentacja_Modelu.md)).

---

## 🇬🇧 English

### Project Overview
This project provides **six independent risk signals** for corporate bankruptcy prediction, covering horizons from **1 to 6 years** before the event. It was developed as a Capstone project at **infoShare Academy** (Data Science + AI program).

The core model is **XGBoost**, trained on the **V4FinBench** dataset (~1M firm-year observations, ~190,000 unique companies from Poland, Czechia, Slovakia, and Hungary).

### Why 6 Horizons?
A single model cannot capture the changing dynamics of financial distress. That's why we trained **six separate XGBoost models**, each optimized for a specific prediction horizon:

| Horizon | Meaning | Use Case |
|---------|---------|----------|
| **h1** | 1 year before bankruptcy | Detect companies already in crisis |
| **h2** | 2 years before bankruptcy | Short-term forecasting, loan decisions |
| **h3** | 3 years before bankruptcy | Medium-term risk assessment |
| **h4** | 4 years before bankruptcy | Long-term exposure monitoring |
| **h5** | 5 years before bankruptcy | Portfolio stress testing |
| **h6** | 6 years before bankruptcy | Strategic & macro analysis |

### Model Performance (per Horizon)
| Horizon | PR-AUC Lift | Recall | Precision | F2 | ROC-AUC | Reliability |
|---------|-------------|--------|-----------|-----|---------|-------------|
| **h1** | **96.9×** | 0.560 | 0.402 | 0.510 | 0.9951 | 🟢 High |
| **h2** | **26.4×** | 0.402 | 0.070 | 0.234 | 0.9377 | 🟡 Medium |
| **h3** | **14.4×** | 0.297 | 0.056 | 0.183 | 0.8916 | 🟡 Medium |
| **h4** | 8.5× | 0.182 | 0.036 | 0.104 | 0.8579 | 🔴 Low |
| **h5** | 9.6× | 0.199 | 0.022 | 0.085 | 0.8428 | 🔴 Low |
| **h6** | **10.8×** | 0.243 | 0.021 | 0.084 | 0.8333 | 🟡 Medium |

**Key insight:** h1 is best for detection (already in crisis), h2 is the best compromise between horizon and accuracy, h4–h6 are useful as early warning signals at portfolio level.

### Key Features
- **Two bundle variants:**
  - `models_bundle_hybrid.pkl` — original (best ranking, no true probabilities).
  - `models_bundle_hybrid_calibrated.pkl` — calibrated (true probabilities, Brier score 10–100× better).
- **Probability calibration:** Isotonic Regression.
- **Explainability (XAI):** SHAP values for every prediction.
- **Streamlit app:** choose bundle, horizon, input 30–50 features, get real-time prediction.
- **REST API (FastAPI):** for integration with transaction systems.
- **Batch processing:** score entire portfolios.

### Dataset
- **Source:** [V4FinBench Dataset on Kaggle](https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data)
- **Size:** ~1M firm-year observations (3+ GB)
- **Features:** ~95–100 numerical indicators (profitability, liquidity, debt, efficiency, market, macro).
- **Target:** Binary classification (bankruptcy / no bankruptcy) for each horizon.

> ⚠️ **Note:** The dataset is not included in this repository due to its size. Download it from Kaggle and place it in the `data/` folder.

### Repository Structure
