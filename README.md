# Corporate Bankruptcy Prediction (V4FinBench)

A full-stack Machine Learning application for predicting corporate bankruptcy within a 1-year horizon (H1). Built with **XGBoost** and deployed as a **Streamlit** web app with a **FastAPI** backend.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🇬🇧 English

### Project Overview
This project predicts the probability of **corporate bankruptcy** within a 1-year horizon using the **V4FinBench** dataset. The model is wrapped in a **Streamlit** web application that allows users to input 50 financial features and get a real-time prediction.

### Business Problem
Financial institutions and investors need to assess the risk of bankruptcy to make informed lending and investment decisions. This tool provides an interactive way to evaluate a company's financial health.

### Dataset
- **Source:** [V4FinBench Dataset on Kaggle](https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data)
- **Features:** 50 financial indicators (e.g., Current assets/short-term liabilities, Working capital/total assets, EBITDA/total assets).
- **Target:** Binary classification (bankruptcy / no bankruptcy) for horizon H1 (1 year).

### Model Performance
| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9951 |
| **Recall (Czułość)** | 0.8384 |
| **Precision (Precyzja)** | 0.3480 |

*Note: The model uses XGBoost with probability calibration (isotonic regression) to improve the reliability of predicted probabilities.*

### Application Features
- **Interactive input:** 50 financial features with example data auto-fill.
- **Real-time prediction:** Probability of bankruptcy within 1 year.
- **Model calibration:** Isotonic regression for reliable probabilities.
- **Deployment:** Streamlit app (temporarily hosted via ngrok).

### Technologies Used
- **Python**
- **XGBoost** — gradient boosting model
- **Streamlit** — web application
- **FastAPI** — backend API
- **Scikit-learn** — preprocessing, calibration
- **Pandas, NumPy** — data processing
- **Ngrok** — temporary hosting

---

## 🇵🇱 Polski

### Przegląd projektu
Projekt przewiduje prawdopodobieństwo **bankructwa przedsiębiorstwa** w horyzoncie 1 roku na podstawie zbioru danych **V4FinBench**. Model został opakowany w aplikację webową **Streamlit**, która pozwala użytkownikowi wprowadzić 50 cech finansowych i otrzymać predykcję w czasie rzeczywistym.

### Problem biznesowy
Instytucje finansowe i inwestorzy potrzebują oceny ryzyka bankructwa, aby podejmować świadome decyzje kredytowe i inwestycyjne. To narzędzie zapewnia interaktywny sposób oceny kondycji finansowej firmy.

### Zbiór danych
- **Źródło:** [V4FinBench Dataset na Kaggle](https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data)
- **Cechy:** 50 wskaźników finansowych (np. Current assets/short-term liabilities, Working capital/total assets, EBITDA/total assets).
- **Cel:** Klasyfikacja binarna (bankructwo / brak bankructwa) dla horyzontu H1 (1 rok).

### Metryki modelu
| Metryka | Wartość |
|---------|---------|
| **ROC-AUC** | 0.9951 |
| **Recall (Czułość)** | 0.8384 |
| **Precision (Precyzja)** | 0.3480 |

*Uwaga: Model wykorzystuje XGBoost z kalibracją prawdopodobieństw (isotonic regression) w celu zwiększenia wiarygodności predykcji.*

### Funkcjonalność aplikacji
- **Interaktywny input:** 50 cech finansowych z automatycznym wypełnieniem przykładu.
- **Predykcja w czasie rzeczywistym:** Prawdopodobieństwo bankructwa w ciągu 1 roku.
- **Kalibracja modelu:** Isotonic regression dla wiarygodnych prawdopodobieństw.
- **Wdrożenie:** Aplikacja Streamlit (tymczasowo hostowana przez ngrok).

### Wykorzystane technologie
- **Python**
- **XGBoost** — model gradient boosting
- **Streamlit** — aplikacja webowa
- **FastAPI** — backend API
- **Scikit-learn** — preprocessing, kalibracja
- **Pandas, NumPy** — przetwarzanie danych
- **Ngrok** — tymczasowy hosting
