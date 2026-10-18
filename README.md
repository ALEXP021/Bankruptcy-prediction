# Corporate Bankruptcy Prediction (V4FinBench)

A full-stack Machine Learning application for predicting corporate bankruptcy across **6 time horizons (1–6 years)**. Built with **XGBoost** and deployed as a **Streamlit** web app with a **FastAPI** backend.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 📄 **Full technical documentation (PL):** [Full technical documentation (PL): Dokumentacja_Modelu.md](Dokumentacja_Modelu.md).

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
```
Bankruptcy-Prediction/
├── README.md
├── .gitignore
├── requirements.txt
├── app_3.py                  # Streamlit application
├── api.py                    # FastAPI backend
├── calibrate_models.py       # Calibration script
├── normalize_bundle.py       # Bundle normalization (grid → app format)
├── train_grid_v3.py          # Training grid search
├── check_bundle.py           # Bundle diagnostics
├── data/
│   └── README.md             # Dataset description & download link
├── docs/
│   └── Dokumentacja_Modelu.md  # Full technical documentation (PL)
└── models/
    └── (trained model files - not tracked)
```

### Technologies Used
- **Python 3.10+**
- **XGBoost** — gradient boosting
- **Streamlit** — web application
- **FastAPI** — backend API
- **Scikit-learn** — preprocessing, calibration
- **SHAP** — model explainability
- **Pandas, NumPy** — data processing
- **Joblib** — model serialization
- **Docker** — containerization (optional)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ALEXP021/Bankruptcy-Prediction.git
   cd Bankruptcy-Prediction
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Download the dataset from Kaggle and place it in the data/ folder.
2. Run the Streamlit app:
   ```bash
   streamlit run app_3.py
   ```
3. (Optional) Run the FastAPI backend:
   ```bash
   uvicorn api:app --reload
   ```
4. Open the app in your browser: http://localhost:8501

### Documentation
Full technical documentation (PL): Dokumentacja_Modelu.md (https://github.com/ALEXP021/Bankruptcy-prediction/blob/main/Dokumentacja_Modelu.md) — covers methodology, metrics, calibration, XAI, economic impact, limitations, and deployment instructions.

### Future Improvements
□ Add more feature engineering (interaction terms, ratios).
□ Hyperparameter tuning with Optuna.
□ Deploy to a permanent cloud service (Streamlit Cloud, Render, AWS).
□ Add unit tests for API endpoints.
□ Add monitoring for model drift.

Author
Oleksii Panin — [LinkedIn](https://www.linkedin.com/in/oleksii-panin-69a509342/) | [GitHub](https://github.com/ALEXP021)
_________________________________________________________________________________________________________________________________________________

## 🇵🇱 Polski

### Przegląd projektu
Projekt dostarcza sześć **niezależnych sygnałów ryzyka** bankructwa przedsiębiorstw na horyzontach od **1 do 6 lat** przed zdarzeniem. Został zrealizowany jako projekt Capstone w **infoShare Academy** (program Data Science + AI).

Model bazowy to **XGBoost**, wytrenowany na zbiorze **V4FinBench** (~1 mln obserwacji firmowo-letnich, ~190 000 unikalnych firm z Polski, Czech, Słowacji i Węgier).

### Dlaczego 6 horyzontów?
Pojedynczy model nie jest w stanie uchwycić zmieniającej się dynamiki kryzysu finansowego. Dlatego wytrenowaliśmy sześć osobnych modeli XGBoost, każdy zoptymalizowany pod konkretny horyzont predykcji:
| Horyzont | Znaczenie | Zastosowanie |
|---------|---------|----------|
| **h1** | 1 rok przed bankructwem | Detekcja firm już w kryzysie |
| **h2** | 2 lata przed bankructwem | Prognoza krótkoterminowa, decyzje kredytowe |
| **h3** | 3 lata przed bankructwem |Ocena ryzyka średnioterminowego |
| **h4** | 4 lata przed bankructwem | Monitoring ekspozycji długoterminowej |
| **h5** | 5 lat przed bankructwem| Stress testing portfela |
| **h6** | 6 lat przed bankructwem | Analiza strategiczna i makro |

### Wyniki modeli (per horyzont)
| 
|Horyzont | PR-AUC Lift | Recall | Precision | F2 | ROC-AUC | Wiarygodność |
|---------|-------------|--------|-----------|-----|---------|-------------|
| **h1** | **96.9×** | 0.560 | 0.402 | 0.510 | 0.9951 | 🟢 wysoka |
| **h2** | **26.4×** | 0.402 | 0.070 | 0.234 | 0.9377 | 🟡 średnia |
| **h3** | **14.4×** | 0.297 | 0.056 | 0.183 | 0.8916 | 🟡 średnia |
| **h4** | 8.5× | 0.182 | 0.036 | 0.104 | 0.8579 | 🔴 niska |
| **h5** | 9.6× | 0.199 | 0.022 | 0.085 | 0.8428 | 🔴 niska |
| **h6** | **10.8×** | 0.243 | 0.021 | 0.084 | 0.8333 | 🟡 średnia |

**Kluczowy wniosek**: h1 najlepszy do detekcji (już w kryzysie), h2 to najlepszy kompromis między horyzontem a dokładnością, h4–h6 przydatne jako sygnały wczesnego ostrzegania na poziomie portfela.

### Kluczowe funkcje
- **Dwa warianty bundle**:
  - models_bundle_hybrid.pkl — oryginalny (lepszy ranking, bez prawdziwych prawdopodobieństw).
  - models_bundle_hybrid_calibrated.pkl — skalibrowany (prawdziwe prawdopodobieństwa, Brier score 10–100× lepszy).
- **Kalibracja prawdopodobieństw**: Isotonic Regression.
- **Wyjaśnialność (XAI)**: wartości SHAP dla każdej predykcji.
- **Aplikacja Streamlit**: wybór bundle, horyzontu, formularz 30–50 cech, predykcja w czasie rzeczywistym.
- **REST API (FastAPI)**: do integracji z systemami transakcyjnymi.
- **Batch processing**: scoring całych portfeli.

### Zbiór danych
- **Źródło**: [V4FinBench Dataset on Kaggle](https://www.kaggle.com/datasets/sebastiantomczak10/v4-group-corporate-bankruptcy/data)
- **Rozmiar**: ~1 mln obserwacji firmowo-letnich (3+ GB).
- **Cechy**: ~95–100 wskaźników numerycznych (rentowność, płynność, zadłużenie, efektywność, rynek, makro).
- **Cel**: Klasyfikacja binarna (bankructwo / brak bankructwa) dla każdego horyzontu.

> ⚠️ Uwaga: Zbiór danych nie jest dołączony do repozytorium ze względu na rozmiar. Pobierz go z Kaggle i umieść w folderze data/.

### Struktura repozytorium
```
Bankruptcy-Prediction/
├── README.md
├── .gitignore
├── requirements.txt
├── app_3.py                  # Streamlit application
├── api.py                    # FastAPI backend
├── calibrate_models.py       # Calibration script
├── normalize_bundle.py       # Bundle normalization (grid → app format)
├── train_grid_v3.py          # Training grid search
├── check_bundle.py           # Bundle diagnostics
├── data/
│   └── README.md             # Dataset description & download link
├── docs/
│   └── Dokumentacja_Modelu.md  # Full technical documentation (PL)
└── models/
    └── (trained model files - not tracked)
```

### Wykorzystane technologie
- **Python 3.10+**
- **XGBoost** — gradient boosting
- **Streamlit** — aplikacja webowa
- **FastAPI** — backend API
- **Scikit-learn** — preprocessing, kalibracja
- **SHAP** — wyjaśnialność modelu
- **Pandas, NumPy** — przetwarzanie danych
- **Joblib** — serializacja modeli
- **Docker** — konteneryzacja (opcjonalnie)

### Instalacja
1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/ALEXP021/Bankruptcy-Prediction.git
   cd Bankruptcy-Prediction
   ```
2. Utwórz środowisko wirtualne::
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```

### Użycie
1. Pobierz zbiór danych z Kaggle i umieść go w folderze data/.
2. Uruchom aplikację Streamlit:
    ```bash
   streamlit run app_3.py
   ```
3. (Opcjonalnie) Uruchom backend FastAPI::
   ```bash
   uvicorn api:app --reload
   ```
4. Otwórz aplikację w przeglądarce:  http://localhost:8501

### Dokumentacja
Pełna dokumentacja techniczna (PL): [Full technical documentation (PL): Dokumentacja_Modelu.md](Dokumentacja_Modelu.md) — obejmuje metodologię, metryki, kalibrację, XAI, efekt ekonomiczny, ograniczenia i instrukcję wdrożenia.

### Przyszłe ulepszenia
□ Dodanie inżynierii cech (interakcje, wskaźniki).
□ Strojenie hiperparametrów za pomocą Optuna.
□ Wdrożenie na stały hosting (Streamlit Cloud, Render, AWS).
□ Dodanie testów jednostkowych dla endpointów API.
□ Dodanie monitoringu dryfu modelu.

Autor
Oleksii Panin — [LinkedIn](https://www.linkedin.com/in/oleksii-panin-69a509342/) | [GitHub](https://github.com/ALEXP021)
