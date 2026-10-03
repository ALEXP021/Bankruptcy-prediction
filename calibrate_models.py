# -*- coding: utf-8 -*-
"""
KALIBRACJA MODELI v3
Dodaje kalibrację (isotonic + Platt) do modeli z models_bundle_v3.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

# ============================================================
# KONFIGURACJA
# ============================================================
KATALOG_DANYCH = r"C:\Users\afina\Documents\Inforshare\Final project\data"
KATALOG_PANEL = os.path.join(KATALOG_DANYCH, "panel")
KATALOG_PRZETWORZONY = os.path.join(KATALOG_DANYCH, "processed")

BUNDLE_PATH = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3.pkl")
OUTPUT_PATH = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3_calibrated.pkl")

CALIBRATION_METHOD = "isotonic"  # "isotonic" lub "platt"

# ============================================================
# FUNKCJE KALIBRACJI
# ============================================================
def calibrate_isotonic(proba_val, y_val, proba_test):
    """Kalibracja izotoniczna."""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(proba_val, y_val)
    proba_test_cal = iso_reg.predict(proba_test)
    return iso_reg, proba_test_cal


def calibrate_platt(proba_val, y_val, proba_test):
    """Kalibracja Platt scaling (sigmoid)."""
    eps = 1e-8
    proba_val_clip = np.clip(proba_val, eps, 1 - eps)
    proba_test_clip = np.clip(proba_test, eps, 1 - eps)
    
    logit_val = np.log(proba_val_clip / (1 - proba_val_clip)).reshape(-1, 1)
    logit_test = np.log(proba_test_clip / (1 - proba_test_clip)).reshape(-1, 1)
    
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(logit_val, y_val)
    proba_test_cal = platt.predict_proba(logit_test)[:, 1]
    
    return platt, proba_test_cal


# ============================================================
# GŁÓWNA LOGIKA
# ============================================================
def main():
    print("="*80)
    print("🔧 KALIBRACJA MODELI v3")
    print("="*80)
    
    # Wczytaj bundle
    print(f"\n📦 Wczytuję bundle: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)
    
    print(f"   Wersja: {bundle.get('version')}")
    print(f"   Horyzonty: {bundle.get('horizons')}")
    print(f"   Liczba modeli: {len(bundle.get('models', {}))}")
    
    # Dla każdego modelu w bundle
    calibrated_models = {}
    
    for key, art in bundle["models"].items():
        print(f"\n{'='*80}")
        print(f"🎯 {key}")
        print(f"{'='*80}")
        
        h = art["horizon"]
        top_n = art["top_n"]
        drop_insolvency = art["drop_insolvency"]
        use_lags = art["use_lags"]
        features = art["features"]
        median_fill = art["median_fill"]
        
        print(f"   Horyzont: {h}, TOP-N: {top_n}")
        print(f"   drop_insolvency: {drop_insolvency}, use_lags: {use_lags}")
        
        # Wczytaj dane
        if use_lags:
            path = os.path.join(KATALOG_PANEL, f"company_years_{h}_lagged.parquet")
        else:
            path = os.path.join(KATALOG_DANYCH, f"company_years_{h}.parquet")
        
        df = pd.read_parquet(path)
        df = df.dropna(subset=['main_label']).copy()
        df['main_label'] = df['main_label'].astype(int)
        
        company_id = df['emis_id'].values
        
        # Podział
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(df, df['main_label'], groups=company_id))
        
        # Przygotuj X
        drop_cols = [c for c in ['company','link','industry','num','emis_id','year','main_label']
                     if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(pd.Series(median_fill))
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = df['main_label'].values[train_idx]
        y_test = df['main_label'].values[test_idx]
        g_train = company_id[train_idx]
        
        # Wybierz cechy
        X_train_top = X_train[features]
        X_test_top = X_test[features]
        
        # Podział train na train/val (do kalibracji)
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
        tr2_idx, val_idx = next(sgkf_val.split(X_train_top, y_train, groups=g_train))
        
        X_tr2 = X_train_top.iloc[tr2_idx].reset_index(drop=True)
        X_val = X_train_top.iloc[val_idx].reset_index(drop=True)
        y_tr2, y_val = y_train[tr2_idx], y_train[val_idx]
        
        # Trening modelu na X_tr2 (do kalibracji)
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        model_for_cal = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=2,
            scale_pos_weight=ratio, random_state=42,
            eval_metric='aucpr', n_jobs=-1
        )
        model_for_cal.fit(X_tr2, y_tr2)
        
        proba_val = model_for_cal.predict_proba(X_val)[:, 1]
        
        # Predykcja oryginalnym modelem z bundle
        proba_test = art["model"].predict_proba(X_test_top)[:, 1]
        
        # Kalibracja
        if (y_val == 1).sum() >= 5:
            if CALIBRATION_METHOD == "isotonic":
                calibrator, proba_test_cal = calibrate_isotonic(proba_val, y_val, proba_test)
            else:
                calibrator, proba_test_cal = calibrate_platt(proba_val, y_val, proba_test)
            
            # Brier score
            brier_before = float(brier_score_loss(y_test, proba_test))
            brier_after = float(brier_score_loss(y_test, proba_test_cal))
            
            # ROC-AUC i PR-AUC (nie powinny się zmienić — kalibracja monotoniczna)
            roc_before = float(roc_auc_score(y_test, proba_test))
            roc_after = float(roc_auc_score(y_test, proba_test_cal))
            pr_before = float(average_precision_score(y_test, proba_test))
            pr_after = float(average_precision_score(y_test, proba_test_cal))
            
            print(f"\n   📊 Kalibracja ({CALIBRATION_METHOD}):")
            print(f"      Brier score:  {brier_before:.6f} → {brier_after:.6f} "
                  f"({'✅ poprawa' if brier_after < brier_before else '⚠️ pogorszenie'})")
            print(f"      ROC-AUC:      {roc_before:.4f} → {roc_after:.4f}")
            print(f"      PR-AUC:       {pr_before:.4f} → {pr_after:.4f}")
            
            # Aktualizuj artefakt
            art["calibrator"] = calibrator
            art["calibrator_type"] = CALIBRATION_METHOD
            art["brier_before"] = brier_before
            art["brier_after"] = brier_after
            art["metrics"]["brier_before"] = brier_before
            art["metrics"]["brier_after"] = brier_after
            art["metrics"]["brier_improvement"] = brier_before - brier_after
        else:
            print(f"\n   ⚠️ Zbyt mało bankructw w val ({int((y_val == 1).sum())}) — pomijam kalibrację")
            art["calibrator"] = None
            art["calibrator_type"] = None
        
        calibrated_models[key] = art
    
    # Zapisz nowy bundle
    bundle["models"] = calibrated_models
    bundle["calibration_method"] = CALIBRATION_METHOD
    bundle["calibration_timestamp"] = pd.Timestamp.now().isoformat()
    
    joblib.dump(bundle, OUTPUT_PATH, compress=3)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\n💾 Zapisano: {OUTPUT_PATH}")
    print(f"   Rozmiar: {size_mb:.2f} MB")
    
    # Podsumowanie
    print("\n" + "="*100)
    print("📊 PODSUMOWANIE KALIBRACJI")
    print("="*100)
    
    summary = []
    for key, art in calibrated_models.items():
        if art.get("brier_before") is not None:
            summary.append({
                "key": key,
                "brier_before": art["brier_before"],
                "brier_after": art["brier_after"],
                "improvement": art["brier_before"] - art["brier_after"],
                "calibrator": art.get("calibrator_type"),
            })
    
    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
    else:
        print("Brak skalibrowanych modeli")


if __name__ == "__main__":
    main()