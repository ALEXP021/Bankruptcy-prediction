# -*- coding: utf-8 -*-
"""
NORMALIZACJA BUNDLE
Konwertuje bundle z formatu grid (h1__base__with_flag) do formatu app_3.py (h1)
"""

import joblib
import os

KATALOG_PRZETWORZONY = r"C:\Users\afina\Documents\Inforshare\Final project\data\processed"

# Wejście / wyjście
WEJSCIE_ORYG = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3.pkl")
WEJSCIE_CAL = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_v3_calibrated.pkl")
WYJSCIE_ORYG = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid.pkl")
WYJSCIE_CAL = os.path.join(KATALOG_PRZETWORZONY, "models_bundle_hybrid_calibrated.pkl")

# Priorytet wariantów (base jest lepszy niż lag według PR-AUC)
PRIORYTET = ["base__with_flag", "lag__with_flag"]


def normalize_bundle(input_path, output_path):
    """Konwertuje bundle z formatu grid do formatu app_3.py."""
    
    print("\n" + "="*80)
    print(f"🔧 NORMALIZACJA: {os.path.basename(input_path)}")
    print("="*80)
    
    # Wczytaj
    bundle_old = joblib.load(input_path)
    print(f"   Liczba modeli: {len(bundle_old.get('models', {}))}")
    
    # Nowy bundle
    bundle_new = {
        "version": bundle_old.get("version", "3.0"),
        "created_at": bundle_old.get("created_at", ""),
        "horizons": [],
        "models": {},
    }
    
    # Kopiuj metadane kalibracji jeśli są
    if "calibration_method" in bundle_old:
        bundle_new["calibration_method"] = bundle_old["calibration_method"]
        bundle_new["calibration_timestamp"] = bundle_old.get("calibration_timestamp", "")
    
    # Dla każdego horyzontu wybierz najlepszy wariant
    for h in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        # Znajdź dostępne warianty
        dostepne = {}
        for key, art in bundle_old["models"].items():
            if key.startswith(f"{h}__"):
                wariant = key[len(f"{h}__"):]
                dostepne[wariant] = (key, art)
        
        if not dostepne:
            print(f"   ⚠️ Brak modelu dla {h}")
            continue
        
        # Wybierz według priorytetu
        wybrany_key, wybrany_art = None, None
        for wariant in PRIORYTET:
            if wariant in dostepne:
                wybrany_key, wybrany_art = dostepne[wariant]
                break
        
        if wybrany_art is None:
            wybrany_key, wybrany_art = list(dostepne.values())[0]
        
        # Normalizuj artefakt
        art_new = wybrany_art.copy()
        art_new["source"] = "v3.0"
        art_new["source_key"] = wybrany_key
        art_new["horizon"] = h
        
        # Wiarygodność na podstawie PR-AUC lift
        lift = art_new["metrics"].get("pr_auc_lift", 0)
        if lift >= 30:
            art_new["reliability"] = "high"
        elif lift >= 10:
            art_new["reliability"] = "medium"
        else:
            art_new["reliability"] = "low"
        
        # Kalibrator — domyślnie None
        if "calibrator" not in art_new:
            art_new["calibrator"] = None
            art_new["calibrator_type"] = None
        
        bundle_new["horizons"].append(h)
        bundle_new["models"][h] = art_new
        
        m = art_new["metrics"]
        print(f"   ✅ {h}: {wybrany_key}")
        print(f"      PR-AUC={m['pr_auc']:.4f} (lift={m['pr_auc_lift']:.1f}x), "
              f"ROC-AUC={m['roc_auc']:.4f}, "
              f"wiarygodność={art_new['reliability']}")
    
    # Zapisz
    joblib.dump(bundle_new, output_path, compress=3)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n💾 Zapisano: {output_path}")
    print(f"   Rozmiar: {size_mb:.2f} MB")
    print(f"   Horyzonty: {bundle_new['horizons']}")
    print(f"   Modele: {list(bundle_new['models'].keys())}")
    
    return bundle_new


if __name__ == "__main__":
    print("="*80)
    print("🔧 NORMALIZACJA BUNDLE — z formatu grid do formatu app_3.py")
    print("="*80)
    
    # 1. Oryginalny
    if os.path.exists(WEJSCIE_ORYG):
        normalize_bundle(WEJSCIE_ORYG, WYJSCIE_ORYG)
    else:
        print(f"\n⚠️ Brak pliku: {WEJSCIE_ORYG}")
    
    # 2. Skalibrowany
    if os.path.exists(WEJSCIE_CAL):
        normalize_bundle(WEJSCIE_CAL, WYJSCIE_CAL)
    else:
        print(f"\n⚠️ Brak pliku: {WEJSCIE_CAL}")
    
    print("\n" + "="*80)
    print("✅ GOTOWE!")
    print("="*80)
    print("\nZaktualizuj app_3.py:")
    print("  BUNDLE_ORYGINALNY   = 'models_bundle_hybrid.pkl'")
    print("  BUNDLE_SKALIBROWANY = 'models_bundle_hybrid_calibrated.pkl'")