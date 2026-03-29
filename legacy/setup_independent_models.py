"""
Script pour créer une structure complète et indépendante pour les modèles ETH et SOL
Chaque dossier contiendra tout le nécessaire: data, scripts, models, documentation
"""

import os
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
CRYPTOS = ['ETH', 'SOL']

def setup_all():
    """Setup complet pour ETH et SOL"""

    for crypto in CRYPTOS:
        print(f"\n{'='*60}")
        print(f"Configuration de {crypto}_MODEL...")
        print(f"{'='*60}")

        model_dir = BASE_DIR / f"{crypto}_MODEL"

        # 1. Copier les modèles entraînés
        src_model = BASE_DIR / "direction_prediction_system" / "models_direction" / crypto / "xgboost_ultimate"
        dst_model = model_dir / "models" / "xgboost_ultimate"

        if src_model.exists() and not dst_model.exists():
            shutil.copytree(src_model, dst_model)
            print(f"[OK] Modèle {crypto} copié")

        # 2. Créer requirements.txt
        requirements = model_dir / "requirements.txt"
        if not requirements.exists():
            requirements.write_text("""pandas>=1.5.0
numpy>=1.23.0
xgboost>=1.7.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
ccxt>=4.0.0
ta>=0.10.0
""")
            print(f"[OK] requirements.txt créé")

        print(f"\n{crypto}_MODEL setup terminé!")

    print(f"\n{'='*60}")
    print("SETUP TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*60}")
    print("\nStructure créée:")
    print("  ETH_MODEL/ - Modèle ETH autonome")
    print("  SOL_MODEL/ - Modèle SOL autonome")
    print("\nProchaines étapes:")
    print("1. cd ETH_MODEL && python train_model.py")
    print("2. cd ETH_MODEL && python backtest.py")
    print("3. cd ETH_MODEL && python production_inference.py")


if __name__ == "__main__":
    setup_all()
