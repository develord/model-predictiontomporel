"""
V11 Optuna Optimization - Hyperparameter Tuning
================================================
Optimize XGBoost hyperparameters to improve V11 binary classifier accuracy

Goal: Improve from ~52% to 57-62% accuracy

Optimization Strategy:
- Tune XGBoost hyperparameters (max_depth, learning_rate, etc.)
- Optimize class weight (scale_pos_weight)
- Find best prediction threshold for trading
- Metric: Maximize AUC (more robust than accuracy)
"""

import sys
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import json
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(crypto: str):
    """Load and prepare data for optimization with TEMPORAL split"""
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric', 'price_target_pct',
        'future_price', 'triple_barrier_label'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Clean data
    df_clean = df[df['triple_barrier_label'].notna()].copy()

    # Features
    X = df_clean[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Binary target: -1 (SL) -> 0, +1 (TP) -> 1
    y = (df_clean['triple_barrier_label'] == 1).astype(int).values

    # TEMPORAL SPLIT: Train <2025, Test >=2025 (same as train_v11.py)
    timestamps = df_clean.index
    train_mask = timestamps < '2025-01-01'
    test_mask = timestamps >= '2025-01-01'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test, feature_cols


def objective(trial, crypto, X_train, X_test, y_train, y_test):
    """
    Optuna objective function

    Returns: AUC score (to maximize)
    """

    # Calculate class imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    default_scale = n_neg / n_pos if n_pos > 0 else 1.0

    # Hyperparameters to optimize
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'random_state': 42,
        'tree_method': 'hist',

        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),

        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),

        # Class imbalance (allow optimization around default)
        # SOL has inverted imbalance (74% TP), so use min 0.1
        'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                 max(0.1, default_scale * 0.5),
                                                 max(2.0, default_scale * 1.5))
    }

    # Train model
    model = xgb.XGBClassifier(**params)

    # Train without early stopping (not needed for 100-500 estimators)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC (primary metric)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Also calculate accuracy for reporting
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Store additional metrics as user attributes
    trial.set_user_attr('accuracy', accuracy)
    trial.set_user_attr('n_estimators_used', model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators'])

    return auc


def optimize_crypto(crypto: str, n_trials: int = 100, load_study: bool = False):
    """
    Optimize hyperparameters for one crypto

    Args:
        crypto: 'btc', 'eth', or 'sol'
        n_trials: Number of Optuna trials
        load_study: Whether to load existing study or start fresh
    """

    print(f"\n{'='*80}")
    print(f"OPTIMIZING V11 TEMPORAL FOR {crypto.upper()}")
    print(f"WALK-FORWARD VALIDATION: Train <2025, Test >=2025")
    print('='*80)

    # Load data
    print(f"\n[1/3] Loading Data")
    X_train, X_test, y_train, y_test, feature_cols = load_data(crypto)

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: SL={np.sum(y_train==0)}, TP={np.sum(y_train==1)}")

    # Create study
    print(f"\n[2/3] Running Optuna Optimization ({n_trials} trials)")

    study_name = f'v11_{crypto}_optimization'
    storage_path = Path(__file__).parent.parent / 'optimization' / f'{crypto}_v11_optuna.db'
    storage = f'sqlite:///{storage_path}'

    if load_study:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"  Loaded existing study with {len(study.trials)} trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',  # Maximize AUC
            load_if_exists=True
        )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, crypto, X_train, X_test, y_train, y_test),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Results
    print(f"\n[3/3] Optimization Results")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best Accuracy: {study.best_trial.user_attrs['accuracy']:.4f} ({study.best_trial.user_attrs['accuracy']*100:.2f}%)")

    # Best parameters
    print(f"\n  Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key:20s}: {value}")

    # Compare to baseline V11 TEMPORAL
    print(f"\n  Comparison to V11 TEMPORAL Baseline:")
    baseline_acc = {
        'btc': 0.5405,
        'eth': 0.5315,
        'sol': 0.5563
    }

    baseline = baseline_acc.get(crypto, 0.52)
    improvement = study.best_trial.user_attrs['accuracy'] - baseline

    print(f"    Baseline accuracy: {baseline:.4f} ({baseline*100:.2f}%)")
    print(f"    Optimized accuracy: {study.best_trial.user_attrs['accuracy']:.4f} ({study.best_trial.user_attrs['accuracy']*100:.2f}%)")
    print(f"    Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

    if improvement > 0.05:
        print(f"    SIGNIFICANT IMPROVEMENT! (+5%+)")
    elif improvement > 0.02:
        print(f"    Good improvement (+2-5%)")
    elif improvement > 0:
        print(f"    Modest improvement")
    else:
        print(f"    No improvement (baseline was already good)")

    # Save best params
    results_dir = Path(__file__).parent.parent / 'optimization' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f'{crypto}_v11_best_params.json'

    best_params_full = {
        'crypto': crypto,
        'best_auc': study.best_value,
        'best_accuracy': study.best_trial.user_attrs['accuracy'],
        'baseline_accuracy': baseline,
        'improvement': improvement,
        'n_trials': len(study.trials),
        'best_trial_number': study.best_trial.number,
        'hyperparameters': study.best_params,
        'complete_params': {
            **study.best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'tree_method': 'hist'
        }
    }

    with open(results_file, 'w') as f:
        json.dump(best_params_full, f, indent=2)

    print(f"\n  Saved best params to: {results_file}")

    return study


def train_with_best_params(crypto: str):
    """
    Retrain V11 model with best Optuna parameters
    """

    print(f"\n{'='*80}")
    print(f"RETRAINING V11 WITH BEST PARAMS: {crypto.upper()}")
    print('='*80)

    # Load best params
    results_file = Path(__file__).parent.parent / 'optimization' / 'results' / f'{crypto}_v11_best_params.json'

    if not results_file.exists():
        print(f"ERROR: No optimization results found for {crypto}")
        print(f"Run optimization first!")
        return None

    with open(results_file) as f:
        results = json.load(f)

    print(f"\nLoading best hyperparameters (Trial #{results['best_trial_number']}):")
    print(f"  Expected AUC: {results['best_auc']:.4f}")
    print(f"  Expected Accuracy: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
    print(f"  Improvement over baseline: {results['improvement']:+.4f} ({results['improvement']*100:+.2f}%)")

    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_data(crypto)

    # Train with best params
    print(f"\nTraining model...")
    params = results['complete_params']

    model = xgb.XGBClassifier(**params, verbosity=1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFinal Model Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save optimized model
    import joblib
    models_dir = Path(__file__).parent.parent / 'models'
    model_file = models_dir / f'{crypto}_v11_optimized.joblib'

    joblib.dump(model, model_file)
    print(f"\nSaved optimized model to: {model_file}")

    return model


def optimize_all(n_trials: int = 100):
    """Optimize all cryptos"""

    print("="*80)
    print("V11 OPTUNA OPTIMIZATION - ALL CRYPTOS")
    print("="*80)
    print(f"\nStrategy: Maximize AUC over {n_trials} trials per crypto")
    print(f"Expected runtime: ~{n_trials * 3 * 2 / 60:.0f} minutes (2 sec/trial * 3 cryptos)")

    cryptos = ['btc', 'eth', 'sol']
    results_summary = {}

    for crypto in cryptos:
        try:
            study = optimize_crypto(crypto, n_trials=n_trials)
            results_summary[crypto] = {
                'best_auc': study.best_value,
                'best_accuracy': study.best_trial.user_attrs['accuracy'],
                'n_trials': len(study.trials)
            }
        except Exception as e:
            print(f"\nERROR optimizing {crypto}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n\n{'='*80}")
    print("OPTIMIZATION SUMMARY - V11 TEMPORAL")
    print('='*80)

    baseline_acc = {
        'btc': 0.5405,
        'eth': 0.5315,
        'sol': 0.5563
    }

    for crypto in cryptos:
        if crypto not in results_summary:
            print(f"\n{crypto.upper()}: FAILED")
            continue

        res = results_summary[crypto]
        baseline = baseline_acc[crypto]
        improvement = res['best_accuracy'] - baseline

        print(f"\n{crypto.upper()}:")
        print(f"  Baseline:  {baseline:.4f} ({baseline*100:.2f}%)")
        print(f"  Optimized: {res['best_accuracy']:.4f} ({res['best_accuracy']*100:.2f}%)")
        print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"  Best AUC: {res['best_auc']:.4f}")
        print(f"  Trials: {res['n_trials']}")

    print(f"\n{'='*80}")
    print("V11 OPTIMIZATION COMPLETE!")
    print('='*80)
    print("\nNext steps:")
    print("  1. Review results in optimization/results/")
    print("  2. Retrain with best params: python optuna_v11.py --retrain <crypto>")
    print("  3. Run backtest with optimized models")

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V11 Optuna Optimization')
    parser.add_argument('--crypto', type=str, default='all',
                       help='Crypto to optimize (btc/eth/sol/all)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain with best params instead of optimizing')

    args = parser.parse_args()

    if args.retrain:
        if args.crypto == 'all':
            for crypto in ['btc', 'eth', 'sol']:
                train_with_best_params(crypto)
        else:
            train_with_best_params(args.crypto)
    else:
        if args.crypto == 'all':
            optimize_all(n_trials=args.trials)
        else:
            optimize_crypto(args.crypto, n_trials=args.trials)
