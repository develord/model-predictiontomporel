"""
Feature Selection - V11 TEMPORAL
==================================
Reduce features from 237-348 to top 50-100 most important
Expected improvement: +3-5% accuracy

Strategy:
1. Train baseline model on all features
2. Extract feature importances
3. Select top N features
4. Retrain with selected features
5. Compare performance
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_feature_importance(crypto: str, top_n: int = 50):
    """
    Analyze feature importance and select top N features

    Args:
        crypto: 'btc', 'eth', or 'sol'
        top_n: Number of top features to keep (default 50)

    Returns:
        selected_features: List of top N feature names
        importance_stats: Dictionary with importance statistics
    """

    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION FOR {crypto.upper()}")
    print(f"Target: Reduce to top {top_n} features")
    print('='*80)

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric', 'price_target_pct',
        'future_price', 'triple_barrier_label'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n[1/4] Loading Data")
    print(f"  Total features: {len(feature_cols)}")

    # Clean data
    df_clean = df[df['triple_barrier_label'].notna()].copy()

    # Temporal split
    timestamps = df_clean.index
    train_mask = timestamps < '2025-01-01'
    test_mask = timestamps >= '2025-01-01'

    X = df_clean[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = (df_clean['triple_barrier_label'] == 1).astype(int).values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Load existing baseline model or train new one
    model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'

    if model_file.exists():
        print(f"\n[2/4] Loading Baseline Model")
        model = joblib.load(model_file)
    else:
        print(f"\n[2/4] Training Baseline Model")
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            tree_method='hist'
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Get baseline performance
    y_pred_baseline = model.predict(X_test)
    y_pred_proba_baseline = model.predict_proba(X_test)[:, 1]

    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_auc = roc_auc_score(y_test, y_pred_proba_baseline)

    print(f"  Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # Extract feature importances
    print(f"\n[3/4] Analyzing Feature Importances")
    feature_importances = model.feature_importances_

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    # Statistics
    print(f"  Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:40s}: {row['importance']:.6f}")

    print(f"\n  Importance distribution:")
    print(f"    Mean: {feature_importances.mean():.6f}")
    print(f"    Median: {np.median(feature_importances):.6f}")
    print(f"    Top 50 cumulative: {importance_df.head(50)['importance'].sum():.4f}")
    print(f"    Top 100 cumulative: {importance_df.head(100)['importance'].sum():.4f}")

    # Select top N features
    selected_features = importance_df.head(top_n)['feature'].tolist()

    print(f"\n[4/4] Retraining with Top {top_n} Features")

    # Retrain with selected features
    feature_indices = [feature_cols.index(f) for f in selected_features]
    X_train_selected = X_train[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model_selected = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method='hist'
    )

    model_selected.fit(
        X_train_selected, y_train,
        eval_set=[(X_test_selected, y_test)],
        verbose=False
    )

    # Evaluate selected model
    y_pred_selected = model_selected.predict(X_test_selected)
    y_pred_proba_selected = model_selected.predict_proba(X_test_selected)[:, 1]

    selected_acc = accuracy_score(y_test, y_pred_selected)
    selected_auc = roc_auc_score(y_test, y_pred_proba_selected)

    print(f"\n  Selected Features Performance:")
    print(f"    Accuracy: {selected_acc:.4f} ({selected_acc*100:.2f}%)")
    print(f"    AUC: {selected_auc:.4f}")

    print(f"\n  Improvement:")
    acc_improvement = selected_acc - baseline_acc
    auc_improvement = selected_auc - baseline_auc
    print(f"    Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
    print(f"    AUC: {auc_improvement:+.4f}")

    # Save results
    results = {
        'crypto': crypto,
        'original_features': len(feature_cols),
        'selected_features': top_n,
        'baseline_accuracy': float(baseline_acc),
        'baseline_auc': float(baseline_auc),
        'selected_accuracy': float(selected_acc),
        'selected_auc': float(selected_auc),
        'accuracy_improvement': float(acc_improvement),
        'auc_improvement': float(auc_improvement),
        'selected_feature_names': selected_features,
        'feature_importances': {
            feat: float(imp)
            for feat, imp in zip(selected_features,
                                importance_df.head(top_n)['importance'].values)
        }
    }

    # Save selected features list
    results_dir = Path(__file__).parent.parent / 'optimization' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f'{crypto}_selected_features_top{top_n}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved results to: {output_file}")

    # Save selected model
    if acc_improvement > 0:  # Only save if improvement
        models_dir = Path(__file__).parent.parent / 'models'
        model_file = models_dir / f'{crypto}_v11_feature_selected_top{top_n}.joblib'
        joblib.dump(model_selected, model_file)
        print(f"  Saved model to: {model_file}")

    return selected_features, results


def select_features_all_cryptos(top_n: int = 50):
    """Run feature selection for all cryptos"""

    print("="*80)
    print(f"FEATURE SELECTION - TOP {top_n} FEATURES")
    print("="*80)
    print(f"\nGoal: Reduce features to top {top_n} most important")
    print(f"Expected: +3-5% accuracy improvement\n")

    all_results = {}

    for crypto in ['btc', 'eth', 'sol']:
        try:
            selected_features, results = analyze_feature_importance(crypto, top_n=top_n)
            all_results[crypto] = results
        except Exception as e:
            print(f"\nERROR processing {crypto}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n\n{'='*80}")
    print("FEATURE SELECTION SUMMARY")
    print('='*80)

    for crypto in ['btc', 'eth', 'sol']:
        if crypto not in all_results:
            print(f"\n{crypto.upper()}: FAILED")
            continue

        res = all_results[crypto]
        print(f"\n{crypto.upper()}:")
        print(f"  Original features: {res['original_features']}")
        print(f"  Selected features: {res['selected_features']}")
        print(f"  Reduction: {(1 - res['selected_features']/res['original_features'])*100:.1f}%")
        print(f"  Baseline accuracy: {res['baseline_accuracy']:.4f}")
        print(f"  Selected accuracy: {res['selected_accuracy']:.4f}")
        print(f"  Improvement: {res['accuracy_improvement']:+.4f} ({res['accuracy_improvement']*100:+.2f}%)")

        if res['accuracy_improvement'] > 0:
            print(f"  Status: IMPROVED!")
        elif res['accuracy_improvement'] > -0.01:
            print(f"  Status: Maintained (acceptable)")
        else:
            print(f"  Status: Degraded (keep original)")

    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION COMPLETE!")
    print('='*80)

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Selection for V11 TEMPORAL')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top features to keep')
    parser.add_argument('--crypto', type=str, default='all', help='Crypto to process (btc/eth/sol/all)')

    args = parser.parse_args()

    if args.crypto == 'all':
        select_features_all_cryptos(top_n=args.top_n)
    else:
        analyze_feature_importance(args.crypto, top_n=args.top_n)
