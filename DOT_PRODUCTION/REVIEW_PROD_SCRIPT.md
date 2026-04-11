# DOT 08_train_production.py — Différences à corriger

## SHORT CNN params divergent du dev (03_train_short_model.py)

| Param | Dev (03_train_short) | Prod (08_train_production) | Action |
|-------|---------------------|---------------------------|--------|
| SHORT seq_length | **30** | 45 | ← corriger à 30 |
| SHORT LR | **0.001** | 0.0005 | ← corriger à 0.001 |
| SHORT noise | **0.02** | 0.01 | ← corriger à 0.02 |
| SHORT label_smooth | **0.05** | 0.1 | ← corriger à 0.05 |
| SHORT batch_size | **32** | 64 | ← corriger à 32 |

## Meta XGBoost manquant

- Le script prod n'a **pas de fonction `train_meta_xgboost()`**
- Sur VPS : `dot_meta_features.json` manquant → les meta models ne fonctionnent pas
- **Action** : ajouter train_meta_xgboost() comme BTC/ETH/SOL/XRP/LINK/NEAR

## Cause probable

Le script semble copié de SOL_PRODUCTION au lieu d'être adapté aux params DOT.
Les températures (L=2.109, S=2.157) sont identiques à SOL — à re-calibrer.
