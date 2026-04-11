# AVAX Production Script — CORRIGE le 2026-04-11

## Corrections appliquees

### SHORT CNN: DeepCNNShortModel -> CNNDirectionModel
- DeepCNNShortModel (158K params) ne convergeait JAMAIS sur AVAX (trop peu de data ~1700 samples)
- CNNDirectionModel (56K params) converge en 10 epochs, val acc 61%, SHORT WR 70%+

### Params corriges dans 03_train_short_model.py et 08_train_production.py
| Param | Ancien | Nouveau |
|-------|--------|---------|
| Model | DeepCNNShortModel | CNNDirectionModel |
| seq_length | 45 | 30 |
| LR | 0.0005 | 0.0015 |
| noise_std | 0.01 | 0.015 |
| dropout | 0.35 | 0.4 |
| augment | 4x | 3x |
| class_weights | oui | non (causait non-convergence) |
| label_smoothing | 0.05 -> corrige a 0.1 | 0.1 |

### Labels SHORT (prod)
- Ancien: ATR-based (divergeait du dev)
- Nouveau: Fixed TP=2% drop, SL=1% rise (conforme au dev)

### Temperatures recalibrees
- LONG: 2.068 (ancien 2.062)
- SHORT: 1.489 (ancien 1.495)

## Backtest Q1 2026 (apres correction)
- 37 trades | WR 54.1% | Return +14.86%
- LONG: 24 trades, WR 50%
- SHORT: 13 trades, WR 62%

## Deploye le 2026-04-11
