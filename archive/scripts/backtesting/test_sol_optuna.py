"""
Quick test: SOL Optuna vs Phase 1
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from phase1_backtest import simulate_trading

print("=" * 70)
print("SOL: PHASE 1 vs OPTUNA COMPARISON")
print("=" * 70)
print()

# Test Phase 1 (feature-selected)
print("Testing Phase 1 (feature-selected + T=0.35)...")
phase1_results, _ = simulate_trading('sol', use_phase1=True, use_optuna=False)
print(f"  ROI: {phase1_results['roi_pct']:+.2f}%")
print(f"  Win Rate: {phase1_results['win_rate']:.1f}%")
print(f"  Trades: {phase1_results['total_trades']}")
print(f"  Capital: ${phase1_results['initial_capital']:.2f} -> ${phase1_results['final_capital']:.2f}")
print()

# Test Optuna
print("Testing Optuna (optimized hyperparams + T=0.35)...")
optuna_results, _ = simulate_trading('sol', use_phase1=False, use_optuna=True)
print(f"  ROI: {optuna_results['roi_pct']:+.2f}%")
print(f"  Win Rate: {optuna_results['win_rate']:.1f}%")
print(f"  Trades: {optuna_results['total_trades']}")
print(f"  Capital: ${optuna_results['initial_capital']:.2f} -> ${optuna_results['final_capital']:.2f}")
print()

# Comparison
print("=" * 70)
print("COMPARISON")
print("=" * 70)
roi_diff = optuna_results['roi_pct'] - phase1_results['roi_pct']
wr_diff = optuna_results['win_rate'] - phase1_results['win_rate']
trades_diff = optuna_results['total_trades'] - phase1_results['total_trades']

print(f"ROI: {roi_diff:+.2f}% ({phase1_results['roi_pct']:.1f}% -> {optuna_results['roi_pct']:.1f}%)")
print(f"Win Rate: {wr_diff:+.1f}% ({phase1_results['win_rate']:.1f}% -> {optuna_results['win_rate']:.1f}%)")
print(f"Trades: {trades_diff:+d} ({phase1_results['total_trades']} -> {optuna_results['total_trades']})")
print()

if roi_diff > 0:
    print(f"OPTUNA IS BETTER! +{roi_diff:.2f}% ROI improvement")
else:
    print(f"PHASE 1 IS BETTER! Optuna is {-roi_diff:.2f}% worse")
print()
