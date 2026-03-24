"""
V12 Main Runner
================
Run the full V12 pipeline: Train + Backtest + Compare with V11

Usage:
    python v12/run_v12.py              # Full pipeline (train + backtest)
    python v12/run_v12.py --train      # Training only
    python v12/run_v12.py --backtest   # Backtest only (requires trained models)
"""

import sys
import argparse
from pathlib import Path

# Add project root AND v12 root to path
PROJECT_ROOT = Path(__file__).parent.parent
V12_ROOT = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V12_ROOT))


def main():
    parser = argparse.ArgumentParser(description='V12 Dynamic ATR Pipeline')
    parser.add_argument('--train', action='store_true', help='Training only')
    parser.add_argument('--backtest', action='store_true', help='Backtest only')
    args = parser.parse_args()

    # Default: run both
    run_train = args.train or (not args.train and not args.backtest)
    run_backtest = args.backtest or (not args.train and not args.backtest)

    if run_train:
        print("\n" + "=" * 80)
        print("PHASE 1: TRAINING V12 MODELS")
        print("=" * 80)
        from v12.training.train_v12 import train_all_cryptos
        train_results = train_all_cryptos()

    if run_backtest:
        print("\n" + "=" * 80)
        print("PHASE 2: BACKTESTING V12 vs V11")
        print("=" * 80)
        from v12.backtesting.backtest_v12 import backtest_all
        bt_results, bt_trades = backtest_all()


if __name__ == '__main__':
    main()
