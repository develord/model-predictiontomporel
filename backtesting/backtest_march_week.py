"""
V11 Backtest - March Week (15-22 Mars 2026)
Detailed analysis for last week of data
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))


class MarchWeekBacktest:
    """Backtest pour la semaine du 15-22 mars 2026"""

    def __init__(self, crypto: str, initial_capital: float = 10000.0):
        self.crypto = crypto.lower()
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Load model
        model_path = Path(__file__).parent.parent / 'models' / f'{self.crypto}_v11_classifier.joblib'
        self.model = joblib.load(model_path)

        # Trading params - SEUIL 0.60
        self.tp_threshold = 0.60  # 60% confidence required
        self.fixed_tp_pct = 1.5   # +1.5% take profit
        self.fixed_sl_pct = 0.75  # -0.75% stop loss
        self.position_size_pct = 10.0  # 10% of capital per trade
        self.fee_pct = 0.1  # 0.1% fee per trade

        # Results
        self.trades = []
        self.predictions = []

    def load_week_data(self):
        """Load data from 15-22 March 2026"""
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Filter for March 15-22
        week_df = df[(df.index >= '2026-03-15') & (df.index <= '2026-03-22')].copy()

        print(f"\n{self.crypto.upper()} Week Data (15-22 Mars):")
        print(f"  Period: {week_df.index[0]} to {week_df.index[-1]}")
        print(f"  Days: {len(week_df)}")

        return week_df

    def prepare_features(self, df):
        """Extract features for prediction"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric',
            'price_target_pct', 'future_price',
            'triple_barrier_label'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def run_analysis(self):
        """Analyze each day of the week"""

        print(f"\n{'='*80}")
        print(f"V11 MARCH WEEK ANALYSIS - {self.crypto.upper()}")
        print(f"{'='*80}")
        print(f"Threshold: P(TP) > {self.tp_threshold} (60%)")
        print(f"TP: +{self.fixed_tp_pct}%, SL: -{self.fixed_sl_pct}%")

        # Load data
        week_df = self.load_week_data()
        feature_cols = self.prepare_features(week_df)

        # Analyze each day
        for i, idx in enumerate(week_df.index):
            row = week_df.loc[idx]

            # Extract features
            features = row[feature_cols].fillna(0).values.reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict P(TP)
            prob_tp = self.model.predict_proba(features)[0, 1]
            decision = 'BUY' if prob_tp >= self.tp_threshold else 'SKIP'

            prediction = {
                'date': idx,
                'close_price': float(row['close']),
                'prob_tp': float(prob_tp),
                'decision': decision,
                'confidence_pct': float(prob_tp * 100)
            }

            self.predictions.append(prediction)

        return self.predictions

    def print_results(self):
        """Print detailed daily predictions"""

        print(f"\n{'='*80}")
        print(f"DAILY PREDICTIONS - {self.crypto.upper()}")
        print(f"{'='*80}\n")

        buy_count = 0
        skip_count = 0

        for pred in self.predictions:
            decision_text = f"[{pred['decision']}]"
            color = "BUY " if pred['decision'] == 'BUY' else "SKIP"

            if pred['decision'] == 'BUY':
                buy_count += 1
            else:
                skip_count += 1

            print(f"{pred['date'].strftime('%Y-%m-%d')}: "
                  f"Price=${pred['close_price']:>10,.2f} | "
                  f"P(TP)={pred['prob_tp']:>6.2%} | "
                  f"{decision_text:6s} | "
                  f"Confidence: {pred['confidence_pct']:>5.1f}%")

        print(f"\nSUMMARY:")
        print(f"  BUY signals:  {buy_count}")
        print(f"  SKIP signals: {skip_count}")
        print(f"  Total days:   {len(self.predictions)}")


def analyze_all_cryptos():
    """Analyze all cryptos for March week"""

    print("\n" + "="*80)
    print("V11 MARCH WEEK ANALYSIS (15-22 Mars 2026)")
    print("="*80)
    print(f"Threshold: 60% (P(TP) >= 0.60)")
    print("="*80)

    cryptos = ['btc', 'eth', 'sol']
    all_predictions = {}

    for crypto in cryptos:
        try:
            engine = MarchWeekBacktest(crypto, initial_capital=10000.0)
            predictions = engine.run_analysis()
            engine.print_results()

            all_predictions[crypto] = predictions

        except Exception as e:
            print(f"\nERROR analyzing {crypto}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'march_week_predictions.json', 'w') as f:
        json.dump({
            crypto: [
                {
                    'date': pred['date'].strftime('%Y-%m-%d'),
                    'close_price': pred['close_price'],
                    'prob_tp': pred['prob_tp'],
                    'decision': pred['decision'],
                    'confidence_pct': pred['confidence_pct']
                }
                for pred in preds
            ]
            for crypto, preds in all_predictions.items()
        }, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {results_dir / 'march_week_predictions.json'}")
    print("="*80)

    return all_predictions


if __name__ == '__main__':
    analyze_all_cryptos()
