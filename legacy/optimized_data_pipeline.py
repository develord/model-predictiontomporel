"""
Optimized Data Pipeline for Advanced Cryptocurrency Trading Model
==================================================================
High-performance data loading, preprocessing, and augmentation pipeline with:
1. Efficient multi-timeframe data loading
2. Advanced normalization techniques
3. Data augmentation for improved generalization
4. Memory-efficient batch generation
5. Real-time data streaming capabilities
6. Cross-validation splits for robust training

Author: Advanced Trading System
Version: 2.0
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import warnings
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Advanced normalization strategies for financial time series
    """

    def __init__(self, method: str = 'robust', clip_outliers: bool = True):
        """
        Initialize normalizer

        Args:
            method: Normalization method ('robust', 'standard', 'minmax', 'rank', 'adaptive')
            clip_outliers: Whether to clip outliers before normalization
        """
        self.method = method
        self.clip_outliers = clip_outliers
        self.scalers = {}
        self.fitted = False

    def fit(self, data: pd.DataFrame, feature_groups: Optional[Dict[str, List[str]]] = None):
        """
        Fit normalizer to training data

        Args:
            data: Training dataframe
            feature_groups: Dictionary grouping features by type for separate normalization
        """
        if feature_groups is None:
            # Default grouping
            feature_groups = self._auto_group_features(data)

        for group_name, features in feature_groups.items():
            group_data = data[features]

            if self.clip_outliers:
                # Clip outliers at 99.9th percentile
                lower = group_data.quantile(0.001)
                upper = group_data.quantile(0.999)
                group_data = group_data.clip(lower=lower, upper=upper, axis=1)

            # Select scaler based on method and feature type
            scaler = self._get_scaler(group_name)
            scaler.fit(group_data)
            self.scalers[group_name] = {
                'scaler': scaler,
                'features': features
            }

        self.fitted = True
        logger.info(f"Normalizer fitted with {len(self.scalers)} feature groups")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers

        Args:
            data: Data to transform

        Returns:
            Normalized dataframe
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        result = data.copy()

        for group_name, group_info in self.scalers.items():
            features = group_info['features']
            scaler = group_info['scaler']

            # Only transform features that exist in the data
            existing_features = [f for f in features if f in data.columns]
            if existing_features:
                if self.clip_outliers and hasattr(scaler, 'data_min_'):
                    # Clip based on training data range
                    group_data = data[existing_features]
                    lower = pd.Series(scaler.data_min_, index=existing_features)
                    upper = pd.Series(scaler.data_max_, index=existing_features)
                    group_data = group_data.clip(lower=lower, upper=upper, axis=1)
                    result[existing_features] = scaler.transform(group_data)
                else:
                    result[existing_features] = scaler.transform(data[existing_features])

        return result

    def _get_scaler(self, group_name: str):
        """Get appropriate scaler for feature group"""
        if self.method == 'adaptive':
            # Use different scalers for different feature types
            if 'price' in group_name or 'volume' in group_name:
                return RobustScaler()
            elif 'indicator' in group_name:
                return StandardScaler()
            else:
                return MinMaxScaler(feature_range=(-1, 1))
        elif self.method == 'robust':
            return RobustScaler()
        elif self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        elif self.method == 'rank':
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _auto_group_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically group features by type"""
        groups = {
            'price': [],
            'volume': [],
            'indicators': [],
            'patterns': [],
            'statistical': []
        }

        for col in data.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['price', 'high', 'low', 'close', 'open']):
                groups['price'].append(col)
            elif 'volume' in col_lower or 'vol' in col_lower:
                groups['volume'].append(col)
            elif any(x in col_lower for x in ['pattern', 'candlestick', 'doji', 'hammer']):
                groups['patterns'].append(col)
            elif any(x in col_lower for x in ['skew', 'kurt', 'var', 'std', 'mean']):
                groups['statistical'].append(col)
            else:
                groups['indicators'].append(col)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def save(self, filepath: str):
        """Save fitted normalizer"""
        joblib.dump(self, filepath)
        logger.info(f"Normalizer saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load fitted normalizer"""
        normalizer = joblib.load(filepath)
        logger.info(f"Normalizer loaded from {filepath}")
        return normalizer


class DataAugmenter:
    """
    Data augmentation techniques for time series
    """

    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise * np.std(data, axis=0)

    @staticmethod
    def time_warp(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping augmentation"""
        from scipy.interpolate import interp1d

        batch_size, seq_len, n_features = data.shape
        warped = np.zeros_like(data)

        for i in range(batch_size):
            # Generate random warping
            warp_steps = np.random.normal(1.0, sigma, seq_len).cumsum()
            warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min())
            warp_steps = warp_steps * (seq_len - 1)

            # Apply warping
            for j in range(n_features):
                interpolator = interp1d(
                    np.arange(seq_len),
                    data[i, :, j],
                    kind='linear',
                    fill_value='extrapolate'
                )
                warped[i, :, j] = interpolator(warp_steps)

        return warped

    @staticmethod
    def magnitude_warp(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply magnitude warping"""
        batch_size, seq_len, n_features = data.shape
        warping = np.random.normal(1.0, sigma, (batch_size, 1, n_features))
        return data * warping

    @staticmethod
    def window_slice(data: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Random window slicing"""
        batch_size, seq_len, n_features = data.shape
        target_len = int(seq_len * reduce_ratio)

        sliced = np.zeros((batch_size, target_len, n_features))
        for i in range(batch_size):
            start = np.random.randint(0, seq_len - target_len + 1)
            sliced[i] = data[i, start:start+target_len]

        return sliced

    @staticmethod
    def mixup(data: np.ndarray, labels: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup augmentation"""
        batch_size = data.shape[0]
        lam = np.random.beta(alpha, alpha, batch_size)

        index = np.random.permutation(batch_size)
        mixed_data = np.zeros_like(data)
        mixed_labels = np.zeros_like(labels)

        for i in range(batch_size):
            mixed_data[i] = lam[i] * data[i] + (1 - lam[i]) * data[index[i]]
            mixed_labels[i] = lam[i] * labels[i] + (1 - lam[i]) * labels[index[i]]

        return mixed_data, mixed_labels


class CryptoTradingDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency trading data
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'target',
        augment: bool = False,
        augmentation_prob: float = 0.3
    ):
        """
        Initialize dataset

        Args:
            data: Dataframe with features and target
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            feature_cols: List of feature columns (if None, uses all except target)
            target_col: Name of target column
            augment: Whether to apply augmentation
            augmentation_prob: Probability of applying augmentation
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.augmenter = DataAugmenter()

        # Identify features and target
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns if col != target_col]
        else:
            self.feature_cols = feature_cols

        self.target_col = target_col

        # Convert to numpy for faster indexing
        self.features = data[self.feature_cols].values
        self.targets = data[self.target_col].values if target_col in data.columns else None

        # Calculate valid indices
        self.valid_indices = list(range(len(data) - sequence_length - prediction_horizon + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # Get sequence
        X = self.features[start_idx:end_idx].copy()

        # Get target (if available)
        if self.targets is not None:
            y = self.targets[end_idx + self.prediction_horizon - 1]
        else:
            y = 0

        # Apply augmentation
        if self.augment and np.random.random() < self.augmentation_prob:
            aug_type = np.random.choice(['noise', 'magnitude', 'time_warp'])

            if aug_type == 'noise':
                X = self.augmenter.add_noise(X.reshape(1, *X.shape), noise_level=0.01)[0]
            elif aug_type == 'magnitude':
                X = self.augmenter.magnitude_warp(X.reshape(1, *X.shape), sigma=0.1)[0]
            elif aug_type == 'time_warp':
                X = self.augmenter.time_warp(X.reshape(1, *X.shape), sigma=0.1)[0]

        return torch.FloatTensor(X), torch.LongTensor([y])


class DataPipeline:
    """
    Complete data pipeline for training and inference
    """

    def __init__(
        self,
        data_dir: str = './data/cache',
        sequence_length: int = 60,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize_method: str = 'robust',
        augment_train: bool = True
    ):
        """
        Initialize data pipeline

        Args:
            data_dir: Directory containing cached data
            sequence_length: Length of input sequences
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            normalize_method: Normalization method
            augment_train: Whether to augment training data
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_method = normalize_method
        self.augment_train = augment_train

        self.normalizer = DataNormalizer(method=normalize_method)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(
        self,
        crypto: str,
        train_end_date: str = '2024-01-01',
        val_end_date: str = '2024-07-01',
        feature_engineering_fn: Optional[callable] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training

        Args:
            crypto: Cryptocurrency symbol ('btc', 'eth', 'sol')
            train_end_date: End date for training data
            val_end_date: End date for validation data
            feature_engineering_fn: Optional feature engineering function

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load merged data
        data_path = self.data_dir / f'{crypto}_multi_tf_merged.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {crypto.upper()} data: {len(df)} rows")

        # Apply feature engineering if provided
        if feature_engineering_fn:
            df = feature_engineering_fn(df)
            logger.info(f"Applied feature engineering: {len(df.columns)} features")

        # Split data
        train_data = df[df.index < train_end_date]
        val_data = df[(df.index >= train_end_date) & (df.index < val_end_date)]
        test_data = df[df.index >= val_end_date]

        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Fit normalizer on training data
        feature_cols = [col for col in train_data.columns if col not in ['triple_barrier_label', 'target']]
        self.normalizer.fit(train_data[feature_cols])

        # Transform all datasets
        train_data[feature_cols] = self.normalizer.transform(train_data[feature_cols])
        val_data[feature_cols] = self.normalizer.transform(val_data[feature_cols])
        test_data[feature_cols] = self.normalizer.transform(test_data[feature_cols])

        # Create datasets
        train_dataset = CryptoTradingDataset(
            train_data,
            sequence_length=self.sequence_length,
            feature_cols=feature_cols,
            target_col='triple_barrier_label' if 'triple_barrier_label' in train_data.columns else 'target',
            augment=self.augment_train
        )

        val_dataset = CryptoTradingDataset(
            val_data,
            sequence_length=self.sequence_length,
            feature_cols=feature_cols,
            target_col='triple_barrier_label' if 'triple_barrier_label' in val_data.columns else 'target',
            augment=False
        )

        test_dataset = CryptoTradingDataset(
            test_data,
            sequence_length=self.sequence_length,
            feature_cols=feature_cols,
            target_col='triple_barrier_label' if 'triple_barrier_label' in test_data.columns else 'target',
            augment=False
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        logger.info("Data loaders created successfully")
        return self.train_loader, self.val_loader, self.test_loader

    def get_feature_importance_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        Get sample data for feature importance analysis

        Args:
            n_samples: Number of samples to return

        Returns:
            Tuple of (data array, feature names)
        """
        if self.train_loader is None:
            raise ValueError("Data must be prepared first")

        samples = []
        for batch_X, _ in self.train_loader:
            samples.append(batch_X.numpy())
            if len(samples) * self.batch_size >= n_samples:
                break

        data = np.concatenate(samples, axis=0)[:n_samples]
        feature_names = self.train_loader.dataset.feature_cols

        return data, feature_names


class RealTimeDataStreamer:
    """
    Real-time data streaming for live trading
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        exchange: str = 'binance',
        update_interval: int = 60
    ):
        """
        Initialize real-time streamer

        Args:
            api_key: API key for exchange
            exchange: Exchange to stream from
            update_interval: Update interval in seconds
        """
        self.api_key = api_key
        self.exchange = exchange
        self.update_interval = update_interval
        self.streaming = False

    async def stream_data(self, symbols: List[str], callback: callable):
        """
        Stream real-time data

        Args:
            symbols: List of symbols to stream
            callback: Callback function for new data
        """
        self.streaming = True

        while self.streaming:
            try:
                # Fetch latest data
                data = await self._fetch_latest_data(symbols)

                # Process and send to callback
                await callback(data)

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(5)

    async def _fetch_latest_data(self, symbols: List[str]) -> Dict:
        """Fetch latest market data"""
        # Implementation would connect to exchange API
        # This is a placeholder
        async with aiohttp.ClientSession() as session:
            data = {}
            for symbol in symbols:
                # Fetch from exchange API
                url = f"https://api.{self.exchange}.com/v3/ticker/24hr?symbol={symbol}"
                async with session.get(url) as response:
                    data[symbol] = await response.json()

            return data

    def stop_streaming(self):
        """Stop data streaming"""
        self.streaming = False
        logger.info("Streaming stopped")


def create_walk_forward_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    train_size: int = 1000,
    test_size: int = 200
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Create walk-forward validation splits

    Args:
        data: DataFrame with datetime index
        n_splits: Number of splits
        train_size: Training window size
        test_size: Test window size

    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    total_size = train_size + test_size
    step = test_size

    for i in range(n_splits):
        start_idx = i * step
        train_end_idx = start_idx + train_size
        test_end_idx = train_end_idx + test_size

        if test_end_idx > len(data):
            break

        train_indices = data.index[start_idx:train_end_idx]
        test_indices = data.index[train_end_idx:test_end_idx]

        splits.append((train_indices, test_indices))

    logger.info(f"Created {len(splits)} walk-forward splits")
    return splits


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Optimized Data Pipeline...")

    # Initialize pipeline
    pipeline = DataPipeline(
        sequence_length=60,
        batch_size=32,
        normalize_method='robust',
        augment_train=True
    )

    # Test normalizer
    print("\nTesting DataNormalizer...")
    sample_data = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    normalizer = DataNormalizer(method='robust')
    normalizer.fit(sample_data)
    normalized = normalizer.transform(sample_data)
    print(f"Normalization complete. Mean: {normalized.mean().mean():.4f}, Std: {normalized.std().mean():.4f}")

    # Test augmentation
    print("\nTesting DataAugmenter...")
    augmenter = DataAugmenter()
    sample_array = np.random.randn(32, 60, 10)
    augmented = augmenter.add_noise(sample_array, noise_level=0.01)
    print(f"Augmentation complete. Shape: {augmented.shape}")

    print("\nPipeline test complete!")