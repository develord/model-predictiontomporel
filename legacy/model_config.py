"""
Model Configuration Module
==========================
Centralized configuration for all model hyperparameters, training settings,
and system configurations for the advanced cryptocurrency trading model.

Author: Advanced Trading System
Version: 2.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    # Model type
    model_type: str = 'explainable'  # 'basic', 'advanced', 'explainable'

    # Input dimensions
    sequence_length: int = 60  # Number of time steps to look back
    feature_dim: int = 500  # Number of features after advanced engineering

    # Transformer configuration
    d_model: int = 512  # Dimension of transformer embeddings
    n_heads: int = 8  # Number of attention heads
    n_transformer_layers: int = 6  # Number of transformer encoder layers
    transformer_feedforward_dim: int = 2048  # Feed-forward dimension

    # LSTM configuration
    lstm_hidden_dim: int = 256  # Hidden dimension for LSTM
    lstm_layers: int = 3  # Number of LSTM layers
    bidirectional: bool = True  # Use bidirectional LSTM

    # General architecture
    dropout: float = 0.2  # Dropout rate
    activation: str = 'gelu'  # Activation function
    num_classes: int = 2  # Binary classification

    # Ensemble configuration
    n_ensemble_heads: int = 3  # Number of ensemble prediction heads

    # Feature extraction
    feature_extractor_layers: List[int] = field(default_factory=lambda: [1024, 512])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config)


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer settings
    optimizer_type: str = 'adamw'  # 'adamw', 'sgd', 'radam'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'onecycle', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    lr_decay_factor: float = 0.5

    # Training optimizations
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_value: float = 1.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Regularization
    label_smoothing: float = 0.1
    use_weight_decay: bool = True

    # Validation
    val_check_interval: float = 1.0  # Check validation every epoch
    val_split_ratio: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config)


@dataclass
class DataConfig:
    """Data configuration"""

    # Data sources
    cryptos: List[str] = field(default_factory=lambda: ['btc', 'eth', 'sol'])
    timeframes: List[str] = field(default_factory=lambda: ['4h', '1d', '1w'])

    # Data paths
    data_dir: str = './data/cache'
    raw_data_dir: str = './data/raw'
    processed_data_dir: str = './data/processed'

    # Preprocessing
    normalization_method: str = 'robust'  # 'robust', 'standard', 'minmax', 'rank', 'adaptive'
    clip_outliers: bool = True
    outlier_threshold: float = 0.001  # Clip at 0.1% and 99.9% percentiles

    # Feature engineering
    use_advanced_features: bool = True
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'chi2', 'anova', 'rfe'
    max_features: Optional[int] = None  # None means use all features

    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    augmentation_types: List[str] = field(default_factory=lambda: ['noise', 'time_warp', 'magnitude_warp'])

    # Labels
    target_column: str = 'triple_barrier_label'
    lookahead_periods: int = 7
    tp_threshold: float = 1.5  # Take profit at 1.5%
    sl_threshold: float = 0.75  # Stop loss at 0.75%

    # Train/test split
    train_end_date: str = '2024-01-01'
    val_end_date: str = '2024-07-01'
    test_start_date: str = '2024-07-01'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config)


@dataclass
class SystemConfig:
    """System configuration"""

    # Hardware
    device: str = 'cuda'  # 'cuda' or 'cpu'
    num_workers: int = 4  # Number of data loading workers
    pin_memory: bool = True

    # Paths
    checkpoint_dir: str = './checkpoints'
    tensorboard_dir: str = './runs'
    log_dir: str = './logs'
    results_dir: str = './results'

    # Logging
    log_level: str = 'INFO'
    use_wandb: bool = False
    wandb_project: str = 'crypto-trading-advanced'
    wandb_entity: Optional[str] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Performance
    benchmark: bool = True  # CUDNN benchmark for performance
    enable_profiling: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config)


@dataclass
class TradingConfig:
    """Trading strategy configuration"""

    # Position sizing
    position_size: float = 0.1  # 10% of portfolio per trade
    max_positions: int = 3  # Maximum concurrent positions
    risk_per_trade: float = 0.02  # 2% risk per trade

    # Entry/Exit thresholds
    entry_confidence_threshold: float = 0.6  # Minimum confidence for entry
    exit_confidence_threshold: float = 0.4  # Exit if confidence drops below

    # Risk management
    max_drawdown: float = 0.15  # 15% maximum drawdown
    use_trailing_stop: bool = True
    trailing_stop_distance: float = 0.005  # 0.5%

    # Trading hours
    trade_24_7: bool = True  # Crypto markets are 24/7
    avoid_weekends: bool = False

    # Slippage and fees
    slippage_bps: float = 10  # 10 basis points slippage
    trading_fee_bps: float = 10  # 10 basis points trading fee

    # Portfolio management
    rebalance_frequency: str = 'daily'  # 'hourly', 'daily', 'weekly'
    use_kelly_criterion: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config)


class ConfigManager:
    """Manage all configurations"""

    def __init__(
        self,
        config_file: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file
            auto_load: Whether to auto-load from file if exists
        """
        self.config_file = Path(config_file) if config_file else Path('config.json')

        # Initialize default configurations
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        self.trading = TradingConfig()

        # Load from file if exists
        if auto_load and self.config_file.exists():
            self.load()

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for model creation"""
        return {
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_transformer_layers': self.model.n_transformer_layers,
            'lstm_hidden_dim': self.model.lstm_hidden_dim,
            'lstm_layers': self.model.lstm_layers,
            'dropout': self.model.dropout,
            'num_classes': self.model.num_classes
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.training.to_dict()

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.data.to_dict()

    def save(self, filepath: Optional[str] = None):
        """Save configuration to file"""
        filepath = Path(filepath) if filepath else self.config_file

        config = {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'data': self.data.to_dict(),
            'system': self.system.to_dict(),
            'trading': self.trading.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to {filepath}")

    def load(self, filepath: Optional[str] = None):
        """Load configuration from file"""
        filepath = Path(filepath) if filepath else self.config_file

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config = json.load(f)

        self.model = ModelConfig.from_dict(config['model'])
        self.training = TrainingConfig.from_dict(config['training'])
        self.data = DataConfig.from_dict(config['data'])
        self.system = SystemConfig.from_dict(config['system'])
        self.trading = TradingConfig.from_dict(config['trading'])

        print(f"Configuration loaded from {filepath}")

    def update_for_crypto(self, crypto: str):
        """Update configuration for specific cryptocurrency"""
        crypto_configs = {
            'btc': {
                'tp_threshold': 1.5,
                'sl_threshold': 0.75,
                'position_size': 0.15
            },
            'eth': {
                'tp_threshold': 2.0,
                'sl_threshold': 1.0,
                'position_size': 0.12
            },
            'sol': {
                'tp_threshold': 2.5,
                'sl_threshold': 1.25,
                'position_size': 0.10
            }
        }

        if crypto.lower() in crypto_configs:
            config = crypto_configs[crypto.lower()]
            self.data.tp_threshold = config['tp_threshold']
            self.data.sl_threshold = config['sl_threshold']
            self.trading.position_size = config['position_size']
            print(f"Configuration updated for {crypto.upper()}")

    def get_optimized_config(self, optimization_results: Dict[str, Any]):
        """Update configuration based on optimization results"""
        if 'learning_rate' in optimization_results:
            self.training.learning_rate = optimization_results['learning_rate']
        if 'dropout' in optimization_results:
            self.model.dropout = optimization_results['dropout']
        if 'batch_size' in optimization_results:
            self.training.batch_size = optimization_results['batch_size']

        print("Configuration updated with optimization results")

    def validate(self) -> bool:
        """Validate configuration consistency"""
        errors = []

        # Check model configuration
        if self.model.sequence_length <= 0:
            errors.append("Sequence length must be positive")
        if self.model.d_model % self.model.n_heads != 0:
            errors.append("d_model must be divisible by n_heads")

        # Check training configuration
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        # Check data configuration
        if not self.data.cryptos:
            errors.append("At least one cryptocurrency must be specified")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("Configuration validation passed")
        return True

    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)

        print("\nModel Configuration:")
        print(f"  Model Type: {self.model.model_type}")
        print(f"  Sequence Length: {self.model.sequence_length}")
        print(f"  Feature Dimension: {self.model.feature_dim}")
        print(f"  Transformer: {self.model.n_transformer_layers} layers, {self.model.n_heads} heads")
        print(f"  LSTM: {self.model.lstm_layers} layers, {self.model.lstm_hidden_dim} hidden")

        print("\nTraining Configuration:")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Optimizer: {self.training.optimizer_type}")

        print("\nData Configuration:")
        print(f"  Cryptos: {', '.join(self.data.cryptos)}")
        print(f"  Timeframes: {', '.join(self.data.timeframes)}")
        print(f"  Normalization: {self.data.normalization_method}")
        print(f"  TP/SL: {self.data.tp_threshold}% / {self.data.sl_threshold}%")

        print("\nTrading Configuration:")
        print(f"  Position Size: {self.trading.position_size*100}%")
        print(f"  Max Positions: {self.trading.max_positions}")
        print(f"  Entry Threshold: {self.trading.entry_confidence_threshold}")

        print("="*50)


# Preset configurations for different scenarios
PRESETS = {
    'quick_test': {
        'training': {
            'batch_size': 64,
            'num_epochs': 10,
            'learning_rate': 1e-3
        },
        'model': {
            'n_transformer_layers': 2,
            'lstm_layers': 1
        }
    },
    'production': {
        'training': {
            'batch_size': 32,
            'num_epochs': 200,
            'learning_rate': 5e-5,
            'use_mixed_precision': True
        },
        'model': {
            'n_transformer_layers': 6,
            'lstm_layers': 3,
            'dropout': 0.3
        },
        'system': {
            'use_wandb': True
        }
    },
    'optimization': {
        'training': {
            'batch_size': 16,
            'num_epochs': 50,
            'learning_rate': 1e-4
        },
        'model': {
            'n_transformer_layers': 4,
            'lstm_layers': 2
        }
    }
}


def get_preset_config(preset_name: str) -> ConfigManager:
    """Get configuration manager with preset"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    config_manager = ConfigManager(auto_load=False)

    preset = PRESETS[preset_name]
    for section, params in preset.items():
        section_config = getattr(config_manager, section)
        for key, value in params.items():
            setattr(section_config, key, value)

    print(f"Loaded preset configuration: {preset_name}")
    return config_manager


if __name__ == "__main__":
    # Test configuration module
    print("Testing Model Configuration Module...")

    # Create configuration manager
    config = ConfigManager(auto_load=False)

    # Print default configuration
    config.print_summary()

    # Validate configuration
    config.validate()

    # Test saving and loading
    config.save('test_config.json')

    # Load configuration
    config2 = ConfigManager('test_config.json')

    # Test preset
    prod_config = get_preset_config('production')
    prod_config.print_summary()

    # Update for specific crypto
    config.update_for_crypto('eth')

    print("\nConfiguration module test complete!")