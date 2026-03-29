"""
Advanced Cryptocurrency Trading Model - Transformer-LSTM Hybrid Architecture
=============================================================================
This is a state-of-the-art deep learning model combining the best of:
1. Transformer attention mechanisms for feature importance
2. Bi-directional LSTM for temporal sequence modeling
3. Multi-head attention for cross-timeframe analysis
4. Ensemble predictions with uncertainty estimation

Author: Advanced AI Trading System
Version: 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer layers to capture temporal relationships
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiTimeframeAttention(nn.Module):
    """
    Custom attention mechanism for multi-timeframe feature fusion
    Learns to weight importance of different timeframes dynamically
    """
    def __init__(self, input_dim: int, num_timeframes: int = 3):
        super().__init__()
        self.num_timeframes = num_timeframes
        self.attention_weights = nn.Parameter(torch.ones(num_timeframes))

        # Cross-timeframe attention
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = 1.0 / math.sqrt(input_dim)

    def forward(self, features_4h: torch.Tensor, features_1d: torch.Tensor,
                features_1w: torch.Tensor) -> torch.Tensor:
        # Stack timeframes
        stacked = torch.stack([features_4h, features_1d, features_1w], dim=1)

        # Compute attention
        Q = self.query(stacked)
        K = self.key(stacked)
        V = self.value(stacked)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Weighted combination
        weights = F.softmax(self.attention_weights, dim=0)
        output = torch.sum(attended * weights.view(1, -1, 1, 1), dim=1)

        return output


class FeatureExtractor(nn.Module):
    """
    Advanced feature extraction layer with residual connections
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Better than ReLU for financial data

        # Residual connection if dimensions match
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fc1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2) if len(x.shape) == 3 else self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2) if len(out.shape) == 3 else self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2) if len(out.shape) == 3 else self.bn3(out)

        if self.residual:
            identity = self.residual(identity)
        elif x.shape[-1] != out.shape[-1]:
            return self.activation(out)

        out = out + identity
        return self.activation(out)


class TransformerLSTMHybrid(nn.Module):
    """
    Main model architecture combining Transformer and LSTM

    Architecture Flow:
    1. Feature extraction and embedding
    2. Multi-timeframe attention fusion
    3. Transformer encoder for global patterns
    4. Bi-directional LSTM for sequential dependencies
    5. Ensemble head with uncertainty estimation
    """
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 60,
        d_model: int = 512,
        n_heads: int = 8,
        n_transformer_layers: int = 6,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Feature extraction layers
        self.feature_extractor = FeatureExtractor(input_dim, 1024, d_model, dropout)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Multi-timeframe attention
        self.mtf_attention = MultiTimeframeAttention(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Attention pooling for LSTM outputs
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim * 2,  # Bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Ensemble prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            ) for _ in range(3)  # 3 ensemble heads
        ])

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - logits: Raw predictions
                - probabilities: Softmax probabilities
                - uncertainty: Uncertainty estimates
                - attention_weights: (optional) Attention weights
        """
        batch_size = x.shape[0]

        # Feature extraction
        features = self.feature_extractor(x)

        # Add positional encoding
        features = self.positional_encoding(features)

        # Transformer encoding
        transformer_out = self.transformer(features)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(transformer_out)

        # Attention pooling
        query = lstm_out.mean(dim=1, keepdim=True)  # Global query
        attended_out, attention_weights = self.attention_pooling(query, lstm_out, lstm_out)
        attended_out = attended_out.squeeze(1)

        # Ensemble predictions
        predictions = []
        for head in self.prediction_heads:
            pred = head(attended_out)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_logits = torch.sum(predictions * weights.view(1, -1, 1), dim=1)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(attended_out)

        # Compute probabilities
        probabilities = F.softmax(ensemble_logits, dim=-1)

        output = {
            'logits': ensemble_logits,
            'probabilities': probabilities,
            'uncertainty': uncertainty,
            'individual_predictions': predictions
        }

        if return_attention:
            output['attention_weights'] = attention_weights

        return output


class AdvancedTradingModel(nn.Module):
    """
    Complete trading model with pre/post processing and risk management
    """
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int = 60,
        model_config: Optional[Dict] = None
    ):
        super().__init__()

        # Default configuration
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_transformer_layers': 6,
            'lstm_hidden_dim': 256,
            'lstm_layers': 3,
            'dropout': 0.2,
            'num_classes': 2
        }

        if model_config:
            config.update(model_config)

        # Main model
        self.model = TransformerLSTMHybrid(
            input_dim=feature_dim,
            sequence_length=sequence_length,
            **config
        )

        # Risk adjustment layer
        self.risk_adjustment = nn.Sequential(
            nn.Linear(2 + 1, 16),  # probabilities + uncertainty
            nn.GELU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, adjust_for_risk: bool = True) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with risk adjustment

        Args:
            x: Input features
            adjust_for_risk: Whether to apply risk adjustment

        Returns:
            Complete prediction dictionary
        """
        # Get base predictions
        output = self.model(x)

        if adjust_for_risk:
            # Combine probabilities with uncertainty for risk adjustment
            risk_input = torch.cat([
                output['probabilities'],
                output['uncertainty']
            ], dim=-1)

            # Apply risk adjustment
            adjusted_probs = self.risk_adjustment(risk_input)
            output['risk_adjusted_probabilities'] = adjusted_probs

            # Trading decision based on risk-adjusted probabilities
            output['trading_decision'] = torch.argmax(adjusted_probs, dim=-1)
        else:
            output['trading_decision'] = torch.argmax(output['probabilities'], dim=-1)

        # Calculate confidence score
        max_prob = torch.max(output['probabilities'], dim=-1)[0]
        output['confidence'] = max_prob * (1 - output['uncertainty'].squeeze())

        return output


class ModelWithExplainability(AdvancedTradingModel):
    """
    Extended model with explainability features using attention visualization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Feature importance layer
        self.feature_importance = nn.Sequential(
            nn.Linear(self.model.input_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.model.input_dim),
            nn.Softmax(dim=-1)
        )

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature importance scores
        """
        # Global average of input
        global_features = x.mean(dim=1)

        # Calculate importance
        importance = self.feature_importance(global_features)

        return importance

    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Provide explanation for predictions
        """
        # Get predictions with attention
        output = self.model(x, return_attention=True)

        # Get feature importance
        feature_importance = self.get_feature_importance(x)

        # Add explanations to output
        output['feature_importance'] = feature_importance
        output['top_features'] = torch.topk(feature_importance, k=10, dim=-1)

        return output


def create_model(
    feature_dim: int,
    sequence_length: int = 60,
    model_type: str = 'advanced',
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Factory function to create model instances

    Args:
        feature_dim: Number of input features
        sequence_length: Length of input sequences
        model_type: Type of model ('basic', 'advanced', 'explainable')
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    if model_type == 'basic':
        return TransformerLSTMHybrid(
            input_dim=feature_dim,
            sequence_length=sequence_length,
            **(config or {})
        )
    elif model_type == 'advanced':
        return AdvancedTradingModel(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            model_config=config
        )
    elif model_type == 'explainable':
        return ModelWithExplainability(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            model_config=config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model size calculator
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing Advanced Trading Model...")

    # Example configuration
    batch_size = 32
    sequence_length = 60
    feature_dim = 350  # Approximate number of features in current system

    # Create model
    model = create_model(
        feature_dim=feature_dim,
        sequence_length=sequence_length,
        model_type='explainable'
    )

    # Test forward pass
    x = torch.randn(batch_size, sequence_length, feature_dim)
    output = model.explain_prediction(x)

    print(f"\nModel Architecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"\nOutput shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\nModel ready for training!")