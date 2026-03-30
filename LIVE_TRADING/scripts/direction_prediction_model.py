"""
Direction Prediction Model - Simple Binary Classification
==========================================================
Modèle optimisé pour prédire la direction de la bougie de demain (1d) : SHORT ou LONG

Author: Advanced Trading System
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class DirectionPredictionModel(nn.Module):
    """
    Modèle simplifié pour prédire la direction (SHORT/LONG) de la bougie 1d de demain.

    Architecture:
    - Feature Extractor avec normalization
    - Transformer Encoder pour patterns temporels
    - LSTM bidirectionnel pour séquences
    - Attention mechanism pour focus sur signaux importants
    - Classification binaire: 0=SHORT, 1=LONG
    """

    def __init__(
        self,
        feature_dim: int = 500,
        sequence_length: int = 60,
        d_model: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.use_attention = use_attention

        # 1. Feature projection et normalization
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Positional Encoding
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

        # 4. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional

        # 5. Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )

        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)  # 2 classes: SHORT (0), LONG (1)
        )

        # Initialize weights
        self._init_weights()

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim)
            return_attention: Whether to return attention weights

        Returns:
            logits: Tensor of shape (batch_size, 2) - probabilities for SHORT/LONG
            attention_weights: Optional attention weights if return_attention=True
        """
        batch_size = x.size(0)

        # 1. Project features
        x = self.feature_projection(x)  # (B, T, d_model)

        # 2. Add positional encoding
        pe = self.positional_encoding.to(x.device)
        x = x + pe

        # 3. Transformer encoding
        x = self.transformer(x)  # (B, T, d_model)

        # 4. LSTM processing
        x, _ = self.lstm(x)  # (B, T, lstm_hidden_dim*2)

        # 5. Attention (optional)
        attention_weights = None
        if self.use_attention:
            x, attention_weights = self.attention(x, x, x)  # (B, T, lstm_hidden_dim*2)

        # 6. Global pooling (take last timestep for direction prediction)
        x = x[:, -1, :]  # (B, lstm_hidden_dim*2)

        # 7. Classification
        logits = self.classifier(x)  # (B, 2)

        if return_attention:
            return logits, attention_weights
        return logits, None

    def predict_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict direction with confidence

        Args:
            x: Input tensor

        Returns:
            direction: 0=SHORT, 1=LONG
            confidence: Probability of predicted direction
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)

        return direction, confidence


class LightweightDirectionModel(nn.Module):
    """
    Modèle léger et rapide pour prédiction de direction
    Idéal pour entraînement rapide et inférence en temps réel
    """

    def __init__(
        self,
        feature_dim: int = 500,
        sequence_length: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU (plus rapide que LSTM)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.feature_extractor(x)  # (B, T, hidden_dim)

        # GRU processing
        x, _ = self.gru(x)  # (B, T, hidden_dim*2)

        # Take last timestep
        x = x[:, -1, :]  # (B, hidden_dim*2)

        # Classify
        logits = self.classifier(x)  # (B, 2)

        return logits

    def predict_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict direction with confidence"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)

        return direction, confidence


class CNNDirectionModel(nn.Module):
    """
    1D-CNN with temporal attention for direction prediction.
    Much fewer parameters than GRU/LSTM, resistant to overfitting.
    Detects local patterns in time series using convolutions.
    """

    def __init__(
        self,
        feature_dim: int = 99,
        sequence_length: int = 30,
        dropout: float = 0.3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(sequence_length),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-scale 1D convolutions (detect patterns at different scales)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(64, 32, kernel_size=7, padding=3)

        self.bn_conv = nn.BatchNorm1d(96)  # 32*3 channels
        self.conv_drop = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, feature_dim) -> (batch, 2)"""
        # Project features
        x = self.input_proj(x)  # (B, T, 64)

        # Transpose for conv1d: (B, C, T)
        x = x.permute(0, 2, 1)

        # Multi-scale convolutions
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        x = torch.cat([c3, c5, c7], dim=1)  # (B, 96, T)

        x = self.bn_conv(x)
        x = self.conv_drop(x)

        # Second conv
        x = self.conv2(x)  # (B, 48, T)

        # Transpose back: (B, T, 48)
        x = x.permute(0, 2, 1)

        # Temporal attention pooling
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B, 48)

        # Classify
        return self.classifier(x)

    def predict_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)
        return direction, confidence


class DeepCNNShortModel(nn.Module):
    """Deeper CNN for SHORT detection with wider kernels."""
    def __init__(self, feature_dim, sequence_length=45, dropout=0.35):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, 96), nn.BatchNorm1d(sequence_length), nn.GELU(), nn.Dropout(dropout))
        self.conv3_1 = nn.Conv1d(96, 48, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv1d(96, 48, kernel_size=5, padding=2)
        self.conv9_1 = nn.Conv1d(96, 48, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(144)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(144, 96, kernel_size=3, padding=1), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.7))
        self.attention = nn.Sequential(nn.Linear(64, 24), nn.Tanh(), nn.Linear(24, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 48), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 2))

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv3_1(x))
        c5 = F.gelu(self.conv5_1(x))
        c9 = F.gelu(self.conv9_1(x))
        x = torch.cat([c3, c5, c9], dim=1)
        x = self.drop1(self.bn1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        attn = F.softmax(self.attention(x), dim=1)
        x = (x * attn).sum(dim=1)
        return self.classifier(x)

    def predict_direction(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)
        return direction, confidence


class EnsembleDirectionModel(nn.Module):
    """
    Ensemble de modèles pour prédiction plus robuste
    Combine plusieurs modèles et fait la moyenne des prédictions
    """

    def __init__(
        self,
        models: list,
        voting: str = 'soft'  # 'soft' or 'hard'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble voting"""
        if self.voting == 'soft':
            # Average probabilities
            probs_list = []
            for model in self.models:
                if hasattr(model, 'forward'):
                    logits = model(x) if not isinstance(model.forward(x), tuple) else model(x)[0]
                else:
                    logits = model(x)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)

            avg_probs = torch.stack(probs_list).mean(dim=0)
            return torch.log(avg_probs + 1e-8)  # Return log-probs

        else:  # hard voting
            # Majority vote
            votes = []
            for model in self.models:
                if hasattr(model, 'forward'):
                    logits = model(x) if not isinstance(model.forward(x), tuple) else model(x)[0]
                else:
                    logits = model(x)
                pred = torch.argmax(logits, dim=1)
                votes.append(pred)

            votes = torch.stack(votes)
            # Get mode (most common prediction)
            final_pred = torch.mode(votes, dim=0)[0]

            # Convert to one-hot then to logits
            one_hot = F.one_hot(final_pred, num_classes=2).float()
            return torch.log(one_hot + 1e-8)

    def predict_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict direction with confidence"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)

        return direction, confidence


def create_direction_model(
    model_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create direction prediction models

    Args:
        model_type: Type of model ('standard', 'lightweight', 'ensemble')
        **kwargs: Model-specific parameters

    Returns:
        Direction prediction model
    """
    if model_type == 'standard':
        return DirectionPredictionModel(**kwargs)
    elif model_type == 'lightweight':
        return LightweightDirectionModel(**kwargs)
    elif model_type == 'ensemble':
        # Create multiple models for ensemble
        n_models = kwargs.pop('n_models', 3)
        base_model_type = kwargs.pop('base_model_type', 'standard')
        voting = kwargs.pop('voting', 'soft')

        models = []
        for i in range(n_models):
            if base_model_type == 'standard':
                # Add slight variation to each model
                model_kwargs = kwargs.copy()
                model_kwargs['dropout'] = kwargs.get('dropout', 0.3) + i * 0.05
                models.append(DirectionPredictionModel(**model_kwargs))
            else:
                models.append(LightweightDirectionModel(**kwargs))

        return EnsembleDirectionModel(models, voting=voting)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing Direction Prediction Models...\n")

    batch_size = 8
    sequence_length = 60
    feature_dim = 500

    # Create dummy input
    x = torch.randn(batch_size, sequence_length, feature_dim)

    # Test standard model
    print("1. Standard Model:")
    model = create_direction_model('standard', feature_dim=feature_dim, sequence_length=sequence_length)
    print(f"   Parameters: {count_parameters(model):,}")

    logits, attention = model(x)
    print(f"   Output shape: {logits.shape}")

    direction, confidence = model.predict_direction(x)
    print(f"   Predictions: {direction}")
    print(f"   Confidence: {confidence}\n")

    # Test lightweight model
    print("2. Lightweight Model:")
    light_model = create_direction_model('lightweight', feature_dim=feature_dim, sequence_length=30)
    print(f"   Parameters: {count_parameters(light_model):,}")

    x_light = x[:, :30, :]  # Use shorter sequence
    logits = light_model(x_light)
    print(f"   Output shape: {logits.shape}\n")

    # Test ensemble model
    print("3. Ensemble Model:")
    ensemble = create_direction_model(
        'ensemble',
        feature_dim=feature_dim,
        sequence_length=sequence_length,
        n_models=3,
        voting='soft'
    )
    print(f"   Total parameters: {count_parameters(ensemble):,}")

    logits = ensemble(x)
    print(f"   Output shape: {logits.shape}")

    direction, confidence = ensemble.predict_direction(x)
    print(f"   Predictions: {direction}")
    print(f"   Confidence: {confidence}\n")

    print("Model testing complete!")
