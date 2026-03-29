"""
Advanced Training Module for Cryptocurrency Trading Model
==========================================================
State-of-the-art training pipeline with:
1. Mixed precision training for efficiency
2. Learning rate scheduling with warmup
3. Early stopping and model checkpointing
4. Gradient clipping and accumulation
5. Ensemble training
6. Advanced metrics tracking
7. Hyperparameter tuning with Optuna
8. Distributed training support

Author: Advanced Trading System
Version: 2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import wandb
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer with all modern optimizations
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: float = 1.0,
        use_wandb: bool = False,
        checkpoint_dir: str = './checkpoints',
        tensorboard_dir: str = './runs'
    ):
        """
        Initialize advanced trainer

        Args:
            model: PyTorch model to train
            device: Device to train on
            use_mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_clip_value: Value for gradient clipping
            use_wandb: Whether to use Weights & Biases for tracking
            checkpoint_dir: Directory for saving checkpoints
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_value = gradient_clip_value
        self.use_wandb = use_wandb

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # TensorBoard
        self.writer = SummaryWriter(tensorboard_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

        logger.info(f"Trainer initialized on {device}")
        if self.use_mixed_precision:
            logger.info("Mixed precision training enabled")

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        patience: int = 10,
        save_best_only: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with advanced optimizations

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ('adamw', 'sgd', 'radam')
            scheduler_type: Type of scheduler ('cosine', 'onecycle', 'plateau')
            warmup_epochs: Number of warmup epochs
            patience: Early stopping patience
            save_best_only: Whether to save only best model

        Returns:
            Dictionary with training history and best metrics
        """
        # Initialize optimizer
        optimizer = self._create_optimizer(
            optimizer_type,
            learning_rate,
            weight_decay
        )

        # Initialize scheduler
        scheduler = self._create_scheduler(
            scheduler_type,
            optimizer,
            num_epochs,
            warmup_epochs,
            len(train_loader)
        )

        # Loss function with class weighting
        criterion = self._create_loss_function(train_loader)

        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler
            )

            # Validate
            val_metrics = self._validate(val_loader, criterion)

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Learning rate scheduling
            if scheduler_type == 'plateau':
                scheduler.step(val_metrics['loss'])
            elif scheduler_type != 'onecycle':
                scheduler.step()

            # Early stopping check
            if self._check_early_stopping(val_metrics, patience):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint
            if not save_best_only or val_metrics['accuracy'] > self.best_metric:
                self._save_checkpoint(optimizer, scheduler, val_metrics)
                self.best_metric = max(self.best_metric, val_metrics['accuracy'])

        # Training complete
        self._finalize_training()

        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_metric': self.best_metric,
            'final_epoch': self.current_epoch
        }

    def _train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        scheduler
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        all_predictions = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device).squeeze()

            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = criterion(output['logits'], target)
            else:
                output = self.model(data)
                loss = criterion(output['logits'], target)

            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision:
                    self.scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                # Update learning rate (OneCycle)
                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()

            # Track metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            predictions = output['trading_decision'].cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(target.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

            self.global_step += 1

        # Calculate epoch metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss / len(train_loader)

        return metrics

    def _validate(self, val_loader, criterion) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device).squeeze()

                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = criterion(output['logits'], target)
                else:
                    output = self.model(data)
                    loss = criterion(output['logits'], target)

                # Track metrics
                val_loss += loss.item()
                predictions = output['trading_decision'].cpu().numpy()
                probabilities = output['probabilities'][:, 1].cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities)

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = val_loss / len(val_loader)
        metrics['auc'] = roc_auc_score(all_targets, all_probabilities)

        return metrics

    def _calculate_metrics(
        self,
        predictions: List,
        targets: List
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='binary'),
            'recall': recall_score(targets, predictions, average='binary'),
            'f1': f1_score(targets, predictions, average='binary')
        }

    def _create_optimizer(
        self,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float
    ) -> optim.Optimizer:
        """Create optimizer"""
        params = self.model.parameters()

        if optimizer_type == 'adamw':
            return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'radam':
            return optim.RAdam(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(
        self,
        scheduler_type: str,
        optimizer: optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int,
        steps_per_epoch: int
    ):
        """Create learning rate scheduler"""
        if scheduler_type == 'cosine':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=warmup_epochs,
                T_mult=2,
                eta_min=1e-7
            )
        elif scheduler_type == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=warmup_epochs / num_epochs
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def _create_loss_function(self, train_loader) -> nn.Module:
        """Create loss function with class weighting"""
        # Calculate class weights
        targets = []
        for _, target in train_loader:
            targets.extend(target.numpy())

        class_counts = np.bincount(targets)
        class_weights = len(targets) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        return nn.CrossEntropyLoss(weight=class_weights)

    def _check_early_stopping(
        self,
        val_metrics: Dict[str, float],
        patience: int
    ) -> bool:
        """Check early stopping condition"""
        if val_metrics['accuracy'] > self.best_metric:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= patience

    def _save_checkpoint(
        self,
        optimizer,
        scheduler,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'global_step': self.global_step
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Also save as best model
        if metrics['accuracy'] >= self.best_metric:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics to TensorBoard and wandb"""
        # Store metrics
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)

        # TensorBoard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, self.current_epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)

        # wandb logging
        if self.use_wandb:
            wandb.log({
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'epoch': self.current_epoch
            })

        # Console logging
        logger.info(
            f"Epoch {self.current_epoch + 1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

    def _finalize_training(self):
        """Finalize training"""
        self.writer.close()

        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'best_metric': self.best_metric,
                'final_epoch': self.current_epoch
            }, f, indent=2)

        logger.info("Training completed")


class HyperparameterTuner:
    """
    Hyperparameter tuning with Optuna
    """

    def __init__(
        self,
        model_fn: callable,
        train_loader,
        val_loader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize tuner

        Args:
            model_fn: Function to create model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
        """
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def objective(self, trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12]),
            'n_transformer_layers': trial.suggest_int('n_transformer_layers', 2, 8),
            'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [128, 256, 512]),
            'lstm_layers': trial.suggest_int('lstm_layers', 1, 4)
        }

        # Create model with suggested params
        model_config = {
            'd_model': params['d_model'],
            'n_heads': params['n_heads'],
            'n_transformer_layers': params['n_transformer_layers'],
            'lstm_hidden_dim': params['lstm_hidden_dim'],
            'lstm_layers': params['lstm_layers'],
            'dropout': params['dropout']
        }

        model = self.model_fn(model_config)

        # Create trainer
        trainer = AdvancedTrainer(
            model,
            device=self.device,
            use_mixed_precision=True
        )

        # Train with early stopping
        history = trainer.train(
            self.train_loader,
            self.val_loader,
            num_epochs=30,  # Reduced for tuning
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            patience=5
        )

        # Return best validation accuracy
        return history['best_metric']

    def tune(
        self,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning

        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Best parameters and results
        """
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1  # Use 1 job to avoid GPU conflicts
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best value: {best_value}")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }


class EnsembleTrainer:
    """
    Train ensemble of models for robust predictions
    """

    def __init__(
        self,
        model_configs: List[Dict],
        n_models: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize ensemble trainer

        Args:
            model_configs: List of model configurations
            n_models: Number of models in ensemble
            device: Device to use
        """
        self.model_configs = model_configs
        self.n_models = n_models
        self.device = device
        self.models = []
        self.trainers = []

    def train_ensemble(
        self,
        train_loader,
        val_loader,
        **train_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Train ensemble of models

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **train_kwargs: Arguments for training

        Returns:
            List of training histories
        """
        histories = []

        for i in range(self.n_models):
            logger.info(f"Training model {i + 1}/{self.n_models}")

            # Create model with different initialization
            torch.manual_seed(i)
            config = self.model_configs[i % len(self.model_configs)]
            model = self._create_model(config)

            # Create trainer
            trainer = AdvancedTrainer(
                model,
                device=self.device,
                checkpoint_dir=f"./checkpoints/ensemble_{i}"
            )

            # Train model
            history = trainer.train(
                train_loader,
                val_loader,
                **train_kwargs
            )

            histories.append(history)
            self.models.append(model)
            self.trainers.append(trainer)

        return histories

    def _create_model(self, config: Dict) -> nn.Module:
        """Create model from config"""
        # This should be implemented based on your model creation logic
        from new_advanced_model import create_model

        return create_model(
            feature_dim=config.get('feature_dim', 350),
            sequence_length=config.get('sequence_length', 60),
            model_type='advanced',
            config=config
        )

    def predict_ensemble(
        self,
        data_loader,
        voting: str = 'soft'
    ) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            data_loader: Data loader for prediction
            voting: Voting strategy ('soft' or 'hard')

        Returns:
            Ensemble predictions
        """
        all_predictions = []

        for model in self.models:
            model.eval()
            predictions = []

            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    output = model(data)

                    if voting == 'soft':
                        pred = output['probabilities'][:, 1].cpu().numpy()
                    else:
                        pred = output['trading_decision'].cpu().numpy()

                    predictions.extend(pred)

            all_predictions.append(predictions)

        # Combine predictions
        ensemble_pred = np.array(all_predictions)

        if voting == 'soft':
            return np.mean(ensemble_pred, axis=0)
        else:
            return np.round(np.mean(ensemble_pred, axis=0))


if __name__ == "__main__":
    # Test the training module
    print("Testing Advanced Training Module...")

    # Create dummy model and data
    from new_advanced_model import create_model
    import torch.utils.data as data

    # Create model
    model = create_model(
        feature_dim=100,
        sequence_length=60,
        model_type='advanced'
    )

    # Create dummy data
    train_dataset = data.TensorDataset(
        torch.randn(1000, 60, 100),
        torch.randint(0, 2, (1000,))
    )
    val_dataset = data.TensorDataset(
        torch.randn(200, 60, 100),
        torch.randint(0, 2, (200,))
    )

    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Test trainer
    trainer = AdvancedTrainer(model)

    print("\nStarting training test...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=2,  # Quick test
        learning_rate=1e-3
    )

    print(f"\nTraining complete!")
    print(f"Best accuracy: {history['best_metric']:.4f}")

    print("\nAdvanced training module test complete!")