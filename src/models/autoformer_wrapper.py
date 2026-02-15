"""Autoformer model wrapper for traffic prediction.

This module implements the real Autoformer architecture with:
- Series decomposition (trend + seasonal components)
- Auto-correlation mechanism for period-based dependencies
- Progressive decomposition through encoder-decoder
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import custom Autoformer implementation
from .autoformer_custom import CustomAutoformer


class AutoformerPredictor:
    """Wrapper for HuggingFace Autoformer model with time series forecasting capabilities.

    Implements the Autoformer architecture from Wu et al.:
    - Series Decomposition: Separates trend-cyclical and seasonal components
    - Auto-Correlation Mechanism: O(L log L) complexity for long-term dependencies
    - Progressive Decomposition: Refines predictions through encoder-decoder structure
    """

    def __init__(
        self,
        input_length: int = 96,
        output_length: int = 96,
        num_features: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        moving_average: int = 25,
        autocorrelation_factor: int = 3,
        label_len: int = None,  # Length of decoder start sequence
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize Autoformer predictor.

        Args:
            input_length: Historical window size (context_length)
            output_length: Prediction horizon (prediction_length)
            num_features: Number of input features (input_size)
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            moving_average: Window size for moving average decomposition
            autocorrelation_factor: Factor for auto-correlation mechanism
            device: Device to run model on
        """
        self.input_length = input_length
        self.output_length = output_length
        self.num_features = num_features
        self.device = device

        # Label length for decoder initialization (use last part of encoder input)
        self.label_len = label_len if label_len is not None else output_length // 2

        # Build real Autoformer model
        self.model = self._build_autoformer_model(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            moving_average=moving_average,
            autocorrelation_factor=autocorrelation_factor,
        )
        self.model.to(self.device)

        self.history = {"train_loss": [], "val_loss": []}

        print("✓ Initialized real Autoformer with:")
        print(f"  - Context length: {input_length}")
        print(f"  - Prediction length: {output_length}")
        print(f"  - Model dimension: {d_model}")
        print(f"  - Encoder layers: {e_layers}, Decoder layers: {d_layers}")
        print(f"  - Attention heads: {n_heads}")
        print(f"  - Moving average window: {moving_average}")
        print(f"  - Auto-correlation factor: {autocorrelation_factor}")
        print(f"  - Device: {device}")

    def _build_autoformer_model(
        self,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        moving_average: int,
        autocorrelation_factor: int,
    ) -> CustomAutoformer:
        """Build real Autoformer model using custom implementation.

        This uses our custom Autoformer implementation which faithfully implements:
        - Series Decomposition blocks
        - Auto-Correlation mechanism
        - Progressive decomposition architecture

        Returns:
            CustomAutoformer model with proper configuration
        """
        model = CustomAutoformer(
            input_size=self.num_features,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            moving_avg=moving_average,
            factor=autocorrelation_factor,
            output_length=self.output_length,
        )

        return model

    def train_model(
        self, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-4, early_stopping_patience=3, verbose=True
    ) -> Dict[str, list]:
        """Train the Autoformer model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Maximum number of training epochs
            learning_rate: Learning rate for Adam optimizer
            early_stopping_patience: Number of epochs to wait before early stopping
            verbose: Whether to print training progress

        Returns:
            Dictionary containing training history (train_loss, val_loss)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if verbose else train_loader

            for batch in train_iterator:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # Prepare inputs for Autoformer
                # Encoder input: historical sequence
                x_enc = X_batch  # (batch_size, context_length, input_size)

                # Decoder input: use last part of encoder + zeros for prediction part
                # This gives the decoder some context about the sequence
                batch_size = x_enc.shape[0]
                dec_start = x_enc[:, -self.label_len :, :]  # Last label_len timesteps
                dec_zeros = torch.zeros(
                    batch_size, self.output_length - self.label_len, self.num_features, device=self.device
                )
                x_dec = torch.cat([dec_start, dec_zeros], dim=1)  # (batch_size, output_length, input_size)

                # Forward pass
                predictions = self.model(x_enc, x_dec)  # (batch_size, prediction_length, input_size)

                # Compute MSE loss
                loss = criterion(predictions, y_batch)

                # Check for NaN loss
                if torch.isnan(loss):
                    print("Warning: NaN loss detected at batch. Skipping.")
                    continue

                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

                if verbose:
                    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}")

        # Load best model if validation was used
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def evaluate(self, data_loader, criterion=None) -> float:
        """Evaluate model on validation/test data.

        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function (defaults to MSELoss)

        Returns:
            Average loss over the dataset
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Prepare inputs for Autoformer
                x_enc = X_batch

                # Decoder input: use last part of encoder + zeros
                batch_size = x_enc.shape[0]
                dec_start = x_enc[:, -self.label_len :, :]
                dec_zeros = torch.zeros(
                    batch_size, self.output_length - self.label_len, self.num_features, device=self.device
                )
                x_dec = torch.cat([dec_start, dec_zeros], dim=1)

                # Forward pass
                predictions = self.model(x_enc, x_dec)

                # Compute loss
                loss = criterion(predictions, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions using the Autoformer model.

        Args:
            X: Input data of shape (num_samples, context_length, input_size)
            batch_size: Batch size for prediction to avoid OOM

        Returns:
            Predictions of shape (num_samples, prediction_length)
        """
        self.model.eval()

        all_predictions = []
        num_samples = len(X)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X[i : i + batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)

                # Prepare inputs for Autoformer
                x_enc = X_tensor  # Encoder input
                current_batch_size = x_enc.shape[0]

                # Decoder input: zeros as placeholder
                x_dec = torch.zeros(current_batch_size, self.output_length, self.num_features, device=self.device)

                # Forward pass
                predictions = self.model(x_enc, x_dec)  # (batch_size, prediction_length, input_size)

                all_predictions.append(predictions.cpu().numpy())

                # Clear cache to free memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Concatenate all predictions
        all_preds = np.concatenate(all_predictions, axis=0)  # (num_samples, prediction_length, input_size)

        # Squeeze last dimension if input_size=1
        if self.num_features == 1:
            all_preds = all_preds.squeeze(-1)  # (num_samples, prediction_length)

        return all_preds

    def save_model(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save the model checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
                "config": {
                    "input_length": self.input_length,
                    "output_length": self.output_length,
                    "num_features": self.num_features,
                },
            },
            path,
        )
        print(f"✓ Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load the model checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]
        print(f"✓ Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information.

        Returns:
            Dictionary containing model configuration and parameter count
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_type": "Autoformer (Custom Implementation)",
            "context_length": self.input_length,
            "prediction_length": self.output_length,
            "input_size": self.num_features,
            "d_model": self.model.d_model,
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "device": self.device,
        }


class TrafficDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for traffic prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)

        if y.ndim == 2:
            y = y[:, :, np.newaxis]

        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
