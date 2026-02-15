"""Baseline models for comparison."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class LSTMPredictor:
    """LSTM baseline model."""

    def __init__(
        self,
        input_length=96,
        output_length=96,
        num_features=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.num_features = num_features
        self.device = device

        self.model = self._build_model(hidden_dim, num_layers, dropout)
        self.model.to(self.device)

        self.history = {"train_loss": [], "val_loss": []}

    def _build_model(self, hidden_dim, num_layers, dropout):
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_length, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_dim, output_length)
                self.output_length = output_length

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                output = self.fc(last_hidden)
                return output.unsqueeze(-1)

        return LSTMModel(self.num_features, hidden_dim, self.output_length, num_layers, dropout)

    def train_model(
        self, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-3, early_stopping_patience=3, verbose=True
    ):
        """Train LSTM model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if verbose else train_loader

            for batch in train_iterator:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)

                if predictions.shape != y_batch.shape:
                    if y_batch.ndim == 2:
                        y_batch = y_batch.unsqueeze(-1)

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

            avg_train_loss = train_loss / len(train_loader)
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

        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)

                if predictions.shape != y_batch.shape:
                    if y_batch.ndim == 2:
                        y_batch = y_batch.unsqueeze(-1)

                loss = criterion(predictions, y_batch)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions in batches to avoid OOM."""
        self.model.eval()

        all_predictions = []
        num_samples = len(X)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X[i : i + batch_size]
                X_tensor = torch.FloatTensor(batch_X).to(self.device)

                predictions = self.model(X_tensor)
                all_predictions.append(predictions.cpu().numpy())

                # Clear cache to free memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return np.concatenate(all_predictions, axis=0).squeeze()


class PersistenceModel:
    """Naive persistence model."""

    def __init__(self, output_length=96):
        self.output_length = output_length

    def predict(self, X: np.ndarray) -> np.ndarray:
        last_values = X[:, -1:, :]
        predictions = np.repeat(last_values, self.output_length, axis=1)
        return predictions.squeeze()

    def train_model(self, *args, **kwargs):
        pass


class MovingAverageModel:
    """Moving average baseline."""

    def __init__(self, window_size=10, output_length=96):
        self.window_size = window_size
        self.output_length = output_length

    def predict(self, X: np.ndarray) -> np.ndarray:
        window_data = X[:, -self.window_size :, :]
        avg_values = np.mean(window_data, axis=1, keepdims=True)
        predictions = np.repeat(avg_values, self.output_length, axis=1)
        return predictions.squeeze()

    def train_model(self, *args, **kwargs):
        pass
