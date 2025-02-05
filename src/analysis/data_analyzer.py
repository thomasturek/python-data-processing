import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.fetcher.data_fetcher import get_aave_tvl
from src.models.predictor import TVLPredictor
from src.utils.model_utils import ModelCheckpointer

class TVLAnalyzer:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = TVLPredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.checkpointer = ModelCheckpointer()
    
    def prepare_data(self):
        """Prepare data for PyTorch model."""
        df = get_aave_tvl()
        
        if df is None or df.empty:
            raise ValueError("No data returned by get_aave_tvl().")

        if len(df) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} data points")
            
        # Scale TVL values
        scaled_tvl = self.scaler.fit_transform(df['tvl'].values.reshape(-1, 1))
        sequences = []
        targets = []
        
        # Generate sequences for the model
        for i in range(len(scaled_tvl) - self.sequence_length):
            sequences.append(scaled_tvl[i:(i + self.sequence_length)])
            targets.append(scaled_tvl[i + self.sequence_length])
        
        if not sequences:
            raise ValueError("No sequences could be created from the data")
            
        # Convert to PyTorch tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        return X.reshape(-1, self.sequence_length, 1), y, df
    
    def train_model(self, epochs=100, learning_rate=0.001, force_retrain=False):
        """Train the PyTorch model with checkpointing and enhanced training procedure."""
        X, y, _ = self.prepare_data()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Try to load previous checkpoint if not force_retrain
        start_epoch = 0
        best_loss = float('inf')
        if not force_retrain:
            model, start_epoch, best_loss = self.checkpointer.load_latest_model(
                self.model, 
                optimizer, 
                self.scaler
            )
            if model is not None:
                print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")
                self.model = model
            else:
                print("No checkpoint found, starting fresh training")
        
        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=1
        )
        
        # Loss functions for different components
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss()
        
        patience = 10
        patience_counter = 0
        
        self.model.train()
        for epoch in range(start_epoch, epochs):
            # Training step
            optimizer.zero_grad()
            final_pred, trend_pred, vol_pred = self.model(X_train)
            
            # Calculate losses for different components
            final_loss = mse_loss(final_pred, y_train)
            trend_loss = huber_loss(trend_pred, y_train)
            vol_loss = mse_loss(vol_pred, y_train)
            
            # Combined loss with weighting
            total_loss = final_loss + 0.3 * trend_loss + 0.2 * vol_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                final_pred_val, _, _ = self.model(X_val)
                val_loss = mse_loss(final_pred_val, y_val)
            
            # Save checkpoint if we have the best validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.checkpointer.save_model(
                    self.model,
                    optimizer,
                    epoch + 1,
                    val_loss.item(),
                    self.scaler
                )
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Loss: {total_loss.item():.4f}, '
                      f'Val Loss: {val_loss.item():.4f}')
            
            self.model.train()
    
    def predict_next_days(self, days=7):
        """Predict TVL for the next 7 days."""
        X, _, df = self.prepare_data()
        self.model.eval()
        predictions = []
        trend_predictions = []
        volatility_predictions = []
        last_sequence = X[-1]
        
        for _ in range(days):
            with torch.no_grad():
                final_pred, trend_pred, vol_pred = self.model(last_sequence.unsqueeze(0))
                predictions.append(final_pred.item())
                trend_predictions.append(trend_pred.item())
                volatility_predictions.append(vol_pred.item())
                
                last_sequence = torch.cat([
                    last_sequence[1:],
                    final_pred.reshape(1, 1)
                ], dim=0)
        
        # Inverse transform all predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        trend_predictions = self.scaler.inverse_transform(
            np.array(trend_predictions).reshape(-1, 1)
        )
        volatility_predictions = self.scaler.inverse_transform(
            np.array(volatility_predictions).reshape(-1, 1)
        )
        
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_tvl': predictions.flatten(),
            'trend_prediction': trend_predictions.flatten(),
            'volatility_prediction': volatility_predictions.flatten()
        })
    
    def analyze_current_trends(self, historical_df):
        """Analyze current TVL trends using provided historical data."""
        if not isinstance(historical_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if historical_df.empty:
            raise ValueError("Input DataFrame is empty")
            
        if 'tvl' not in historical_df.columns:
            raise ValueError("DataFrame must contain 'tvl' column")
            
        if len(historical_df) < 30:
            raise ValueError("Need at least 30 days of data for trend analysis")
            
        stats = {
            'current_tvl': historical_df['tvl'].iloc[-1],
            'avg_tvl_7d': historical_df['tvl'].tail(7).mean(),
            'avg_tvl_30d': historical_df['tvl'].tail(30).mean(),
            'tvl_change_7d': (historical_df['tvl'].iloc[-1] - historical_df['tvl'].iloc[-7]) / historical_df['tvl'].iloc[-7] * 100,
            'tvl_change_30d': (historical_df['tvl'].iloc[-1] - historical_df['tvl'].iloc[-30]) / historical_df['tvl'].iloc[-30] * 100
        }
        
        return stats