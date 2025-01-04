import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.fetcher.data_fetcher import get_aave_tvl
from src.models.predictor import TVLPredictor

class TVLAnalyzer:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = TVLPredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def prepare_data(self):
        """Prepare data for PyTorch model."""
        raw_data = get_aave_tvl()
        if not raw_data:
            raise ValueError("No data returned by get_aave_tvl().")

        # Convert raw_data to a DataFrame
        df = pd.DataFrame(raw_data)
        
        # Check for missing or invalid TVL data
        if df.empty or 'tvl' not in df.columns or df['tvl'].isnull().all():
            raise ValueError("TVL data is missing or invalid.")
        
        # Drop rows with missing TVL values
        df = df.dropna(subset=['tvl'])
        if df.empty:
            raise ValueError("TVL data is empty after dropping missing values.")

        # Scale TVL values
        scaled_tvl = self.scaler.fit_transform(df['tvl'].values.reshape(-1, 1))
        sequences = []
        targets = []
        
        # Generate sequences for the model
        for i in range(len(scaled_tvl) - self.sequence_length):
            sequences.append(scaled_tvl[i:(i + self.sequence_length)])
            targets.append(scaled_tvl[i + self.sequence_length])
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        return X.reshape(-1, self.sequence_length, 1), y, df
    
    def train_model(self, epochs=100, learning_rate=0.001):
        """Train the PyTorch model."""
        X, y, _ = self.prepare_data()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict_next_days(self, days=7):
        """Predict TVL for the next 7 days."""
        X, _, df = self.prepare_data()
        self.model.eval()
        predictions = []
        last_sequence = X[-1]
        
        for _ in range(days):
            with torch.no_grad():
                pred = self.model(last_sequence.unsqueeze(0))
                predictions.append(pred.item())
                
                last_sequence = torch.cat([
                    last_sequence[1:],
                    pred.reshape(1, 1)
                ], dim=0)
        
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_tvl': predictions.flatten()
        })
    
    def analyze_current_trends(self, historical_df):
        """Analyze current TVL trends using provided historical data.
        
        Args:
            historical_df (pd.DataFrame): DataFrame containing historical TVL data
        """
        if 'tvl' not in historical_df.columns:
            raise ValueError("DataFrame must contain 'tvl' column")
            
        stats = {
            'current_tvl': historical_df['tvl'].iloc[-1],
            'avg_tvl_7d': historical_df['tvl'].tail(7).mean(),
            'avg_tvl_30d': historical_df['tvl'].tail(30).mean(),
            'tvl_change_7d': (historical_df['tvl'].iloc[-1] - historical_df['tvl'].iloc[-7]) / historical_df['tvl'].iloc[-7] * 100,
            'tvl_change_30d': (historical_df['tvl'].iloc[-1] - historical_df['tvl'].iloc[-30]) / historical_df['tvl'].iloc[-30] * 100
        }
        
        return stats
