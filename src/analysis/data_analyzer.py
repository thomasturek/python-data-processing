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
        # Getting raw data from our fetcher file
        df = get_aave_tvl()
        
        if df.empty:
            raise ValueError("No data available for analysis")
            
        # Ensure we have enough data for the sequence length
        if len(df) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} data points")
        
        # Scaling the TVL values
        scaled_tvl = self.scaler.fit_transform(df['tvl'].values.reshape(-1, 1))
        
        # Creating sequences
        sequences = []
        targets = []
        
        for i in range(len(scaled_tvl) - self.sequence_length):
            sequences.append(scaled_tvl[i:(i + self.sequence_length)])
            targets.append(scaled_tvl[i + self.sequence_length])
        
        if not sequences:
            raise ValueError("No sequences could be created from the data")
            
        # Converting to correct PyTorch tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        return X.reshape(-1, self.sequence_length, 1), y, df
    
    def train_model(self, epochs=100, learning_rate=0.001):
        """Train the PyTorch model."""
        # Calling and and preparing data
        X, y, _ = self.prepare_data()
        
        # Defining loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop for our training
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
                
                # Updating the sequence for the next prediction
                last_sequence = torch.cat([
                    last_sequence[1:],
                    pred.reshape(1, 1)
                ], dim=0)
        
        # Inverse transforming predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        # Generating dates for predictions
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_tvl': predictions.flatten()
        })
    
    def analyze_current_trends(self):
        """Analyze current TVL trends."""
        _, _, df = self.prepare_data()
        
        # Calculating basic statistics
        stats = {
            'current_tvl': df['tvl'].iloc[-1],
            'avg_tvl_7d': df['tvl'].tail(7).mean(),
            'avg_tvl_30d': df['tvl'].tail(30).mean(),
            'tvl_change_7d': (df['tvl'].iloc[-1] - df['tvl'].iloc[-7]) / df['tvl'].iloc[-7] * 100,
            'tvl_change_30d': (df['tvl'].iloc[-1] - df['tvl'].iloc[-30]) / df['tvl'].iloc[-30] * 100
        }
        
        return stats