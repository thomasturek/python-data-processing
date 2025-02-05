import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(attention_weights * lstm_output, dim=1)

class TVLPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=3, dropout=0.2):
        super(TVLPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Multiple LSTM layers with different purposes
        self.trend_lstm = nn.LSTM(
            input_size=1, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.volatility_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanisms
        self.trend_attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        self.volatility_attention = AttentionLayer(hidden_size//2)
        
        # Prediction heads
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)
        )
        
        # Final aggregation layer
        self.final_layer = nn.Sequential(
            nn.Linear(2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
    
    def forward(self, x):
        # Trend analysis
        trend_out, _ = self.trend_lstm(x)
        trend_att = self.trend_attention(trend_out)
        trend_pred = self.trend_predictor(trend_att)
        
        # Volatility analysis
        vol_out, _ = self.volatility_lstm(x)
        vol_att = self.volatility_attention(vol_out)
        vol_pred = self.volatility_predictor(vol_att)
        
        # Combine predictions
        combined = torch.cat([trend_pred, vol_pred], dim=1)
        final_pred = self.final_layer(combined)
        
        return final_pred, trend_pred, vol_pred