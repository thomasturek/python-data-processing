import torch
import torch.nn as nn

class TVLPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, num_layers=2):
        super(TVLPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions