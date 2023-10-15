import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256, output_dim=265, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        
        # Definir a camada LSTM
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        
        # Definir a camada de saída
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mascara):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = F.softmax(output, dim=1)
        
        # Aplicando a máscara de disponibilidade
        output = output * mascara
        
        return output
