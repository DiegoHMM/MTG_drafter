import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModel(nn.Module):
    def __init__(self, input_dim=2250, hidden_dim1=512, hidden_dim2=256, hidden_dim3 = 256, output_dim=324, dropout=0.5):
        super(FeedForwardModel, self).__init__()
        
        # Definir as camadas 
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Ajustar o input_dim aqui
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,hidden_dim3 )
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mascara):
        # Redimensiona o tensor de entrada para se adequar à MLP
        x = x.view(x.size(0), -1) 
        
        # Camada oculta 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Camada oculta 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        #camada oculta 3
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        
        # Camada de saída
        output = self.fc4(x)
        
        
        # Aplicando a máscara de disponibilidade
        output = output * mascara
        
        return output
