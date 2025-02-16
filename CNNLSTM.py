import torch
import torch.nn as nn
from evotorch.decorators import pass_info

@pass_info
class CNNLSTM(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        act_length: int,
        hidden_size: int,
        num_layers: int,
        **kwargs
    ):
        super().__init__()
        
        # Extract channels from observation shape (H, W, C)
        in_channels = obs_shape[-1]  
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU6(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU6(),
            nn.Flatten()
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape).permute(0, 3, 2, 1)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
            
        # LSTM Network
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Final output layer
        self.fc = nn.Linear(hidden_size * 2, act_length)
        
        # Hidden state management
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = None

    # if cnn read (c, w, h)
    def forward(self, x):
        # Normalize the input
        x = x / 255.0
        # Input shape: (C, W, H) -> (batch, C, H, W)
        x = x.unsqueeze(0)
        x = x.permute(0, 3, 2, 1)  # Channel-first for Conv2d
        
        # CNN Feature extraction
        features = self.cnn(x)  # (batch, cnn_out_dim)
        
        # LSTM Processing
        lstm_out, self.hidden = self.lstm(features, self.hidden)
        
        # Get final output
        out = self.fc(lstm_out[-1, :]).detach()
        # return a np.int46 value
        return out

    def reset_hidden(self):
        # Initialize hidden state for new episodes
        self.hidden = None
