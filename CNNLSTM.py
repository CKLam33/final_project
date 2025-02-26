import torch
import torch.nn as nn
from evotorch.decorators import pass_info

@pass_info
class CNNLSTM(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        act_length: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Extract channels from observation shape (H, W, C)
        in_channels = obs_shape[-1]  
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_input = None
            # If shape is (W, H)
            if len(obs_shape) == 2:
                dummy_input = torch.zeros(1, *obs_shape).permute(0, 2, 1)

            # If shape is (W, H, C)
            elif len(obs_shape) == 3:
                dummy_input = torch.zeros(*obs_shape).permute(2, 1, 0)  # Channel-first for Conv2d

            # If shape is (Batch, W, H, C)
            elif len(obs_shape) == 4:
                dummy_input = torch.zeros(*obs_shape).permute(0, 3, 2, 1)  # Channel-first for Conv2d
    
            cnn_out_dim = self.cnn(dummy_input).shape[1]
            
        # LSTM Network
        self.lstm = nn.LSTM(
            input_size = cnn_out_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = False
        )
        
        # Final output layer
        self.fc = nn.Linear(hidden_size, act_length)
        
        # Hidden state management
        self.hidden = None

    def forward(self, x):
        # Normalize the input
        x = x / 255.0
        if len(x.shape) == 2:
            # Input shape: (W, H) -> (C, H, W), for grayscale
            x = x.unsqueeze(0).permute(0, 2, 1)

        elif len(x.shape) == 3:
            # Input shape: (C, W, H) -> (batch, C, H, W)
            x = x.permute(2, 1, 0)  # Channel-first for Conv2d

        elif len(x.shape) == 4:
            # Input shape: (C, W, H, D) -> (batch, C, H, W, D)
            x = x.permute(0, 3, 2, 1)  # Channel-first for Conv2d
        
        # LSTM Processing
        lstm_out, self.hidden = self.lstm(self.cnn(x), self.hidden)
        
        # Get final output
        out = self.fc(lstm_out[-1, :]).detach()

        # return a np.int46 value
        return out

    def reset_hidden(self):
        # Initialize hidden state for new episodes
        self.hidden = None
