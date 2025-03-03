import torch
import torch.nn as nn
from evotorch.decorators import pass_info

from AttentionModel import AttentionModel

@pass_info
class CNN(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        act_length: int,
        use_AttentionModel: bool = False,
        **kwargs
    ):
        super().__init__()

        self.use_AttentionModel = use_AttentionModel

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
        )

        self.flatten = nn.Flatten()
        
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

            cnn_out_dim = self.flatten(self.cnn(dummy_input)).shape[1]

        
        self.attention = AttentionModel(64)
            
        # Final output layer
        self.fcout = nn.Sequential(nn.Linear(cnn_out_dim, act_length), nn.ReLU())

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

        # Apply CNN
        cnn_out = self.cnn(x)

        if self.use_AttentionModel:
            attn_out = self.attention(cnn_out)
            flattened = self.flatten(attn_out)
        else:
            # Get final output
            flattened = self.flatten(cnn_out)
        out = self.fcout(flattened[-1, :]).detach()

        # return a np.int46 value
        return out
