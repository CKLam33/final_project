import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Project features
        Q = self.query(x).view(batch_size, C, -1).permute(0, 2, 1)
        K = self.key(x).view(batch_size, C, -1)
        V = self.value(x).view(batch_size, C, -1).permute(0, 2, 1)

        # Attention computation
        energy = torch.bmm(Q, K) / (C ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(attention, V)

        # Residual connection
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)
        return self.gamma * out + x
