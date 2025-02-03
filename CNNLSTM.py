import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, obs_shape, action_space_size=30):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dynamic CNN output size calculation
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[-1], obs_shape[1], obs_shape[2])
            self.cnn_output_size = self.cnn(dummy_input).shape[1]

        # LSTM and final layers
        self.lstm = nn.LSTM(self.cnn_output_size, 512, 
                        batch_first=True,
                        dropout=0.2,
                        bidirectional=False)  # Keep unidirectional for GA

        self.fc = nn.Linear(512, action_space_size)
        
    def forward(self, x, hidden=None):
        features = self.cnn(x.permute(0, 3, 1, 2))  # NHWC to NCHW
        lstm_out, hidden = self.lstm(features.unsqueeze(1), hidden)
        return torch.softmax(self.fc(lstm_out.squeeze(1)), dim=-1), hidden