
import torch.nn as nn


class EEGConvBiLSTM(nn.Module):
# -------------------------- MODEL -----------------------#

    def __init__(self, in_channels):
        super().__init__()
        # Result CNN: compress the time series while increasing the number of channels.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), # Input: 1D time signal for each channel [batch_size, 19, time_samples=fs*12] -> 32 conv. filters (-5+4+1->perseve temporal length)
            nn.ReLU(),
            nn.MaxPool1d(2), # Time sample dim reduced by 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Output: [batch_size, 64, time_samples/4=fs*3]
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        # Output: final hidden state -> [1, batch_size, 128]
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reorder output so features at time steps are in third dimension
        _, (hn, _) = self.rnn(x)    # Just need final hidden message
        out = self.fc(hn.squeeze(0))  # Squeeze: [1, batch_size, 128]  -> [1, batch_size, 128]

        
        return out # No sigmoid to get prob. ->[batch_size, probs.]; already incorporated in loss