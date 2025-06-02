import torch.nn as nn

class EEGBaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        '''
        # Extracted features
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        '''
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # Binary output (logit)
        )

    def forward(self, x):
        return self.model(x), 1
