import torch.nn as nn

class EEGBaselineMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
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
        # Got : 0.71 - DE; 0.87 -first
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # Binary output (logit)
        )
        '''
        self.model = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(128, 32),
                nn.ReLU(),

                nn.Linear(32, 1)
            )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1) # (B, CH, F) -> (B, CH*F)
        return self.model(x), 1
