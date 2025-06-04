import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG1DCNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Input: (B, T) → reshape to (B, 1, T)
        x = x.unsqueeze(1)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        print("Pre-activation conv3: min", x.min().item(), "max", x.max().item())
        x = self.activation(x)

        x = self.pool(x).squeeze(-1)  # shape: (B, output_dim)
        print("Post-CNN encoder pooled output stats: mean", x.mean().item(), "std", x.std().item())
        return x


class EEG1DCNNE(nn.Module):
    def __init__(self, out_dim=128,  input_timesteps=3000):
        super().__init__()
        

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=out_dim, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        # Global pooling over time dimension after convolutions
        # The output of conv3 is [B, 128, T'] where T' is the output length.
        # We need to pool this to [B, 128]
        # You can use AdaptiveAvgPool1d to get a fixed output size regardless of T'
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pools to [B, 128, 1]

        self.project = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )

        # The output of the CNN before pooling is the encoder's feature dimension.
        # This will be `output_embedding_dim`
        # self.output_layer = nn.Linear(128, output_embedding_dim) # Not strictly needed if last conv output matches desired dim

    def forward(self, x):
        """
        x: [batch_size, time_steps]
        """
        print(f"Input to EEG1DCNNEncoder.forward - Shape: {x.shape}, Min={x.min().item():.4f}, Max={x.max().item():.4f}, Mean={x.mean().item():.4f}, Std={x.std().item():.4f}")
        
        # CNNs expect (batch, channels, time_steps)
        x = x.unsqueeze(1) # [B, T] -> [B, 1, T]

        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        

        # After convolutions, x is [B, num_filters_last_conv, T_out]
        # For this example, x is [B, 128, T_out]
        
        pooled = self.avg_pool(x).squeeze(-1) # [B, 128, 1] -> [B, 128]

        print(f"CNN Encoder output (pooled): shape={pooled.shape}, min={pooled.min().item():.4f}, max={pooled.max().item():.4f}, mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")

        pooled = self.avg_pool(x).squeeze(-1)  # [B, 128]

        # Normalize across feature dim (per node)
        pooled = (pooled - pooled.mean(dim=1, keepdim=True)) / (pooled.std(dim=1, keepdim=True) + 1e-6)

        pooled = self.project(pooled)  
        print(f"CNN Encoder output (normalized): shape={pooled.shape}, min={pooled.min().item():.4f}, max={pooled.max().item():.4f}, mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")

        return pooled
       