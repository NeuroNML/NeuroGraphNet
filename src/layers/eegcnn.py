import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool1d(nn.Module):
    """Attention Pooling for 1D data (e.g., sequence of features per channel)."""
    def __init__(self, input_feature_dim: int, heads: int = 4):
        super().__init__()
        if input_feature_dim % heads != 0:
            raise ValueError(f"input_feature_dim ({input_feature_dim}) must be divisible by heads ({heads}).")
        
        self.heads = heads
        self.dim_head = input_feature_dim // heads
        self.scale = self.dim_head ** -0.5
        
        # Learnable query parameter, one for each head
        self.q_param = nn.Parameter(torch.randn(heads, 1, self.dim_head)) # (H, 1, C_head) - query per head, broadcastable over batch and time

        self.kv_projection = nn.Linear(input_feature_dim, input_feature_dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Expected x: (B, C, T) -> (Batch, Channels/Features, Time/Sequence)
        B, C, T = x.shape # C is input_feature_dim

        # Project and split into k, v
        # x.permute(0, 2, 1) to make C the last dim for nn.Linear: (B, T, C)
        kv = self.kv_projection(x.permute(0, 2, 1)).chunk(2, dim=-1) # Each is (B, T, C)
        k, v = map(lambda t: t.reshape(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3), kv) # (B, H, T, C_head)
        
        # q shape will be (B, H, 1, C_head) after broadcasting B
        # self.q_param is (H, 1, C_head), scaling it:
        q_scaled = self.q_param * self.scale # (H, 1, C_head)

        # Calculate attention scores: sim = (q * k^T) -> (q_scaled @ k.transpose(-2,-1))
        # Here, using element-wise product then sum, which is a form of scaled dot product if q is broadcast.
        # Simpler: dot product for each head
        # k is (B, H, T, C_head), q_scaled is (H, 1, C_head) -> needs proper alignment for batch dot product
        # Let's adjust q_scaled to (B, H, 1, C_head) for broadcasting with k
        q_for_attn = q_scaled.unsqueeze(0).expand(B, -1, -1, -1) # (B, H, 1, C_head)
        
        # Attention scores: (B, H, 1, C_head) @ (B, H, C_head, T) -> (B, H, 1, T)
        sim = torch.matmul(q_for_attn, k.transpose(-1, -2)) # (B, H, 1, T)
        attn_weights = F.softmax(sim.squeeze(2), dim=-1) # (B, H, T)

        # Apply attention to values v: (B, H, T) x (B, H, T, C_head) -> (B, H, C_head)
        # attn_weights.unsqueeze(2) gives (B, H, 1, T)
        # torch.matmul(attn_weights.unsqueeze(2), v) gives (B, H, 1, C_head)
        pooled_features = torch.matmul(attn_weights.unsqueeze(2), v).squeeze(2) # (B, H, C_head)
        
        # Concatenate heads and reshape
        # pooled_features.permute(0, 1, 2) -> (B, H, C_head)
        return pooled_features.reshape(B, self.heads * self.dim_head) # (B, C)

class EEGCNN(nn.Module):
    def __init__(self, input_channels=19, n_classes=1, base_filters=32, dropout=0.3):
        super().__init__()
        
        # Inner block function
        def _conv_block(c_in, c_out, kernel_size=3, stride=1, dilation=1, padding=None):
            if padding is None:
                padding = dilation # Common for kernel_size=3 to keep length
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, 
                          padding=padding, dilation=dilation),
                nn.GroupNorm(1, c_out), # Equivalent to LayerNorm per channel's features
                nn.ReLU(inplace=True),
            )

        # Stem: Increase channels, maintain sequence length
        self.stem = _conv_block(input_channels, base_filters, kernel_size=7, dilation=1, padding=3) # Larger kernel for stem

        # Layer 0 with MaxPool
        self.layer0_conv = _conv_block(base_filters, base_filters, dilation=1)
        self.down0 = nn.MaxPool1d(2) # Output: T/2

        # Layer 1 & 2 with Residual and MaxPool
        self.layer1_conv = _conv_block(base_filters, base_filters * 2, dilation=1) # To 64 filters
        self.layer2_conv_res = _conv_block(base_filters * 2, base_filters * 2, dilation=2)
        # self.skip1_conv for residual if channel numbers change - not needed if layer1_conv output matches layer2_conv input
        self.down1 = nn.MaxPool1d(2) # Output: T/4

        # Layer 3 & 4 with Residual
        self.layer3_conv = _conv_block(base_filters * 2, base_filters * 4, dilation=1) # To 128 filters
        self.layer4_conv_res = _conv_block(base_filters * 4, base_filters * 4, dilation=2)

        final_cnn_channels = base_filters * 4 # 128 if base_filters=32
        self.attn_pool = AttnPool1d(input_feature_dim=final_cnn_channels, heads=4)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(final_cnn_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (batch_size, sequence_length, input_channels) e.g. (B, T, C)
        """
        x = x.permute(0, 2, 1)  # Permute to (B, C, T) for Conv1D

        # Stem
        x = self.stem(x)        # (B, 32, T)

        # Layer 0
        x = self.layer0_conv(x) # (B, 32, T)
        x = self.down0(x)       # (B, 32, T/2)

        # Layer 1 & 2 (Residual Block)
        res1 = self.layer1_conv(x) # (B, 64, T/2)
        x1 = self.layer2_conv_res(res1) # (B, 64, T/2)
        x = x1 + res1           # Residual connection
        x = self.down1(x)       # (B, 64, T/4)

        # Layer 3 & 4 (Residual Block)
        res2 = self.layer3_conv(x) # (B, 128, T/4)
        x2 = self.layer4_conv_res(res2) # (B, 128, T/4)
        x = x2 + res2           # Residual connection
                                # Output x shape: (B, 128, T/4)
        
        # Attention Pooling
        x_pooled = self.attn_pool(x) # Expected output: (B, 128)
        
        x_dropped = self.dropout_fc(x_pooled)
        logits = self.fc(x_dropped)
        return logits
