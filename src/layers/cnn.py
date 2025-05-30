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

class EEG_CNN(nn.Module):
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

class EEG_CNN_LSTM(nn.Module):
    def __init__(self, input_channels=19, sequence_length=3000, 
                 cnn_output_channels=128, lstm_hidden_dim=128, 
                 lstm_num_layers=1, lstm_dropout=0.2, 
                 fc_dropout=0.5, num_classes=1, bidirectional_lstm=True):
        """
        A CNN-LSTM hybrid model for EEG seizure detection.
        Args:
            input_channels (int): Number of EEG channels.
            sequence_length (int): Original length of EEG sequence (for context).
            cnn_output_channels (int): Number of output channels from the last CNN layer.
            lstm_hidden_dim (int): Hidden dimension of the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            lstm_dropout (float): Dropout for LSTM (applied if lstm_num_layers > 1).
            fc_dropout (float): Dropout before the final classification layer.
            num_classes (int): Number of output classes.
            bidirectional_lstm (bool): Whether to use a bidirectional LSTM.
        """
        super().__init__()
        self.input_channels = input_channels

        # CNN Part (similar to EEG_CNN's convolutional base)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=cnn_output_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_output_channels)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # Output seq_len will be original_seq_len / 8

        # LSTM Part
        # The input features to LSTM will be cnn_output_channels (features extracted by CNN at each new time step)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_directions = 2 if bidirectional_lstm else 1
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_channels, # Features from CNN for each "time step"
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True, # Crucial: input to LSTM will be (batch, seq_after_cnn_pooling, features_from_cnn)
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional_lstm
        )

        # Fully Connected Layer
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(lstm_hidden_dim * self.lstm_num_directions, num_classes)


    def _forward_cnn(self, x):
        # x expected shape: [batch_size, input_channels, sequence_length]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Output shape: [batch_size, cnn_output_channels, sequence_length_after_pooling]
        return x

    def forward(self, x):
        """
        Input x shape: [batch_size, original_sequence_length, input_channels]
        or [batch_size, input_channels, original_sequence_length]
        """
        # 1. Permute for CNN if necessary: [batch_size, input_channels, original_sequence_length]
        if x.shape[1] == self.input_channels and x.shape[2] != self.input_channels:
            # Assuming input is already [batch_size, input_channels, sequence_length]
            pass
        elif x.shape[2] == self.input_channels and x.shape[1] != self.input_channels:
            # Input is [batch_size, sequence_length, input_channels], permute it
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Input tensor shape {x.shape} is ambiguous or does not match input_channels {self.input_channels}")


        # 2. Pass through CNN feature extractor
        # cnn_out shape: [batch_size, cnn_output_channels, sequence_length_after_cnn_pooling]
        cnn_out = self._forward_cnn(x)

        # 3. Prepare for LSTM: LSTM expects (batch, seq, feature)
        # Permute cnn_out to: [batch_size, sequence_length_after_cnn_pooling, cnn_output_channels]
        lstm_input = cnn_out.permute(0, 2, 1)

        # 4. Pass through LSTM
        # lstm_output shape: [batch_size, sequence_length_after_cnn_pooling, lstm_hidden_dim * num_directions]
        # h_n shape: [num_layers * num_directions, batch_size, lstm_hidden_dim]
        # c_n shape: [num_layers * num_directions, batch_size, lstm_hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # 5. Get the output from the last time step of LSTM
        # (or use attention, mean/max pooling over LSTM outputs, or h_n)
        # lstm_out[:, -1, :] takes the last hidden state from the sequence of outputs.
        # For a bidirectional LSTM, this will be the concatenation of the last forward hidden state
        # and the first backward hidden state (which corresponds to the last time step).
        last_lstm_output = lstm_out[:, -1, :] # Shape: [batch_size, lstm_hidden_dim * num_directions]
        
        # Alternatively, for bidirectional LSTMs, you could explicitly use h_n:
        # if self.lstm.bidirectional:
        #     # h_n is (D*num_layers, N, H_out) where D is 2 for bidirectional
        #     # Get the last forward hidden state: h_n[-2, :, :]
        #     # Get the last backward hidden state: h_n[-1, :, :]
        #     last_lstm_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        # else:
        #     # h_n is (num_layers, N, H_out)
        #     last_lstm_output = h_n[-1,:,:]
        # The current approach using lstm_out[:, -1, :] is generally fine and simpler.

        # 6. Dropout and Final Classification
        last_lstm_output_dropped = self.dropout_fc(last_lstm_output)
        logits = self.fc(last_lstm_output_dropped) # Shape: [batch_size, num_classes]
        
        return logits

