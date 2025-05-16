import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CNN(nn.Module):
    def __init__(self, input_channels=19, num_classes=1, sequence_length=3000, dropout_rate=0.5):
        """
        A 1D CNN model for EEG seizure detection.
        Args:
            input_channels (int): Number of EEG channels (e.g., 19).
            num_classes (int): Number of output classes (1 for binary seizure/non-seizure).
            sequence_length (int): Length of the input EEG sequence (number of time steps).
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length

        # Convolutional layers
        # Conv1d expects input shape (batch_size, channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # Halves sequence length

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # Halves sequence length again

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # Halves sequence length again
        
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the flattened size after convolutions and pooling
        # Example: seq_len = 3000
        # After pool1: 3000 / 2 = 1500
        # After pool2: 1500 / 2 = 750
        # After pool3: 750 / 2 = 375
        # flattened_size = 128 (out_channels from conv3) * (sequence_length // 8)
        
        # Dynamically calculate flattened_size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # Output size of 1 for the time dimension
        self.flattened_size = 128 # (out_channels from conv3) * 1 (output from adaptive pool)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes) # Output logits

    def _forward_conv(self, x):
        # x expected shape: [batch_size, input_channels, sequence_length]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        """
        Input x shape: [batch_size, sequence_length, input_channels] (typical for EEGDataset)
        """
        # Permute to [batch_size, input_channels, sequence_length] for Conv1D
        x = x.permute(0, 2, 1)

        x = self._forward_conv(x) # Output shape: [batch_size, 128, sequence_length_after_pooling]
        
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1) # Shape: [batch_size, flattened_size]
        # If using AdaptiveAvgPool1d:
        # x = self.adaptive_pool(x) # Output shape: [batch_size, 128, 1]
        # x = x.view(x.size(0), -1)   # Shape: [batch_size, 128]

        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x) # Shape: [batch_size, num_classes]
        return logits


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=19, sequence_length=3000, 
                 cnn_output_channels=128, lstm_hidden_dim=128, 
                 lstm_num_layers=1, lstm_dropout=0.2, 
                 fc_dropout=0.5, num_classes=1, bidirectional_lstm=True):
        """
        A CNN-LSTM hybrid model for EEG seizure detection.
        Args:
            input_channels (int): Number of EEG channels.
            sequence_length (int): Original length of EEG sequence.
            cnn_output_channels (int): Number of output channels from the last CNN layer.
            lstm_hidden_dim (int): Hidden dimension of the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            lstm_dropout (float): Dropout for LSTM (applied if lstm_num_layers > 1).
            fc_dropout (float): Dropout before the final classification layer.
            num_classes (int): Number of output classes.
            bidirectional_lstm (bool): Whether to use a bidirectional LSTM.
        """
        super().__init__()

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
        """
        # 1. Permute for CNN: [batch_size, input_channels, original_sequence_length]
        x = x.permute(0, 2, 1)

        # 2. Pass through CNN feature extractor
        # cnn_out shape: [batch_size, cnn_output_channels, sequence_length_after_cnn_pooling]
        cnn_out = self._forward_cnn(x)

        # 3. Prepare for LSTM: LSTM expects (batch, seq, feature)
        # Permute cnn_out to: [batch_size, sequence_length_after_cnn_pooling, cnn_output_channels]
        lstm_input = cnn_out.permute(0, 2, 1)

        # 4. Pass through LSTM
        # lstm_output shape: [batch_size, sequence_length_after_cnn_pooling, lstm_hidden_dim * num_directions]
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # 5. Get the output from the last time step of LSTM
        # (or use attention, mean/max pooling over LSTM outputs)
        last_lstm_output = lstm_out[:, -1, :] # Shape: [batch_size, lstm_hidden_dim * num_directions]
        
        # 6. Dropout and Final Classification
        last_lstm_output_dropped = self.dropout_fc(last_lstm_output)
        logits = self.fc(last_lstm_output_dropped) # Shape: [batch_size, num_classes]
        
        return logits

if __name__ == '__main__':
    batch_size = 4
    seq_len = 3000 # Example sequence length from your project (12s * 250Hz)
    num_channels = 19 # Standard 19 channels

    # Test EEG_CNN
    print("Testing EEG_CNN...")
    cnn_model = EEG_CNN(input_channels=num_channels, sequence_length=seq_len, num_classes=1)
    dummy_data_cnn = torch.randn(batch_size, seq_len, num_channels)
    output_cnn = cnn_model(dummy_data_cnn)
    print("EEG_CNN Output shape:", output_cnn.shape) # Expected: [batch_size, 1]
    assert output_cnn.shape == (batch_size, 1)
    print("EEG_CNN Test Passed!\n")

    # Test CNN_LSTM
    print("Testing CNN_LSTM...")
    cnn_lstm_model = CNN_LSTM(input_channels=num_channels, sequence_length=seq_len, num_classes=1)
    dummy_data_cnn_lstm = torch.randn(batch_size, seq_len, num_channels)
    output_cnn_lstm = cnn_lstm_model(dummy_data_cnn_lstm)
    print("CNN_LSTM Output shape:", output_cnn_lstm.shape) # Expected: [batch_size, 1]
    assert output_cnn_lstm.shape == (batch_size, 1)
    print("CNN_LSTM Test Passed!\n")

    # Example of a BiLSTM-CNN (CNN output fed into BiLSTM)
    print("Testing CNN_BiLSTM (using bidirectional=True by default in CNN_LSTM class)...")
    cnn_bilstm_model = CNN_LSTM(input_channels=num_channels, sequence_length=seq_len, num_classes=1, bidirectional_lstm=True)
    output_cnn_bilstm = cnn_bilstm_model(dummy_data_cnn_lstm) # Using same dummy data
    print("CNN_BiLSTM Output shape:", output_cnn_bilstm.shape)
    assert output_cnn_bilstm.shape == (batch_size, 1)
    print("CNN_BiLSTM Test Passed!\n")