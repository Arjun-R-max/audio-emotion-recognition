import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Because after 3 pools: 128x128 -> 16x16
        cnn_output_size = 128 * 16 * 16  # = 32768

        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)            # (batch, 128, 16, 16)

        # reshape for LSTM
        x = x.view(x.size(0), 1, -1)    # (batch, seq=1, 2048)

        out, _ = self.lstm(x)      # LSTM input is (batch, seq, features)

        out = out[:, -1, :]
        out = self.fc(out)

        return out
