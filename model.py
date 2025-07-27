import torch.nn as nn

class ShorthandModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(input_size=64 * 8, hidden_size=128, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)  # BiLSTM = 2x

    def forward(self, x):
        x = self.cnn(x)               # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)     # [B, W, C, H]
        x = x.flatten(2)              # [B, W, C*H]
        x, _ = self.rnn(x)            # [B, W, 256]
        return self.fc(x)             # [B, W, num_classes]

