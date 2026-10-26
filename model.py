from torch import nn
from config import *

class RNN(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=128, num_layers=2, num_labels=60, output_dim=2):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.pool = nn.AdaptiveAvgPool1d(num_labels)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, mel):
        # mel: (B, 1, 64, 938)
        mel = mel.squeeze(1)            # (B, 64, 938)

        # RNN 需要 (B, T, feature)
        x = mel.transpose(1, 2)             # (B, 938, 64)

        # RNN 沿 938 步展开
        x, _ = self.rnn(x)                  # (B, 938, hidden*2)

        # 池化：时间轴 938 → 60
        x = x.transpose(1, 2)               # (B, hidden*2, 938)
        x = self.pool(x)                    # (B, hidden*2, 60)
        x = x.transpose(1, 2)               # (B, 60, hidden*2)

        # 输出 60 个时间点，每个 2 个值
        x = self.fc(x)                      # (B, 60, 2)
        return x
