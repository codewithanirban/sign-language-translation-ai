import torch
import torch.nn as nn

class GRUAttentionModel(nn.Module):
    """
    GRU + Attention model for sequence classification from MediaPipe hand keypoints.
    Input: (batch, seq_len, 63)
    Output: (batch, num_classes)
    """
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.attn = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden_dim)
        out = self.classifier(context)  # (batch, num_classes)
        return out