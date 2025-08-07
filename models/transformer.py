import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """
    Lightweight Transformer for sequence classification from MediaPipe hand keypoints.
    Input: (batch, seq_len, 63)
    Output: (batch, num_classes)
    """
    def __init__(self, input_dim=63, model_dim=128, num_heads=4, num_layers=2, num_classes=10, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, model_dim)
        x = self.transformer(x) # (batch, seq_len, model_dim)
        x = x.mean(dim=1)       # (batch, model_dim)
        out = self.classifier(x)
        return out