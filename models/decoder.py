import torch
import torch.nn as nn
import torch.nn.functional as F


class RCTA_Attention(nn.Module):
    """
    Radiology Cross-Token Attention (RCTA)
    Focuses on important visual regions for report generation
    """

    def __init__(self, feature_dim=256):
        super(RCTA_Attention, self).__init__()

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.scale = feature_dim ** 0.5

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.size()

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_map = torch.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_map, V)

        attended_features = attended_features.permute(0, 2, 1).view(B, C, H, W)

        return attended_features, attention_map


class RCTA_ReportGenerator(nn.Module):
    """
    RCTA Decoder: Generates Radiology Report Text
    """

    def __init__(self, feature_dim=256, hidden_dim=512, vocab_size=1000, max_len=60):
        super(RCTA_ReportGenerator, self).__init__()

        self.attention = RCTA_Attention(feature_dim)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.max_len = max_len

    def forward(self, features):
        # Apply attention
        attended, attn_map = self.attention(features)

        pooled = self.global_pool(attended)
        pooled = pooled.view(pooled.size(0), 1, -1)  # (B, 1, C)

        # Repeat feature for sequence generation
        seq_input = pooled.repeat(1, self.max_len, 1)

        lstm_out, _ = self.lstm(seq_input)

        logits = self.fc(lstm_out)

        return logits, attn_map
