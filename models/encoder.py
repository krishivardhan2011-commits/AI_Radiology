import torch
import torch.nn as nn
import torch.nn.functional as F


class PRO_FA_Encoder(nn.Module):
    """
    PRO-FA Encoder
    Progressive Feature Aggregation Encoder
    Extracts multi-scale radiology features from X-ray image
    """

    def __init__(self, in_channels=1):
        super(PRO_FA_Encoder, self).__init__()

        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Stage 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Feature Aggregation
        self.agg = nn.Conv2d(32 + 64 + 128, 256, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Stage 1
        f1 = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Stage 2
        f2 = self.pool(F.relu(self.bn2(self.conv2(f1))))

        # Stage 3
        f3 = self.pool(F.relu(self.bn3(self.conv3(f2))))

        # Resize features to same size
        f1_up = F.interpolate(f1, size=f3.shape[2:], mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=f3.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate multi-scale features
        combined = torch.cat([f1_up, f2_up, f3], dim=1)

        # Aggregated representation
        out = self.agg(combined)

        return out
