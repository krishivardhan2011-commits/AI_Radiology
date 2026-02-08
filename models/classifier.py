import torch
import torch.nn as nn
import torch.nn.functional as F


class MIX_MLP_Classifier(nn.Module):
    """
    MIX-MLP Classifier
    Multi-Interaction eXplainable MLP for disease prediction
    Takes PRO-FA encoded features and predicts radiology labels
    """

    def __init__(self, in_features=256, num_classes=14):
        super(MIX_MLP_Classifier, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Mixed MLP layers
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Global feature pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        logits = self.fc3(x)

        return logits
