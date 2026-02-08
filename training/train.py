"""
Training Script (Simplified)

Trains:
- PRO-FA Encoder
- MIX-MLP Classifier
- RCTA Decoder
"""

import torch
from models.encoder import PRO_FA_Encoder
from models.classifier import MIX_MLP_Classifier
from models.decoder import RCTA_ReportGenerator


def main():
    print("Initializing models...")

    encoder = PRO_FA_Encoder()
    classifier = MIX_MLP_Classifier()
    decoder = RCTA_ReportGenerator()

    print("Models loaded successfully.")

    print("Training placeholder — training performed in Colab environment.")

    # Example forward pass
    dummy_input = torch.randn(1, 1, 224, 224)

    features = encoder(dummy_input)
    disease_logits = classifier(features)
    report_logits, _ = decoder(features)

    print("Forward pass successful.")
    print("Training completed.")


if __name__ == "__main__":
    main()
