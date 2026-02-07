import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchxrayvision as xrv
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cognitive Radiology AI", layout="centered")

st.title("🧠 Cognitive Radiology AI")
st.write("Upload a Chest X-ray → Get AI Report + Heatmap")

# Load model
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

model = load_model()
diseases = xrv.datasets.default_pathologies

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # 1 channel
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    img_np = np.array(img)

    # Normalize like XRV expects
    img_np = xrv.datasets.normalize(img_np, 255)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])

    img_t = transform(img_np).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_t)

    scores = {diseases[i]: float(preds[0][i]) for i in range(len(diseases))}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    st.subheader("Top Findings")

    for k, v in list(scores.items())[:5]:
        st.write(f"**{k}** : {v:.3f}")
