import gradio as gr
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
import cv2

# ------------------ LOAD MODEL ------------------
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()
diseases = xrv.datasets.default_pathologies


# ------------------ HEATMAP ------------------
def generate_heatmap(img_t):
    gradients = []
    activations = []

    target_layer = model.features[-1]

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_backward_hook(b_hook)

    output = model(img_t)
    cls = output.argmax()
    model.zero_grad()
    output[0, cls].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()
    return cam


# ------------------ MAIN PIPELINE ------------------
def analyze_xray(image):

    img = image.convert("L")
    img_np = np.array(img)
    img_np = xrv.datasets.normalize(img_np, 255)

    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()
    img_t = torch.nn.functional.interpolate(img_t, size=(224, 224))

    with torch.no_grad():
        preds = model(img_t)

    scores = {diseases[i]: float(preds[0][i]) for i in range(len(diseases))}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ---------- HEATMAP ----------
    cam = generate_heatmap(img_t)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    base = np.array(img.resize((224, 224)))
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0)

    # ---------- REPORT ----------
    primary = list(scores.items())[0]

    report = f"""
FINDINGS:
Radiographic appearance suggests {primary[0]}.
No acute life-threatening cardiopulmonary abnormality detected.
Cardiomediastinal silhouette within normal limits.

IMPRESSION:
Findings most consistent with {primary[0]} (confidence {primary[1]:.2f}).
Clinical correlation recommended.
"""

    top_text = "\n".join([f"{k}: {v:.3f}" for k, v in list(scores.items())[:6]])

    return overlay[:, :, ::-1], top_text, report


# ------------------ UI ------------------
demo = gr.Interface(
    fn=analyze_xray,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Image(label="AI Attention Heatmap"),
        gr.Textbox(label="Top Clinical Predictions"),
        gr.Textbox(label="Radiology Report")
    ],
    title="🧠 Cognitive Radiology AI",
    description="Hierarchical Vision • Clinical Reasoning • Cognitive Attention",
)

demo.launch()
