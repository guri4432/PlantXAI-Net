"""
PlantXAI-Net — Gradio Web Interface
=====================================
Upload a plant image → full 5-stage AI pipeline:
  Stage 1 : Image Preprocessing
  Stage 2 : Disease Classification  (EfficientNet-B0, 38 classes)
  Stage 3 : Grad-CAM XAI Heatmap
  Stage 4 : Severity Estimation     (Mild / Moderate / Severe)
  Stage 5 : Treatment Recommendation

Run:
    python app.py
Then the browser opens automatically at  http://127.0.0.1:7860
"""

# ==============================================================================
# Standard Library
# ==============================================================================

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ==============================================================================
# Numerical / Vision
# ==============================================================================

import cv2
import numpy as np
from PIL import Image

# ==============================================================================
# Deep Learning
# ==============================================================================

import torch
import torch.nn as nn
from torchvision import models, transforms

# ==============================================================================
# Gradio
# ==============================================================================

import gradio as gr

# ==============================================================================
# Paths
# ==============================================================================

PROJECT_ROOT    = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "last_checkpoint.pth"

# ==============================================================================
# 38 PlantVillage Class Names  (alphabetical — matches ImageFolder order)
# ==============================================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)   # 38

# ==============================================================================
# ImageNet normalisation (same as training)
# ==============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.70

# ==============================================================================
# Disease Recommendations
# ==============================================================================

DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab":
        "Remove infected fallen leaves and affected plant material, "
        "maintain canopy airflow, and avoid prolonged leaf wetness.",
    "Apple___Black_rot":
        "Remove mummified fruit and infected plant material, "
        "maintain orchard sanitation, and prune affected branches where practical.",
    "Apple___Cedar_apple_rust":
        "Remove heavily affected plant material where practical, "
        "maintain orchard sanitation, and monitor for recurring symptoms.",
    "Apple___healthy":
        "No disease treatment indicated. Continue routine monitoring and preventive crop management.",
    "Blueberry___healthy":
        "No disease treatment indicated. Continue routine monitoring and maintain good crop hygiene.",
    "Cherry_(including_sour)___Powdery_mildew":
        "Improve airflow, avoid excessive humidity around foliage, "
        "remove heavily affected material where practical, and monitor progression.",
    "Cherry_(including_sour)___healthy":
        "No disease treatment indicated. Continue routine monitoring and maintain good sanitation.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Maintain field sanitation, improve crop airflow where possible, "
        "and monitor disease progression.",
    "Corn_(maize)___Common_rust_":
        "Monitor disease progression, maintain crop health, "
        "and use locally recommended disease-management practices when necessary.",
    "Corn_(maize)___Northern_Leaf_Blight":
        "Maintain field sanitation, monitor spread, "
        "and use locally recommended disease-management practices.",
    "Corn_(maize)___healthy":
        "No disease treatment indicated. Continue routine crop monitoring and preventive management.",
    "Grape___Black_rot":
        "Remove infected fruit and plant material where practical, "
        "maintain vineyard sanitation, and improve canopy airflow.",
    "Grape___Esca_(Black_Measles)":
        "Remove or manage severely affected plant material where practical, "
        "maintain vineyard sanitation, and seek local agronomic guidance.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Remove affected plant material where practical, "
        "improve canopy airflow, and maintain vineyard sanitation.",
    "Grape___healthy":
        "No disease treatment indicated. Continue routine vineyard monitoring and sanitation.",
    "Orange___Haunglongbing_(Citrus_greening)":
        "Monitor affected trees carefully, manage infected plant material "
        "according to local citrus-health guidance, and control disease vectors.",
    "Peach___Bacterial_spot":
        "Remove severely affected plant material where practical, "
        "maintain orchard sanitation, and reduce conditions favoring prolonged leaf wetness.",
    "Peach___healthy":
        "No disease treatment indicated. Continue routine monitoring and orchard sanitation.",
    "Pepper,_bell___Bacterial_spot":
        "Remove affected plant material where practical, maintain sanitation, "
        "and avoid practices that spread moisture and contamination between plants.",
    "Pepper,_bell___healthy":
        "No disease treatment indicated. Continue routine monitoring and maintain good crop hygiene.",
    "Potato___Early_blight":
        "Remove affected foliage where practical, maintain field sanitation, "
        "and monitor disease progression.",
    "Potato___Late_blight":
        "Remove or isolate severely affected plant material where practical, "
        "monitor closely for rapid spread, and seek locally recommended late-blight management guidance.",
    "Potato___healthy":
        "No disease treatment indicated. Continue routine crop monitoring and preventive management.",
    "Raspberry___healthy":
        "No disease treatment indicated. Continue routine monitoring and maintain good crop hygiene.",
    "Soybean___healthy":
        "No disease treatment indicated. Continue routine crop monitoring and preventive field management.",
    "Squash___Powdery_mildew":
        "Improve airflow, remove heavily affected leaves where practical, "
        "and monitor disease development.",
    "Strawberry___Leaf_scorch":
        "Remove severely affected foliage where practical, "
        "maintain plant sanitation, and monitor symptom progression.",
    "Strawberry___healthy":
        "No disease treatment indicated. Continue routine monitoring and maintain good crop hygiene.",
    "Tomato___Bacterial_spot":
        "Remove affected plant material where practical, maintain sanitation, "
        "and reduce practices that spread moisture between plants.",
    "Tomato___Early_blight":
        "Remove affected foliage where practical, improve airflow, and maintain field sanitation.",
    "Tomato___Late_blight":
        "Monitor closely for rapid disease progression, remove severely affected material "
        "where practical, and seek locally recommended late-blight management guidance.",
    "Tomato___Leaf_Mold":
        "Improve ventilation and airflow, reduce prolonged leaf wetness, "
        "and remove heavily affected material where practical.",
    "Tomato___Septoria_leaf_spot":
        "Remove affected foliage where practical, improve airflow, "
        "and maintain good field sanitation.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Monitor the underside of leaves, remove heavily affected material where practical, "
        "and use locally appropriate integrated pest-management practices.",
    "Tomato___Target_Spot":
        "Remove affected foliage where practical, improve airflow, and maintain field sanitation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Remove severely affected plants where appropriate, monitor nearby plants, "
        "and manage insect vectors using locally recommended integrated pest-management practices.",
    "Tomato___Tomato_mosaic_virus":
        "Remove severely affected plants where appropriate, maintain sanitation, "
        "and prevent mechanical spread through contaminated tools or handling.",
    "Tomato___healthy":
        "No disease treatment indicated. Continue routine monitoring and preventive crop management.",
}

MILD_ACTION = (
    "Monitor the plant closely, remove visibly affected leaves where practical, "
    "maintain sanitation, and improve airflow around the plant."
)
MODERATE_ACTION = (
    "Remove affected plant material where practical, improve airflow and sanitation, "
    "and consider an appropriate disease-control treatment according to local "
    "agricultural guidance and product labeling."
)
SEVERE_ACTION = (
    "Isolate or remove severely affected plant material where practical, "
    "improve field sanitation, and seek appropriate crop-management or "
    "agronomic guidance for disease-control treatment."
)


# ==============================================================================
# Model
# ==============================================================================

def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features=in_features, out_features=NUM_CLASSES),
    )
    return model


def load_model() -> nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Ensure models/checkpoints/last_checkpoint.pth exists."
        )
    model = build_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state_dict" not in checkpoint:
        raise KeyError("model_state_dict not found in checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print(f"[PlantXAI-Net] Model loaded on {DEVICE}")
    return model


print("[PlantXAI-Net] Loading EfficientNet-B0 ...")
MODEL = load_model()
print("[PlantXAI-Net] Ready.")


# ==============================================================================
# Stage 2: Classification
# ==============================================================================

def classify_image(pil_img: Image.Image):
    tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = MODEL(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASS_NAMES[int(idx.item())], float(conf.item()), probs[0].cpu().numpy()


# ==============================================================================
# Stage 3: Grad-CAM
# ==============================================================================

def generate_gradcam(pil_img: Image.Image, class_idx: int) -> np.ndarray:
    w, h   = pil_img.size
    tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    target_layer = MODEL.features[-1]
    activations, gradients = [], []

    fwd = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach()) if go else None)

    try:
        MODEL.eval()
        MODEL.zero_grad(set_to_none=True)
        out = MODEL(tensor)
        out[0, class_idx].backward()

        if not activations or not gradients:
            return np.zeros((h, w), dtype=np.float32)

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * activations[0]).sum(dim=1))[0]
        cam     = cam.cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(cam, 0.0, 1.0).astype(np.float32)
    finally:
        fwd.remove()
        bwd.remove()


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb      = np.array(pil_img.convert("RGB"))
    colored  = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (alpha * colored + (1 - alpha) * rgb).astype(np.uint8)


# ==============================================================================
# Stage 4: Severity Estimation
# ==============================================================================

def estimate_severity(pil_img: Image.Image, is_healthy: bool):
    if is_healthy:
        return "Healthy", 0.0

    rgb  = np.array(pil_img.convert("RGB"))
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]

    leaf_mask  = (sat > 30) & (val > 30) & (val < 250)
    green_mask = (hue >= 30) & (hue <= 90) & leaf_mask

    leaf_px  = float(leaf_mask.sum())
    green_px = float(green_mask.sum())

    if leaf_px < 100:
        return "Mild", 5.0

    affected = max(0.0, (1.0 - green_px / leaf_px) * 100.0)
    level    = "Mild" if affected < 20 else "Moderate" if affected < 50 else "Severe"
    return level, round(affected, 1)


# ==============================================================================
# Stage 5: Recommendation
# ==============================================================================

def get_recommendation(disease: str, severity: str, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "Prediction confidence is below 70%. Please retake the image under "
            "clear, even lighting and try again. Consider expert agronomic "
            "verification before taking any action."
        )
    mgmt = DISEASE_RECOMMENDATIONS.get(disease, "")
    if disease.endswith("healthy"):
        return mgmt
    action = {"Mild": MILD_ACTION, "Moderate": MODERATE_ACTION, "Severe": SEVERE_ACTION}.get(
        severity, SEVERE_ACTION
    )
    return (
        f"{mgmt} {action} "
        "Any chemical or biological treatment should be selected "
        "according to local agricultural guidance and the product label."
    )


# ==============================================================================
# HTML helpers
# ==============================================================================

def fmt_class(raw: str):
    if "___" in raw:
        p, d = raw.split("___", 1)
        return p.replace("_", " "), d.replace("_", " ")
    return raw.replace("_", " "), "Unknown"


SEV_COLOR = {"Healthy": "#22c55e", "Mild": "#f59e0b", "Moderate": "#f97316", "Severe": "#ef4444"}
SEV_ICON  = {"Healthy": "🌿", "Mild": "🟡", "Moderate": "🟠", "Severe": "🔴"}


def conf_bar(conf: float) -> str:
    pct   = round(conf * 100, 1)
    color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
    return (
        f'<div style="margin:6px 0 12px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:13px;'
        f'color:#555 !important;margin-bottom:4px;"><span>Confidence</span>'
        f'<span><b>{pct}%</b></span></div>'
        f'<div style="background:#e5e7eb;border-radius:8px;height:12px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:8px;"></div>'
        f'</div></div>'
    )


def top5_table(all_probs: np.ndarray) -> str:
    top5 = np.argsort(all_probs)[::-1][:5]
    rows = ""
    for rank, idx in enumerate(top5):
        plant, dis = fmt_class(CLASS_NAMES[idx])
        pct        = round(float(all_probs[idx]) * 100, 2)
        bar_color  = "#3b82f6" if rank == 0 else "#93c5fd"
        bold       = "700" if rank == 0 else "400"
        tcol       = "#2563eb" if rank == 0 else "#64748b"
        rows += (
            f'<tr style="border-bottom:1px solid #f0f0f0;">'
            f'<td style="padding:5px 8px;font-weight:{bold};color:#1e293b !important;">{plant}</td>'
            f'<td style="padding:5px 8px;color:#475569 !important;">{dis}</td>'
            f'<td style="padding:5px 8px;width:120px;">'
            f'<div style="background:#e5e7eb;border-radius:4px;height:7px;">'
            f'<div style="width:{min(pct,100)}%;height:100%;background:{bar_color};border-radius:4px;"></div>'
            f'</div></td>'
            f'<td style="padding:5px 8px;text-align:right;font-weight:{bold};color:{tcol} !important;">{pct}%</td>'
            f'</tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:#f8fafc;color:#64748b !important;font-size:11px;">'
        '<th style="padding:5px 8px;text-align:left;">Plant</th>'
        '<th style="padding:5px 8px;text-align:left;">Disease</th>'
        '<th style="padding:5px 8px;text-align:left;">Probability</th>'
        '<th style="padding:5px 8px;text-align:right;">%</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(input_image):
    if input_image is None:
        err = "<p style='color:#ef4444;font-weight:600;padding:20px;'>⚠️ Please upload an image first.</p>"
        return None, None, err, err, err, err

    pil_img = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")

    # ── Stage 1: Preprocessing ────────────────────────────────────────
    preprocessed_arr = np.array(pil_img.resize((224, 224)))

    s1 = (
        '<div style="background:var(--background-fill-secondary);border-left:4px solid #22c55e;border-radius:8px;'
        'padding:14px 18px;margin:6px 0;">'
        '<div style="font-size:15px;font-weight:700;color:#22c55e;margin-bottom:8px;">'
        '✅ Stage 1 — Image Preprocessing</div>'
        '<ul style="margin:0;padding-left:18px;color:var(--body-text-color);font-size:13px;line-height:1.9;">'
        '<li>Image loaded and converted to <b>RGB</b></li>'
        '<li>Resized to <b>224 × 224 px</b></li>'
        '<li>Normalized: ImageNet mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)</li>'
        '<li>Ready for EfficientNet-B0 inference ✓</li>'
        '</ul></div>'
    )

    # ── Stage 2: Classification ───────────────────────────────────────
    pred_class, conf, all_probs = classify_image(pil_img)
    plant, disease = fmt_class(pred_class)
    is_healthy     = pred_class.endswith("healthy")
    class_idx      = CLASS_NAMES.index(pred_class)
    badge_color    = "#22c55e" if is_healthy else "#ef4444"
    badge_icon     = "🌿" if is_healthy else "🦠"
    conf_pct       = round(conf * 100, 1)

    s2 = (
        '<div style="background:var(--background-fill-secondary);border-left:4px solid #3b82f6;border-radius:8px;'
        'padding:14px 18px;margin:6px 0;">'
        '<div style="font-size:15px;font-weight:700;color:#3b82f6;margin-bottom:10px;">'
        '✅ Stage 2 — Disease Classification</div>'
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">'
        f'<div style="font-size:32px;">{badge_icon}</div>'
        f'<div><div style="font-size:20px;font-weight:800;color:var(--body-text-color);">{plant}</div>'
        f'<span style="display:inline-block;background:{badge_color};color:white;'
        f'font-size:12px;font-weight:700;padding:2px 12px;border-radius:999px;margin-top:4px;">'
        f'{disease}</span></div></div>'
        + conf_bar(conf) +
        f'<div style="font-size:12px;color:var(--body-text-color);margin-bottom:10px;opacity:0.8;">'
        f'Model: <b>EfficientNet-B0</b> &nbsp;|&nbsp; Device: <b>{str(DEVICE).upper()}</b>'
        f' &nbsp;|&nbsp; Classes: <b>38</b></div>'
        f'<details><summary style="font-size:12px;color:#3b82f6;font-weight:600;cursor:pointer;">'
        f'Show Top-5 Predictions ▾</summary>'
        f'<div style="margin-top:8px;">{top5_table(all_probs)}</div></details></div>'
    )

    # ── Stage 3: Grad-CAM ─────────────────────────────────────────────
    heatmap     = generate_gradcam(pil_img, class_idx)
    overlay_arr = overlay_heatmap(pil_img, heatmap)
    cam_mean    = round(float(heatmap.mean()) * 100, 1)

    s3 = (
        '<div style="background:var(--background-fill-secondary);border-left:4px solid #a855f7;border-radius:8px;'
        'padding:14px 18px;margin:6px 0;">'
        '<div style="font-size:15px;font-weight:700;color:#a855f7;margin-bottom:8px;">'
        '✅ Stage 3 — Grad-CAM XAI Heatmap</div>'
        '<p style="font-size:13px;color:var(--body-text-color);margin:0 0 8px 0;">'
        'The heatmap highlights leaf regions that most influenced the model\'s decision. '
        '<b>Red/Yellow = high attention</b>, Blue = low attention.</p>'
        f'<div style="font-size:12px;color:var(--body-text-color);opacity:0.8;">'
        f'🎯 Layer: <b>EfficientNet features[-1]</b> &nbsp;|&nbsp; '
        f'🔥 Mean activation: <b>{cam_mean}%</b> &nbsp;|&nbsp; '
        f'See "Grad-CAM Heatmap" tab for the visual overlay.</div></div>'
    )

    # ── Stage 4: Severity ─────────────────────────────────────────────
    sev_level, affected_pct = estimate_severity(pil_img, is_healthy)
    sev_col  = SEV_COLOR.get(sev_level, "#6b7280")
    sev_icon = SEV_ICON.get(sev_level, "⚪")
    sev_bar  = (
        f'<div style="background:#e5e7eb;border-radius:8px;height:10px;overflow:hidden;margin:8px 0;">'
        f'<div style="width:{min(affected_pct,100)}%;height:100%;background:{sev_col};border-radius:8px;"></div>'
        f'</div>'
    ) if not is_healthy else ""
    healthy_note = (
        '<div style="font-size:13px;color:#22c55e;">Plant appears healthy — no affected area detected 🎉</div>'
        if is_healthy else
        f'<div style="font-size:13px;color:var(--body-text-color);opacity:0.8;">Affected leaf area: <b>{affected_pct}%</b></div>'
    )

    s4 = (
        f'<div style="background:var(--background-fill-secondary);border-left:4px solid {sev_col};border-radius:8px;'
        f'padding:14px 18px;margin:6px 0;">'
        f'<div style="font-size:15px;font-weight:700;color:{sev_col};margin-bottom:10px;">'
        f'✅ Stage 4 — Severity Estimation</div>'
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">'
        f'<div style="font-size:30px;">{sev_icon}</div>'
        f'<div><div style="font-size:24px;font-weight:900;color:{sev_col};">{sev_level}</div>'
        f'{healthy_note}</div></div>'
        f'{sev_bar}'
        f'<div style="font-size:12px;color:var(--body-text-color);opacity:0.8;">Method: HSV-based leaf segmentation '
        f'(green hue 30–90° = healthy tissue)</div></div>'
    )

    # ── Stage 5: Recommendation ───────────────────────────────────────
    rec       = get_recommendation(pred_class, sev_level, conf)
    rec_bdr   = "#22c55e" if is_healthy else "#ef4444"
    rec_title = "✅ Stage 5 — Treatment Recommendation" if is_healthy else "💊 Stage 5 — Treatment Recommendation"

    s5 = (
        f'<div style="background:var(--background-fill-secondary);border-left:4px solid {rec_bdr};border-radius:8px;'
        f'padding:14px 18px;margin:6px 0;">'
        f'<div style="font-size:15px;font-weight:700;color:{rec_bdr};margin-bottom:8px;">'
        f'{rec_title}</div>'
        f'<p style="font-size:13px;color:var(--body-text-color);line-height:1.8;margin:0 0 10px 0;">{rec}</p>'
        f'<div style="font-size:11px;color:var(--body-text-color);opacity:0.6;border-top:1px solid var(--border-color-primary);padding-top:8px;">'
        f'⚠️ For informational purposes only. Always consult a certified agronomist before applying any treatment.'
        f'</div></div>'
    )

    # ── Combined Result Panel ─────────────────────────────────────────
    summary_badge = (
        f'<span style="background:{badge_color};color:white;font-size:13px;font-weight:700;'
        f'padding:3px 14px;border-radius:999px;">{plant} — {disease}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="background:{sev_col};color:white;font-size:13px;font-weight:700;'
        f'padding:3px 14px;border-radius:999px;">{sev_level} Severity</span>'
        f'&nbsp;&nbsp;'
        f'<span style="background:#3b82f6;color:white;font-size:13px;font-weight:700;'
        f'padding:3px 14px;border-radius:999px;">{conf_pct}% Confidence</span>'
    )

    full_html = (
        '<div style="font-family:\'Segoe UI\',sans-serif;">'
        '<div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);color:white;'
        'border-radius:12px;padding:16px 20px;margin-bottom:12px;">'
        '<div style="font-size:17px;font-weight:800;margin-bottom:6px;">🌱 PlantXAI-Net — Analysis Complete</div>'
        '<div style="font-size:12px;opacity:0.85;margin-bottom:10px;">'
        'EfficientNet-B0 · Grad-CAM · Severity · Treatment Recommendation</div>'
        f'{summary_badge}</div>'
        f'{s1}{s2}{s3}{s4}{s5}'
        '</div>'
    )

    return preprocessed_arr, overlay_arr, s1, s2, s3 + s4 + s5, full_html


# ==============================================================================
# CSS
# ==============================================================================

CSS = """
#analyze-btn {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important; font-size: 15px !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important; padding: 14px !important;
}
#clear-btn {
    border-radius: 10px !important; border: 1px solid #e2e8f0 !important;
    font-size: 13px !important; background: #f1f5f9 !important; color: #475569 !important;
}
details summary { list-style: none; }
details summary::-webkit-details-marker { display: none; }
"""


# ==============================================================================
# Gradio UI
# ==============================================================================

def build_ui():
    with gr.Blocks(
        css=CSS,
        title="PlantXAI-Net",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="blue",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Header
        gr.HTML(
            '<div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);color:white;'
            'border-radius:12px;padding:22px 26px;margin-bottom:6px;">'
            '<h1 style="margin:0;font-size:26px;font-weight:800;letter-spacing:-0.5px;color:white;">'
            '🌱 PlantXAI-Net</h1>'
            '<p style="margin:6px 0 0 0;font-size:13px;opacity:0.85;color:white;">'
            'Explainable AI for Plant Disease Detection &amp; Recommendation &nbsp;·&nbsp; '
            'EfficientNet-B0 &nbsp;·&nbsp; Grad-CAM XAI &nbsp;·&nbsp; '
            '38 Disease Classes &nbsp;·&nbsp; 95.83% Test Accuracy'
            '</p></div>'
        )

        # Pipeline steps overview
        gr.HTML(
            '<div style="display:flex;gap:8px;margin:4px 0 8px 0;flex-wrap:wrap;">'
            + "".join(
                f'<div style="flex:1;min-width:120px;background:var(--background-fill-secondary);'
                f'border:1px solid var(--border-color-primary);'
                f'border-radius:8px;padding:10px;text-align:center;font-size:12px;color:var(--body-text-color);">'
                f'<div style="font-size:18px;">{icon}</div><b>{title}</b><br>{sub}</div>'
                for icon, title, sub in [
                    ("📤", "1. Upload", "Plant leaf image"),
                    ("🔬", "2. Classify", "EfficientNet-B0"),
                    ("🔥", "3. Grad-CAM", "XAI Heatmap"),
                    ("📊", "4. Severity", "Mild/Moderate/Severe"),
                    ("💊", "5. Recommend", "Treatment Guidance"),
                ]
            )
            + "</div>"
        )

        with gr.Row(equal_height=False):

            # ── Left Column ───────────────────────────────────────────
            with gr.Column(scale=4, min_width=300):

                input_image = gr.Image(
                    label="Upload Plant Leaf Image",
                    type="numpy",
                    height=300,
                    sources=["upload", "clipboard"],
                )

                analyze_btn = gr.Button(
                    "🔍  Analyze Plant Image",
                    elem_id="analyze-btn",
                    variant="primary",
                )

                gr.ClearButton(
                    components=[input_image],
                    value="🗑️  Clear Image",
                    elem_id="clear-btn",
                )

                gr.HTML(
                    '<div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;'
                    'padding:10px 14px;margin-top:8px;font-size:12px;color:#78350f !important;">'
                    '<b>💡 Tips for best results:</b>'
                    '<ul style="margin:4px 0 0 0;padding-left:16px;line-height:1.9;">'
                    '<li>Use a clear, well-lit photo</li>'
                    '<li>Focus on a single leaf</li>'
                    '<li>Avoid blurry or dark images</li>'
                    '<li>Supported: JPG, PNG, WEBP</li>'
                    '</ul></div>'
                )

                # Supported classes accordion
                with gr.Accordion("📋 Supported Plant & Disease Classes (38)", open=False):
                    plants_map: dict = {}
                    for c in CLASS_NAMES:
                        p, d = fmt_class(c)
                        plants_map.setdefault(p, []).append(d)
                    rows_html = "".join(
                        f'<tr style="border-bottom:1px solid #f1f5f9;">'
                        f'<td style="padding:5px 8px;font-weight:600;color:#1e293b !important;white-space:nowrap;">{p}</td>'
                        f'<td style="padding:5px 8px;color:#475569 !important;font-size:12px;">{", ".join(ds)}</td>'
                        f'</tr>'
                        for p, ds in sorted(plants_map.items())
                    )
                    gr.HTML(
                        '<div style="max-height:260px;overflow-y:auto;">'
                        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                        '<thead><tr style="background:#f8fafc;">'
                        '<th style="padding:6px 8px;text-align:left;color:#64748b !important;font-size:11px;">PLANT</th>'
                        '<th style="padding:6px 8px;text-align:left;color:#64748b !important;font-size:11px;">CONDITIONS</th>'
                        f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
                    )

            # ── Right Column ──────────────────────────────────────────
            with gr.Column(scale=6, min_width=380):

                with gr.Tabs():

                    with gr.Tab("📊 Full Pipeline Results"):
                        result_panel = gr.HTML(
                            value=(
                                '<div style="text-align:center;padding:60px 20px;color:#94a3b8 !important;'
                                'border:2px dashed #e2e8f0;border-radius:12px;">'
                                '<div style="font-size:52px;margin-bottom:12px;">🌿</div>'
                                '<div style="font-size:16px;font-weight:600;margin-bottom:6px;">'
                                'Upload a plant image to begin</div>'
                                '<div style="font-size:13px;">The full AI pipeline results will appear here</div>'
                                '</div>'
                            )
                        )

                    with gr.Tab("🖼️ Preprocessed Image"):
                        preprocessed_img = gr.Image(
                            label="224×224 Preprocessed Image",
                            type="numpy",
                            interactive=False,
                            height=300,
                        )

                    with gr.Tab("🔥 Grad-CAM Heatmap"):
                        gradcam_img = gr.Image(
                            label="Grad-CAM XAI Overlay",
                            type="numpy",
                            interactive=False,
                            height=300,
                        )
                        gr.HTML(
                            '<div style="font-size:12px;color:#64748b !important;padding:4px 0;">'
                            '🔴 Red/Yellow = high model attention &nbsp;|&nbsp; '
                            '🔵 Blue = low attention &nbsp;|&nbsp; '
                            'Method: Gradient-weighted Class Activation Maps (Grad-CAM)'
                            '</div>'
                        )

                    with gr.Tab("📋 Stage-by-Stage Details"):
                        gr.HTML('<div style="font-size:13px;font-weight:600;color:#64748b !important;margin-bottom:4px;">'
                                'Stage 1 — Preprocessing</div>')
                        stage1_out = gr.HTML()
                        gr.HTML('<div style="font-size:13px;font-weight:600;color:#64748b !important;margin:8px 0 4px 0;">'
                                'Stage 2 — Classification</div>')
                        stage2_out = gr.HTML()
                        gr.HTML('<div style="font-size:13px;font-weight:600;color:#64748b !important;margin:8px 0 4px 0;">'
                                'Stages 3–5 — XAI · Severity · Recommendation</div>')
                        stage_rest = gr.HTML()

        # Footer
        gr.HTML(
            '<div style="text-align:center;color:#94a3b8 !important;font-size:12px;'
            'border-top:1px solid #e2e8f0;padding-top:12px;margin-top:10px;">'
            'PlantXAI-Net &nbsp;·&nbsp; EfficientNet-B0 · Grad-CAM · XAI '
            '&nbsp;·&nbsp; Author: <b>Gursimran Singh</b> '
            '&nbsp;·&nbsp; Running locally at <b>http://127.0.0.1:7860</b>'
            '</div>'
        )

        # Wire button
        analyze_btn.click(
            fn=run_pipeline,
            inputs=[input_image],
            outputs=[preprocessed_img, gradcam_img, stage1_out, stage2_out, stage_rest, result_panel],
        )

    return demo


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,     # Opens browser automatically
        show_error=True,
    )
