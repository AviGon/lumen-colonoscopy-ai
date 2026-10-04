"""Classify one image as benign or malignant with the EfficientNet-B0 and write artifacts.

Classes: benign, malignant (trained on GastroVision + ERCPMP). Used by the web app.
Writes input.png, gradcam.png (input | Grad-CAM overlay) and pathology.json to out_dir."""
import json, os
from functools import lru_cache
import numpy as np
import torch
import timm
from PIL import Image

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "malignancy", "malignancy_effb0.pth")
LABELS = {
    "benign": "Benign (looks like a polyp)",
    "malignant": "Malignant (looks like colorectal cancer)",
}


@lru_cache(maxsize=2)
def load_model(ckpt=CKPT):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    meta = ck["meta"]
    model = timm.create_model(meta["arch"], pretrained=False, num_classes=len(meta["classes"])).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, meta, dev


def run_pathology(img_path, out, ckpt=CKPT):
    os.makedirs(out, exist_ok=True)
    model, meta, dev = load_model(ckpt)
    h, w = meta["input_hw"]
    mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
    img = np.array(Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR))
    x = torch.from_numpy(((img.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))[None].to(dev)

    # Grad-CAM on the last conv feature map
    feats = {}
    hook = model.conv_head.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    logits = model(x)
    hook.remove()
    probs = torch.softmax(logits.float(), 1)[0]
    top = int(probs.argmax())
    act = feats["a"]
    grad = torch.autograd.grad(logits[0, top], act)[0]
    cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1))[0]
    cam = torch.nn.functional.interpolate(cam[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()

    classes = meta["classes"]
    p = probs.detach().cpu().numpy()
    ranked = sorted(range(len(classes)), key=lambda i: -p[i])
    result = {
        "image": os.path.abspath(img_path),
        "model": os.path.basename(ckpt),
        "dataset": meta.get("dataset", "ERCPMP-v5"),
        "test_metrics_file": os.path.join(os.path.dirname(os.path.abspath(ckpt)), "test_eval", "test_metrics.txt"),
        "classes": classes,
        "probabilities": {c: round(float(p[i]), 4) for i, c in enumerate(classes)},
        "predicted_class": classes[top],
        "predicted_label": LABELS.get(classes[top], classes[top]),
        "top_probability": round(float(p[top]), 4),
        "margin_over_second": round(float(p[ranked[0]] - p[ranked[1]]), 4),
        "gradcam_focus_fraction": round(float((cam > 0.5).mean()), 4),
        "brightness_mean": round(float(img.mean()), 1),
        "specular_highlight_fraction": round(float((img.min(2) > 235).mean()), 4),
        "notice": "Experimental research model; not clinically validated. Histopathology is the reference standard.",
    }
    json.dump(result, open(os.path.join(out, "pathology.json"), "w"), indent=2)

    Image.fromarray(img).save(os.path.join(out, "input.png"))
    heat = np.stack([cam, np.clip(1.5 - np.abs(cam * 4 - 2), 0, 1), 1 - cam], 2)  # blue -> green -> red
    o = (img.astype(np.float32) * 0.55 + heat * 255 * 0.45).astype(np.uint8)
    Image.fromarray(np.concatenate([img, o], 1)).save(os.path.join(out, "gradcam.png"))
    return result
