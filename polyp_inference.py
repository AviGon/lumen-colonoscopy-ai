"""Run the trained U-Net on one RGB image and extract measurable polyp features.

Shared by scripts/infer_single.py and the MCP server's segment_polyp tool.
Writes mask.png, overlay.png, prob_heatmap.png and features.json to out_dir."""
import json, os
from functools import lru_cache
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "polyp_unet_r34.pth")


@lru_cache(maxsize=1)
def load_model(ckpt=CKPT):
    """Load the checkpoint once per process. Returns (model, meta, device)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    meta = ck["meta"]
    model = smp.Unet(meta["encoder"], encoder_weights=None, in_channels=3, classes=1).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, meta, dev


def run_inference(img_path, out, threshold=None, ckpt=CKPT):
    """Segment one image, write artifacts to out and return the features dict.
    threshold=None uses the threshold stored in the checkpoint (0.5)."""
    os.makedirs(out, exist_ok=True)
    model, meta, dev = load_model(ckpt)
    mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
    thr = meta["threshold"] if threshold is None else threshold

    img = np.array(Image.open(img_path).convert("RGB").resize((352, 352), Image.BILINEAR))
    x = torch.from_numpy(((img.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1))[None].to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = prob > thr

    H, W = mask.shape
    lab, n = ndimage.label(mask)
    regions = []
    for k in range(1, n + 1):
        r = lab == k
        area = int(r.sum())
        if area < 50:  # ignore specks
            continue
        ys, xs = np.nonzero(r)
        perim = int((r & ~ndimage.binary_erosion(r)).sum())
        # convex hull area for solidity (border irregularity proxy)
        pts = np.stack([xs, ys], 1)
        hull_area = ConvexHull(pts).volume if len(pts) > 3 else area
        cy, cx = ys.mean() / H, xs.mean() / W
        vert = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
        horz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        rgb = img[r].mean(0); bg = img[~mask].mean(0) if (~mask).any() else rgb
        regions.append({
            "area_px": area,
            "area_fraction_of_frame": round(area / (H * W), 4),
            "bbox_xywh": [int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)],
            "max_diameter_px": int(max(np.ptp(xs), np.ptp(ys)) + 1),
            "aspect_ratio": round(float(max(np.ptp(xs), np.ptp(ys)) + 1) / float(min(np.ptp(xs), np.ptp(ys)) + 1), 2),
            "circularity": round(4 * np.pi * area / max(perim, 1) ** 2, 3),
            "solidity": round(area / max(hull_area, 1), 3),
            "position_in_frame": f"{vert} {horz}",
            "touches_frame_edge": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1),
            "mean_prob_inside": round(float(prob[r].mean()), 3),
            "mean_rgb_polyp": [round(float(v), 1) for v in rgb],
            "mean_rgb_background": [round(float(v), 1) for v in bg],
        })
    regions.sort(key=lambda d: -d["area_px"])

    # Approximate size range in mm. No reference object, so we assume a wide angle
    # colonoscope (140 deg diagonal FOV, equidistant fisheye) and a 5 to 20 mm working
    # distance, project the mask onto a plane facing the scope, and take its max diameter.
    FOV_DEG, DISTS_MM, TYPICAL_MM = 140.0, (5.0, 20.0), 10.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - (W - 1) / 2, yy - (H - 1) / 2
    rr = np.hypot(dx, dy)
    theta = np.deg2rad(rr / rr.max() * FOV_DEG / 2)
    scale = np.where(rr > 0, np.tan(theta) / np.maximum(rr, 1e-9), 0)
    px, py = dx * scale, dy * scale  # plane coords at distance 1
    # area each pixel covers on that plane (Jacobian of the mapping), so area_mm2 = sum * dist^2
    gpx_y, gpx_x = np.gradient(px)
    gpy_y, gpy_x = np.gradient(py)
    pix_area = np.abs(gpx_x * gpy_y - gpx_y * gpy_x)

    def size_category(mm):
        return "<6 mm" if mm < 6 else ("6-9 mm" if mm < 10 else ">=10 mm")

    kept = [k for k in range(1, n + 1) if (lab == k).sum() >= 50]
    for reg, k in zip(regions, sorted(kept, key=lambda k: -(lab == k).sum())):
        r = (lab == k) & ~ndimage.binary_erosion(lab == k)  # border pixels are enough
        pts = np.stack([px[r], py[r]], 1)
        d1 = float(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max())
        lo, typ, hi = (d1 * d for d in (DISTS_MM[0], TYPICAL_MM, DISTS_MM[1]))
        order = ["<6 mm", "6-9 mm", ">=10 mm"]
        cats = order[order.index(size_category(lo)):order.index(size_category(hi)) + 1]
        a1 = float(pix_area[lab == k].sum())
        reg["area_estimate_mm2"] = {
            "low": round(a1 * DISTS_MM[0] ** 2, 1), "typical": round(a1 * TYPICAL_MM ** 2, 1),
            "high": round(a1 * DISTS_MM[1] ** 2, 1),
        }
        reg["size_estimate_mm"] = {
            "low": round(lo, 1), "typical": round(typ, 1), "high": round(hi, 1),
            "likely_category": size_category(typ),
            "categories_in_range": cats,
            "assumptions": f"{FOV_DEG:.0f} deg diagonal FOV fisheye, working distance "
                           f"{DISTS_MM[0]:.0f}-{DISTS_MM[1]:.0f} mm (typical {TYPICAL_MM:.0f} mm), "
                           "lesion facing the scope. Estimate only, physician to confirm.",
        }

    # Confidence: share of uncertain pixels (0.3..0.7) along the predicted region
    uncertain = ((prob > 0.3) & (prob < 0.7)).sum()
    feat = {
        "image": os.path.abspath(img_path),
        "model": os.path.basename(ckpt),
        "threshold": thr,
        "polyp_detected": len(regions) > 0,
        "num_regions": len(regions),
        "regions": regions,
        "uncertain_pixel_fraction_of_mask": round(float(uncertain / max(mask.sum(), 1)), 3),
        "brightness_mean": round(float(img.mean()), 1),
        "specular_highlight_fraction": round(float((img.min(2) > 235).mean()), 4),
    }
    json.dump(feat, open(os.path.join(out, "features.json"), "w"), indent=2)

    Image.fromarray(img).save(os.path.join(out, "input.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out, "mask.png"))
    o = img.astype(np.float32).copy()
    o[mask] = o[mask] * 0.55 + np.array([255, 255, 0]) * 0.45
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    o[edge] = [255, 0, 0]
    Image.fromarray(np.concatenate([img, o.astype(np.uint8)], 1)).save(os.path.join(out, "overlay.png"))
    heat = (prob * 255).astype(np.uint8)
    Image.fromarray(np.stack([heat, np.zeros_like(heat), 255 - heat], 2)).save(os.path.join(out, "prob_heatmap.png"))
    return feat
