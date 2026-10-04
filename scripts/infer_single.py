"""Run the trained U-Net on one RGB image and extract measurable polyp features.
Usage: python infer_single.py <image_path> <out_dir>
Writes mask.png, overlay.png, prob_heatmap.png and features.json to out_dir.
The logic lives in polyp_inference.py (project root), shared with the MCP server."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polyp_inference import run_inference

print(json.dumps(run_inference(sys.argv[1], sys.argv[2]), indent=2))
