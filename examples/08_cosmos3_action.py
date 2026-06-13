"""Cosmos 3 Action / World-Model — forward dynamics (Cosmos Framework, no NIM).

Prerequisites:
    just c3-setup-framework   # one-time: clone cosmos-framework + uv sync (cu130-train)

Forward dynamics rolls out a future-state video from a start image + an action
trajectory. Inverse dynamics predicts the action trajectory from a video.
Policy predicts an action chunk + rollout video from an image + instruction.

Input is a JSONL spec, one line per run. This example builds an autonomous-vehicle
forward-dynamics spec from the sample assets shipped in the cosmos repo.

📓 Learn-first notebook: ../notebooks/07_cosmos3_generate.ipynb
   ("Cosmos 3: Generator (see the mode menu)" — same concepts, explained step by step with diagrams.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands_cosmos import cosmos3_forward_dynamics

# Adjust to your cosmos repo checkout (clone of github.com/NVIDIA/cosmos).
COSMOS = os.environ.get("C3_REPO", os.path.expanduser("~/nvidia-test/cosmos"))
ASSETS = f"{COSMOS}/cookbooks/cosmos3/generator/action/assets"

spec = {
    "action_chunk_size": 60,
    "action_path": f"{ASSETS}/actions/av_traj_forward.json",
    "domain_name": "av",
    "fps": 10,
    "image_size": 480,
    "view_point": "ego_view",
    "model_mode": "forward_dynamics",
    "name": "av_forward",
    "prompt": "You are an autonomous vehicle planning system.",
    "seed": 0,
    "vision_path": f"{ASSETS}/images/av_0.jpg",
}

os.makedirs("/tmp/c3_action_in", exist_ok=True)
spec_path = "/tmp/c3_action_in/fd_av.jsonl"
with open(spec_path, "w") as f:
    f.write(json.dumps(spec) + "\n")

print("=== Cosmos 3 Forward Dynamics ===")
result = cosmos3_forward_dynamics(input_jsonl=spec_path, out="/tmp/c3_action_out")
print(result)
print("Output video: /tmp/c3_action_out/av_forward/vision.mp4")
