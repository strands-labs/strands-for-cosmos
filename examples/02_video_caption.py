"""Video captioning with Cosmos-Reason2.

Captions a sample video using the vision model.
Uses sample.mp4 from the project root.

📓 Learn-first notebook: ../notebooks/02_video_caption.ipynb
   ("Giving the Agent Eyes: Video Captioning" — same concepts, explained step by step with diagrams.)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent
from strands_cosmos import CosmosVisionModel

# Resolve sample video path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sample_video = os.path.join(project_root, "sample.mp4")

if not os.path.exists(sample_video):
    print(f"⚠️  sample.mp4 not found at {sample_video}")
    print("   Download a sample video or set SAMPLE_VIDEO env var")
    sample_video = os.environ.get("SAMPLE_VIDEO", sample_video)

print("=== 02: Video Caption ===")
t0 = time.time()

model = CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

result = agent(f"Caption in detail: <video>{sample_video}</video>")
print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
