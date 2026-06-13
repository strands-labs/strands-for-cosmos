"""Autonomous driving analysis with chain-of-thought reasoning.

Uses sample.mp4 as a dashcam video for safety analysis.

📓 Learn-first notebook: ../notebooks/03_driving_analysis.ipynb
   ("Chain-of-Thought for Driving Safety" — same concepts, explained step by step with diagrams.)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent
from strands_cosmos import CosmosVisionModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sample_video = os.environ.get("SAMPLE_VIDEO", os.path.join(project_root, "sample.mp4"))

print("=== 03: Driving Analysis (CoT) ===")
t0 = time.time()

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    fps=4,
    params={"max_tokens": 2048, "temperature": 0.6},
)
agent = Agent(model=model)

result = agent(f"<video>{sample_video}</video> Identify safety hazards and recommend actions.")
print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
