"""Robot embodied reasoning — next action prediction from an image.

Uses sample.png as the robot's camera view.

📓 Learn-first notebook: ../notebooks/04_embodied_reasoning.ipynb
   ("Robot Brain: Next-Action Prediction" — same concepts, explained step by step with diagrams.)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent
from strands_cosmos import CosmosVisionModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sample_image = os.environ.get("SAMPLE_IMAGE", os.path.join(project_root, "sample.png"))

print("=== 04: Embodied Reasoning ===")
t0 = time.time()

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    params={"max_tokens": 2048, "temperature": 0.6},
)
agent = Agent(model=model)

result = agent(f"<image>{sample_image}</image> What can be the next immediate action?")
print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
