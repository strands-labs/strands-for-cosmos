"""Basic text inference with Cosmos-Reason2.

Tests text-only physics reasoning — no video/image needed.

📓 Learn-first notebook: ../notebooks/01_basic_text.ipynb
   ("Your First Agent: Pure Text Reasoning" — same concepts, explained step by step with diagrams.)
"""

import os
import sys
import time

# Allow running from examples/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent
from strands_cosmos import CosmosModel

print("=== 01: Basic Text Inference ===")
t0 = time.time()

model = CosmosModel(model_id="nvidia/Cosmos-Reason2-2B")
agent = Agent(model=model)

result = agent("Explain the physics of a ball rolling down a ramp. Be concise.")
print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
