"""Cosmos 3 Reasoner — video understanding via local vLLM (no NIM).

Prerequisites:
    just c3-setup-reason      # one-time: vllm + vllm-cosmos3 (cu130)
    just c3-serve-reason      # start Cosmos3-Nano server on :8000

Then run this example. Caption + temporal + embodied reasoning on a video.

📓 Learn-first notebook: ../notebooks/06_cosmos3_understand.ipynb
   ("Cosmos 3: The Reasoner" — same concepts, explained step by step with diagrams.)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

video = os.environ.get("SAMPLE_VIDEO", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample.mp4"))

print("=== Cosmos 3 Reasoner ===")
model = Cosmos3ReasonerModel(base_url="http://localhost:8000/v1", max_tokens=512)
agent = Agent(model=model)

t0 = time.time()
agent(f"Caption in detail: <video>{video}</video>")
print(f"\n[caption {time.time() - t0:.1f}s]")

# Reasoning mode (explicit <think>)
model.update_config(reasoning=True)
agent(f"What is the most likely next action? <video>{video}</video>")
print("=== done ===")
