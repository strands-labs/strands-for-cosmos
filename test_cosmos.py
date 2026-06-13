"""Test strands-cosmos on Jetson AGX Thor."""
import time
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n=== Loading CosmosVisionModel ===")
t0 = time.time()

from strands_cosmos import CosmosVisionModel

model = CosmosVisionModel(
    model_id="nvidia/Cosmos-Reason2-2B",
    reasoning=True,
    fps=4,
    params={"max_tokens": 2048, "temperature": 0.6},
)
print(f"Model loaded in {time.time() - t0:.1f}s")

print("\n=== Test 1: Video Caption ===")
t0 = time.time()
from strands import Agent
agent = Agent(model=model, system_prompt="You are a helpful video analyzer.")

result = agent("Caption in detail: <video>/home/cagatay/strands-cosmos/sample.mp4</video>")
print(f"\nTime: {time.time() - t0:.1f}s")

print("\n=== Test 2: Driving Analysis with CoT ===")
t0 = time.time()
result = agent("<video>/home/cagatay/strands-cosmos/sample.mp4</video> What are the potential safety hazards in this scene? Think step by step.")
print(f"\nTime: {time.time() - t0:.1f}s")

print("\n=== Test 3: Physics Reasoning (text only) ===")
t0 = time.time()
result = agent("What would happen if a heavy box slides off the back of a moving truck on a highway? Think about the physics.")
print(f"\nTime: {time.time() - t0:.1f}s")

print("\n=== ALL TESTS COMPLETE ===")
