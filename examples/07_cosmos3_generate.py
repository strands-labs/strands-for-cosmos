"""Cosmos 3 Generator — text→image / text→video / video+sound (in-proc Diffusers, no NIM).

Prerequisites:
    just c3-setup-gen     # one-time: diffusers(main) + cosmos_guardrail (cu130)

Note: On a single GPU, STOP the reasoner server before running the generator
(both load a 16B model; they won't fit together on ~46GB).

📓 Learn-first notebook: ../notebooks/07_cosmos3_generate.ipynb
   ("Cosmos 3: The Generator" — same concepts, explained step by step with diagrams.)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands_cosmos import Cosmos3GeneratorModel

m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")

# 1) Text → image (fast)
print("=== text2image ===")
t0 = time.time()
print(m.generate(mode="text2image",
                 prompt="A mobile robot in a warehouse aisle, photorealistic.",
                 out_path="/tmp/cosmos3_image.png",
                 resolution="480", num_inference_steps=20))
print(f"[{time.time() - t0:.1f}s]")

# 2) Text → video
print("=== text2video ===")
t0 = time.time()
print(m.generate(mode="text2video",
                 prompt="A robot navigates a warehouse and stops at a shelf.",
                 out_path="/tmp/cosmos3_video.mp4",
                 resolution="480", num_frames=49, fps=16, num_inference_steps=25))
print(f"[{time.time() - t0:.1f}s]")

# 3) Text → video WITH SOUND (stereo AAC 48kHz)
print("=== text2video-with-sound ===")
t0 = time.time()
print(m.generate(mode="text2video-with-sound",
                 prompt="A robot arm pours water into a glass, with pouring sounds.",
                 out_path="/tmp/cosmos3_sound.mp4",
                 resolution="480", num_frames=49, fps=16, num_inference_steps=25,
                 enable_sound=True))
print(f"[{time.time() - t0:.1f}s]")
print("=== done ===")
