"""Cosmos 3 showcase — full omnimodal loop: REASON about a video, then GENERATE
similar videos (incl. one with synchronized audio) from the reasoner's own prompt.

Prerequisites (single ~46GB GPU is fine — we run the two phases sequentially):
    just c3-setup-reason && just c3-serve-reason    # Phase A: vLLM reasoner on :8000
    just c3-setup-gen                               # Phase B: Diffusers generator

Usage:
    SHOWCASE_VIDEO=/path/to/input.mp4 python examples/09_cosmos3_showcase.py

Outputs land in demo/cosmos3_showcase/. See that folder's README.md for results.

📓 Learn-first notebook: ../notebooks/06_cosmos3_understand.ipynb
   ("Cosmos 3: Reason → Generate (06 + 07)" — same concepts, explained step by step with diagrams.)
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VIDEO = os.environ.get("SHOWCASE_VIDEO", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample.mp4"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "demo", "cosmos3_showcase")
os.makedirs(OUT, exist_ok=True)

# ── Phase A: REASON (needs the vLLM reasoner server running) ────────────────
def reason():
    from strands_cosmos import Cosmos3ReasonerModel
    from strands import Agent

    model = Cosmos3ReasonerModel(base_url="http://localhost:8000/v1", max_tokens=300)
    agent = Agent(model=model)
    print("=== Phase A: Reasoning ===")
    caption = str(agent(f"Caption this video in detail: <video>{VIDEO}</video>"))
    prompt = str(agent(
        f"In one vivid sentence (<40 words), describe this scene as a prompt to "
        f"generate a similar video: <video>{VIDEO}</video>"))
    json.dump({"caption": caption, "gen_prompt": prompt},
              open(f"{OUT}/reasoning_output.json", "w"), indent=2)
    return prompt

# ── Phase B: GENERATE (stop the reasoner first to free the GPU) ─────────────
def generate(prompt):
    from strands_cosmos import Cosmos3GeneratorModel
    m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")
    print("=== Phase B: Generation ===")
    for mode, out, kw in [
        ("text2video", "01_text2video.mp4", {}),
        ("text2video-with-sound", "02_text2video_sound.mp4", {"enable_sound": True}),
    ]:
        t0 = time.time()
        r = m.generate(mode=mode, prompt=prompt, out_path=f"{OUT}/{out}",
                       resolution="480", num_frames=49, fps=16,
                       num_inference_steps=30, seed=0, **kw)
        print(f"  {mode} -> {r['out_path']} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    prompt = reason()
    print(f"\nGeneration prompt (from reasoner): {prompt}\n")
    print("Stop the vLLM reasoner now to free the GPU, then run generate().")
    # generate(prompt)   # uncomment after stopping the reasoner server
