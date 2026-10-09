"""Cosmos3-Edge (4B) on one GPU — reason, generate and act with no vLLM server and no Cosmos Framework.

Measured on a Jetson AGX Thor (arm64, sm_110): reasoner 4.6 GiB, generator 7.6 GiB, action peak 12.1 GiB.
See docs/guide/cosmos3-edge.md for the full table.

Setup:
    just c3-setup-edge && just c3-edge-doctor
    export COSMOS_WORKSPACE=$PWD            # media paths are confined to this directory
    export HF_HUB_OFFLINE=1                 # once nvidia/Cosmos3-Edge is in the HF cache

Run:
    .venv-c3-edge/bin/python examples/13_cosmos3_edge.py [--generate] [--action]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("COSMOS_WORKSPACE", os.getcwd())


def reason() -> None:
    from strands import Agent
    from strands_cosmos import Cosmos3EdgeHFModel

    # Lazy: weights load on the first call. <think> is streamed as reasoningContent, answer as text.
    agent = Agent(model=Cosmos3EdgeHFModel(params={"max_tokens": 512}), callback_handler=None)
    for prompt in (
        "Caption the image in detail: <image>ws_red.png</image>",
        "Caption the video in detail: <video>sample.mp4</video>",
        "In one sentence: what kind of camera recorded that video?",   # multi-turn: history carries
    ):
        t0 = time.time()
        reply = agent(prompt)
        print(f"\n> {prompt}\n{str(reply).strip()}\n[{time.time() - t0:.1f}s] {reply.metrics.accumulated_usage}")


def generate() -> None:
    from strands_cosmos import Cosmos3GeneratorModel  # same Diffusers pipeline as Nano

    m = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Edge")
    print(m.get_config())
    print("Edge has no sound tower: cosmos3_text2video_sound() returns an error naming Cosmos3-Nano instead.")


def action() -> None:
    from strands_cosmos import cosmos3_action_edge
    from strands_cosmos.cosmos3_models import trained_embodiments, validate_embodiment

    print("trained embodiments:", ", ".join(trained_embodiments()))
    for bad in ("so101", "pusht"):
        try:
            validate_embodiment(bad)
        except ValueError as e:
            print(f"{bad}: refused -> {str(e)[:90]}...")
    res = cosmos3_action_edge(image="sample.png", prompt="pick up the object on the table",
                              mode="policy", embodiment="droid_lerobot", chunk_size=16, steps=8,
                              out="/tmp/c3_edge_action")
    print(res["content"][0]["text"])


if __name__ == "__main__":
    reason()
    if "--generate" in sys.argv:
        generate()
    if "--action" in sys.argv:
        action()
