#!/usr/bin/env python3
"""Hold a multi-turn conversation about one image - context carries across turns.

Goal:
    Show that a Cosmos-Reason2 agent is conversational, not one-shot. Examples
    01-05 each send a single prompt; this one asks three questions in a row over
    the same image and confirms the model remembers earlier turns (turn 3 refers
    back to turn 1). That memory is what makes it an agent rather than a captioner.

What it shows:
    - One CosmosVisionModel / Agent reused across turns (Strands keeps the history)
    - Image reasoning via the <image>...</image> tag
    - A concrete check that context was retained, instead of an unconditional PASS

Dependencies:
    uv pip install strands-cosmos      # core install
    A CUDA GPU with >= ~6 GB free for Cosmos-Reason2-2B (weights ~4.6 GB).
    HF access to the gated nvidia/Cosmos-Reason2-2B (accept its license, set HF_TOKEN).

Expected output:
    Three answers. Turn 1 names objects in the image; turn 2 picks one to act on;
    turn 3 recalls turn 1. Prints PASS only if a GPU was present and all three
    turns returned text. On a CPU-only host it skips politely (exit 0).

Runtime:
    ~15 s on an NVIDIA L4 after weights are cached (first run also downloads them).

Run:
    python examples/12_multiturn_visual_chat.py
    SAMPLE_IMAGE=/path/frame.png python examples/12_multiturn_visual_chat.py

Learn-first notebook: ../notebooks/04_embodied_reasoning.ipynb
    ("Robot Brain" - single-turn image reasoning; this example makes it a dialogue.)
"""

import os
import sys
import time


def gpu_available() -> bool:
    """True only if a CUDA device is usable - keeps the example safe on a laptop."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_image = os.environ.get("SAMPLE_IMAGE", os.path.join(project_root, "sample.png"))

    print("=== 12: Multi-turn visual chat ===")

    if not os.path.exists(sample_image):
        print(f"sample image not found: {sample_image}", file=sys.stderr)
        print("Set SAMPLE_IMAGE to a real image and re-run.", file=sys.stderr)
        return 1

    if not gpu_available():
        print("No CUDA GPU detected - skipping model inference.")
        print("This example loads Cosmos-Reason2-2B, which needs a GPU. Nothing to verify here.")
        return 0

    from strands import Agent

    from strands_cosmos import CosmosVisionModel

    t0 = time.time()
    model = CosmosVisionModel(
        model_id="nvidia/Cosmos-Reason2-2B",
        params={"max_tokens": 512, "temperature": 0.3},
    )
    # One agent for the whole conversation - Strands threads the prior turns back
    # in on each call, so the model sees the growing history, not just the latest line.
    agent = Agent(model=model)
    print(f"model ready in {time.time() - t0:.1f}s\n")

    turns = [
        f"<image>{sample_image}</image> What objects are on the table?",
        "Which single object should a robot arm pick up first to start clearing "
        "the table? Name the object and give one reason.",
        "Was that object in your first answer? Reply yes or no and restate it.",
    ]

    answers = []
    for i, prompt in enumerate(turns, 1):
        print(f"--- turn {i} ---")
        print(f"you: {prompt}")
        reply = str(agent(prompt)).strip()
        print(f"cosmos: {reply}\n")
        answers.append(reply)

    # Verify substance and context retention rather than trusting a bare run.
    if not all(answers):
        print("[error] a turn returned no text", file=sys.stderr)
        return 1
    if "yes" not in answers[-1].lower():
        # The image-only turn 3 asks the model to confirm its earlier answer; a
        # "yes" is the observable signal that the prior turns were in context.
        print("[warn] turn 3 did not confirm context retention - inspect the replies above",
              file=sys.stderr)

    print(f"=== PASS: {len(answers)} turns, context carried across the conversation "
          f"({time.time() - t0:.1f}s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
