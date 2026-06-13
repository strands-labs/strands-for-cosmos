"""Using Cosmos vision as a tool within another agent.

Demonstrates cosmos_vision_invoke as a Strands tool — callable from
any agent (Bedrock, OpenAI, Ollama, etc.) for video/image analysis.

📓 Learn-first notebook: ../notebooks/05_tool_usage.ipynb
   ("Cosmos as Tools: Building Blocks" — same concepts, explained step by step with diagrams.)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands_cosmos import cosmos_vision_invoke

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sample_video = os.environ.get("SAMPLE_VIDEO", os.path.join(project_root, "sample.mp4"))

print("=== 05: Tool Usage (direct invoke) ===")
t0 = time.time()

# Direct tool invocation (no outer agent needed for testing)
result = cosmos_vision_invoke(
    prompt="Describe the scene briefly.",
    video_path=sample_video,
    max_tokens=512,
)

print(f"\nStatus: {result['status']}")
if result["status"] == "success":
    print(f"Response: {result['content'][0]['text'][:200]}...")
print(f"\nTime: {time.time() - t0:.1f}s")
print("=== PASS ===")
