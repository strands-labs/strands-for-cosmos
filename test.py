# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "strands-agents[openai]",
#     "strands-cosmos",
# ]
# ///
import os
import sys
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel
from strands_cosmos import cosmos3_reason, cosmos3_caption, cosmos3_temporal, video_probe, image_read, cosmos_sysinfo

base_url = os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
max_tokens = int(os.environ.get("COSMOS_MAX_TOKENS", "512"))
model = Cosmos3ReasonerModel(base_url=base_url, max_tokens=max_tokens)
tools = [cosmos3_reason, cosmos3_caption, cosmos3_temporal, video_probe, image_read, cosmos_sysinfo]
agent = Agent(
    model=model,  # Either use the Cosmos3 Reasoner as model
    # tools=tools # Or use as set of tools in your agent
)

one_shot = " ".join(sys.argv[1:]).strip()
if one_shot:
    agent(one_shot)
    sys.exit(0)

print(f"strands-cosmos @ {base_url} | type 'exit' to quit")
while True:
    try:
        query = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if query in ("exit", "quit", "q"):
        break
    if not query:
        continue
    agent(query)
