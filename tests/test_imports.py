"""Test that all imports work."""


def test_cosmos_model_import():
    from strands_cosmos import CosmosModel
    assert CosmosModel is not None


def test_cosmos_vision_model_import():
    from strands_cosmos import CosmosVisionModel
    assert CosmosVisionModel is not None


def test_tools_import():
    from strands_cosmos import cosmos_invoke, cosmos_vision_invoke
    assert cosmos_invoke is not None
    assert cosmos_vision_invoke is not None


def test_task_prompts():
    from strands_cosmos.cosmos_vision_model import TASK_PROMPTS
    assert "caption" in TASK_PROMPTS
    assert "driving" in TASK_PROMPTS
    assert "embodied_reasoning" in TASK_PROMPTS
    assert "robot_cot" in TASK_PROMPTS
