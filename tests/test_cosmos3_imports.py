"""Phase 0 gate: Cosmos 3 providers and tools import without GPU/model load."""


def test_cosmos3_providers_import():
    from strands_cosmos import Cosmos3ReasonerModel, Cosmos3GeneratorModel

    r = Cosmos3ReasonerModel()
    assert r.get_config()["model_id"] == "nvidia/Cosmos3-Nano"
    g = Cosmos3GeneratorModel()
    assert g.get_config()["model_id"] == "nvidia/Cosmos3-Nano"


def test_cosmos3_reasoner_media_parsing():
    from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel

    m = Cosmos3ReasonerModel()
    msgs = [{"role": "user", "content": [
        {"text": "Caption: <video>https://example.com/y.mp4</video> and <image>https://example.com/z.jpg</image> please"}
    ]}]
    oai = m._extract_media_to_openai(msgs)
    assert len(oai) == 1
    parts = oai[0]["content"]
    types = [p["type"] for p in parts]
    assert "video_url" in types and "image_url" in types and "text" in types


def test_cosmos3_tools_import():
    from strands_cosmos import (
        cosmos3_reason, cosmos3_caption, cosmos3_temporal, cosmos3_embodied,
        cosmos3_ground, cosmos3_plausibility, cosmos3_situation, cosmos3_action_cot,
        cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
        cosmos3_text2video_sound, cosmos3_forward_dynamics, cosmos3_inverse_dynamics,
        cosmos3_policy, cosmos3_serve,
    )
    # All decorated tools expose a tool spec
    for t in [cosmos3_reason, cosmos3_caption, cosmos3_serve]:
        assert hasattr(t, "tool_spec") or callable(t)


def test_cosmos3_generator_modes_and_resolution():
    from strands_cosmos.cosmos3_generator_model import (
        Cosmos3GeneratorModel, RES_TIERS, GEN_DEFAULTS,
    )
    m = Cosmos3GeneratorModel()
    cfg = m.get_config()
    assert cfg["model_id"] == "nvidia/Cosmos3-Nano"
    assert cfg["guardrails"] is True
    # resolution tiers present
    assert RES_TIERS["256"] == (320, 192)
    assert RES_TIERS["480"] == (832, 480)
    assert RES_TIERS["720"] == (1280, 720)
    # has the audio mux helper
    assert hasattr(m, "_mux_audio")
    assert GEN_DEFAULTS["fps"] == 24


def test_cosmos3_reasoner_task_prompts():
    from strands_cosmos.cosmos3_reasoner_model import TASK_PROMPTS, SAMPLING_THINK, SAMPLING_NO_THINK
    for k in ["caption", "temporal", "embodied", "plausibility", "situation",
              "grounding", "describe", "action_cot", "driving"]:
        assert k in TASK_PROMPTS
    assert SAMPLING_THINK["temperature"] == 0.6
    assert SAMPLING_NO_THINK["temperature"] == 0.7


def test_cosmos3_training_tools_import():
    from strands_cosmos import (
        cosmos3_train, cosmos3_train_recipes, cosmos3_train_show,
        cosmos3_train_convert, cosmos3_train_convert_vlm,
        cosmos3_train_prep_dataset, cosmos3_train_export,
    )
    for t in [cosmos3_train, cosmos3_train_convert, cosmos3_train_export]:
        assert hasattr(t, "tool_spec") or callable(t)


def test_cosmos3_training_tools_in_all():
    import strands_cosmos as sc
    for n in ["cosmos3_train", "cosmos3_train_convert", "cosmos3_train_convert_vlm",
              "cosmos3_train_prep_dataset", "cosmos3_train_show",
              "cosmos3_train_recipes", "cosmos3_train_export"]:
        assert n in sc.__all__, f"{n} missing from __all__"


def test_justfile_is_locatable():
    """Regression: justfile-backed tools need _find_justfile to resolve a real file.

    Bundled justfile ships at strands_cosmos/justfile (build-time copy) so pip
    installs work; in a repo checkout the root justfile is found instead.
    """
    from strands_cosmos.tools._common import _find_justfile
    jf = _find_justfile()
    assert jf is not None, "justfile not found — justfile-backed tools would fail"
    assert jf.is_file()


def test_cosmos3_video2video_tool():
    from strands_cosmos import cosmos3_video2video
    assert hasattr(cosmos3_video2video, "tool_spec") or callable(cosmos3_video2video)
    import strands_cosmos as sc
    assert "cosmos3_video2video" in sc.__all__


def test_cosmos3_video2video_conditioning_params():
    import inspect
    from strands_cosmos import cosmos3_video2video
    # @tool wraps the function; introspect the underlying signature via tool_spec
    spec = getattr(cosmos3_video2video, "tool_spec", None)
    props = {}
    if spec:
        schema = spec.get("inputSchema", {})
        props = (schema.get("json") or schema).get("properties", {}) if isinstance(schema, dict) else {}
    # Either the decorated spec exposes them, or the raw function does.
    names = set(props) if props else set(inspect.signature(
        getattr(cosmos3_video2video, "__wrapped__", cosmos3_video2video)).parameters)
    assert "condition_frames" in names
    assert "condition_keep" in names
    assert "guidance" in names


def test_cosmos3_image2video_sound_tool():
    from strands_cosmos import cosmos3_image2video_sound
    assert hasattr(cosmos3_image2video_sound, "tool_spec") or callable(cosmos3_image2video_sound)
    import strands_cosmos as sc
    assert "cosmos3_image2video_sound" in sc.__all__


def test_cosmos3_video2video_sound_params():
    import inspect
    from strands_cosmos import cosmos3_video2video
    names = set(inspect.signature(
        getattr(cosmos3_video2video, "__wrapped__", cosmos3_video2video)).parameters)
    assert "generate_sound" in names
    assert "max_sequence_length" in names


def test_cosmos3_extra_tools_import():
    from strands_cosmos import (
        cosmos3_upsample_prompt, cosmos3_caption_batch, cosmos3_eval_videophy2,
    )
    import strands_cosmos as sc
    for n in ["cosmos3_upsample_prompt", "cosmos3_caption_batch", "cosmos3_eval_videophy2"]:
        assert n in sc.__all__, f"{n} missing from __all__"
    for t in [cosmos3_upsample_prompt, cosmos3_caption_batch, cosmos3_eval_videophy2]:
        assert hasattr(t, "tool_spec") or callable(t)


def test_cosmos3_upsample_task_validation():
    from strands_cosmos import cosmos3_upsample_prompt
    # invalid task returns an error result without hitting any server
    r = cosmos3_upsample_prompt("a robot", task="bogus")
    assert r["status"] == "error"


def test_cosmos3_default_resolution_is_480():
    """Upstream Generator default resolution is 480p; our defaults should match."""
    import inspect
    from strands_cosmos import (
        cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
        cosmos3_text2video_sound, cosmos3_image2video_sound,
    )
    from strands_cosmos.cosmos3_generator_model import Cosmos3GeneratorModel
    for fn in [cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
               cosmos3_text2video_sound, cosmos3_image2video_sound]:
        sig = inspect.signature(getattr(fn, "__wrapped__", fn))
        assert sig.parameters["res"].default == "480", f"{fn} res default != 480"
    gsig = inspect.signature(Cosmos3GeneratorModel.generate)
    assert gsig.parameters["resolution"].default == "480"
