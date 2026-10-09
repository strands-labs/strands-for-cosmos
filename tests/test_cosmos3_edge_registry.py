"""Cosmos 3 model catalog + Edge embodiment registry (CPU only, no model downloads)."""
import pytest

from strands_cosmos import cosmos3_models as m


@pytest.mark.parametrize("name,expected", [
    ("nvidia/Cosmos3-Edge", m.EDGE), ("Edge", m.EDGE), ("cosmos3-edge", m.EDGE), ("NVIDIA/cosmos3-edge", m.EDGE),
    ("nvidia/Cosmos3-Nano", m.NANO), ("nano", m.NANO), ("", m.NANO),
    ("edge-policy-droid", m.EDGE_POLICY_DROID),
])
def test_resolve_model_id(name, expected):
    assert m.resolve_model_id(name) == expected


def test_unknown_names_pass_through():
    assert m.resolve_model_id("/data/my-finetune") == "/data/my-finetune"
    assert m.get_model("/data/my-finetune") is None


def test_edge_catalog_entry_is_honest():
    e = m.get_model("edge")
    assert e.hf_reasoner_class == "Cosmos3EdgeForConditionalGeneration"
    assert e.min_transformers == "5.19.0"
    assert not e.supports("sound")  # sound_gen=False on the published checkpoint
    assert e.supports("reason") and e.supports("generate") and e.supports("action")
    assert e.params_b["understanding"] == pytest.approx(2.4356)
    assert e.license == "OpenMDW-1.1"


def test_require_surface_blocks_sound_on_edge_but_not_nano():
    with pytest.raises(ValueError, match="no 'sound' surface"):
        m.require_surface("edge", "sound")
    assert m.require_surface("nano", "sound") == m.NANO
    # unknown checkpoints are not blocked (we cannot know their surfaces)
    assert m.require_surface("/data/x", "sound") == "/data/x"


def test_trained_embodiments_match_measured_domain_ids():
    ids = {m.EDGE_EMBODIMENTS[n].domain_id for n in m.trained_embodiments()}
    assert ids == set(m.EDGE_TRAINED_DOMAIN_IDS) == {1, 2, 3, 6, 7, 8, 12, 13, 15, 20}
    assert len(m.EDGE_TRAINED_DOMAIN_IDS) == 10
    assert all(0 <= e.domain_id < m.EDGE_NUM_DOMAINS for e in m.EDGE_EMBODIMENTS.values())


@pytest.mark.parametrize("bad", ["so101", "so100", "franka", ""])
def test_unknown_embodiment_errors_early(bad):
    with pytest.raises(ValueError, match="not a Cosmos3-Edge embodiment"):
        m.validate_embodiment(bad)


@pytest.mark.parametrize("untrained", ["pusht", "libero", "galbot", "no_action"])
def test_untrained_embodiment_is_refused_unless_overridden(untrained):
    with pytest.raises(ValueError, match="untrained"):
        m.validate_embodiment(untrained)
    emb = m.validate_embodiment(untrained, allow_untrained=True)
    assert emb.trained is False


def test_trained_embodiment_resolves_with_raw_dim():
    emb = m.validate_embodiment("droid_lerobot")
    assert (emb.domain_id, emb.raw_action_dim, emb.trained) == (8, 10, True)
    assert m.validate_embodiment("bridge_orig_lerobot").raw_action_dim == 10
    assert m.validate_embodiment("hand_pose").raw_action_dim == 57
