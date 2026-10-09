# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos 3 model catalog + embodiment registry (no heavy imports).

Every Cosmos 3 surface in this package used to assume ``nvidia/Cosmos3-Nano``.
This module is the single place that knows which checkpoints exist, what each
one can do, and - for ``nvidia/Cosmos3-Edge`` - which action embodiments are
actually trained.

Numbers marked MEASURED come from logs on a Jetson AGX Thor (arm64, CUDA
sm_110, torch 2.14.1+cu130, transformers 5.19.0, diffusers 0.41.0), Edge
revision ``344d602b128d1bbdacb43b08d0a3626f46343e29``; see
``docs/guide/cosmos3-edge.md`` for the table and the commands.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

NANO = "nvidia/Cosmos3-Nano"
EDGE = "nvidia/Cosmos3-Edge"
NANO_POLICY_DROID = "nvidia/Cosmos3-Nano-Policy-DROID"
EDGE_POLICY_DROID = "nvidia/Cosmos3-Edge-Policy-DROID"

#: Pinned Edge revision all measurements refer to.
EDGE_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"


@dataclass(frozen=True)
class Cosmos3Model:
    """What one Cosmos 3 checkpoint is and can do."""

    id: str
    short: str
    size_label: str
    surfaces: FrozenSet[str]  # subset of {"reason", "generate", "action", "sound"}
    #: transformers class that loads the understanding tower in-process, or None
    #: when the only reasoner route is a vLLM server (``Cosmos3ReasonerForConditionalGeneration``).
    hf_reasoner_class: Optional[str]
    min_transformers: Optional[str]
    min_diffusers: Optional[str]
    license: str
    #: Understanding / generation tower parameter counts in billions (MEASURED for Edge).
    params_b: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def supports(self, surface: str) -> bool:
        return surface in self.surfaces


CATALOG: Dict[str, Cosmos3Model] = {
    NANO: Cosmos3Model(
        id=NANO, short="Nano", size_label="16B",
        surfaces=frozenset({"reason", "generate", "action", "sound"}),
        hf_reasoner_class=None, min_transformers=None, min_diffusers="0.36",
        license="NVIDIA Open Model License",
        notes="Omnimodal; reasoner served by vLLM (Cosmos3ReasonerForConditionalGeneration); fits one ~46 GB GPU.",
    ),
    EDGE: Cosmos3Model(
        id=EDGE, short="Edge", size_label="4B",
        surfaces=frozenset({"reason", "generate", "action"}),  # sound_gen=False in transformer/config.json
        hf_reasoner_class="Cosmos3EdgeForConditionalGeneration",
        min_transformers="5.19.0", min_diffusers="0.40.0",
        license="OpenMDW-1.1",
        params_b={"understanding": 2.4356, "generation": 3.3697, "vae": 0.7047, "total": 4.5637},  # MEASURED
        notes="MoT: shared 1.95B trunk + und vision tower / gen diffusion tower. Loads natively with transformers>=5.19 "
              "(no vLLM needed); action_dim 64, 32 embodiment slots of which 10 are trained; no sound tower.",
    ),
    NANO_POLICY_DROID: Cosmos3Model(
        id=NANO_POLICY_DROID, short="Nano-Policy-DROID", size_label="16B",
        surfaces=frozenset({"action"}), hf_reasoner_class=None, min_transformers=None, min_diffusers="0.36",
        license="NVIDIA Open Model License", notes="VL robot policy post-trained on DROID.",
    ),
    EDGE_POLICY_DROID: Cosmos3Model(
        id=EDGE_POLICY_DROID, short="Edge-Policy-DROID", size_label="4B",
        surfaces=frozenset({"action"}), hf_reasoner_class=None, min_transformers=None, min_diffusers="0.40.0",
        license="OpenMDW-1.1", notes="Edge-size VL robot policy post-trained on DROID (not measured here).",
    ),
}

_ALIASES = {
    "nano": NANO, "cosmos3-nano": NANO, "cosmos3_nano": NANO,
    "edge": EDGE, "cosmos3-edge": EDGE, "cosmos3_edge": EDGE,
    "nano-policy-droid": NANO_POLICY_DROID, "cosmos3-nano-policy-droid": NANO_POLICY_DROID,
    "edge-policy-droid": EDGE_POLICY_DROID, "cosmos3-edge-policy-droid": EDGE_POLICY_DROID,
}


def resolve_model_id(name: str) -> str:
    """Normalise ``"Edge"`` / ``"cosmos3-edge"`` / ``"nvidia/Cosmos3-Edge"`` to the HF id.

    Unknown names (local paths, other checkpoints) pass through unchanged.
    """
    key = (name or "").strip()
    if not key:
        return NANO
    if key in CATALOG:
        return key
    low = key.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    if low.startswith("nvidia/") and low[len("nvidia/"):] in _ALIASES:
        return _ALIASES[low[len("nvidia/"):]]
    return key


def get_model(name: str) -> Optional[Cosmos3Model]:
    """Catalog entry for a (possibly aliased) name, or None for unknown checkpoints."""
    return CATALOG.get(resolve_model_id(name))


def require_surface(name: str, surface: str) -> str:
    """Return the resolved id, raising ``ValueError`` if the checkpoint lacks ``surface``."""
    mid = resolve_model_id(name)
    m = CATALOG.get(mid)
    if m is not None and not m.supports(surface):
        have = ", ".join(sorted(m.surfaces))
        raise ValueError(
            f"{mid} has no '{surface}' surface (it offers: {have}). "
            + ("Cosmos3-Nano is the sound-capable checkpoint." if surface == "sound" else "")
        )
    return mid


# -- Edge action embodiments -------------------------------------------------
# Names/ids/raw dims mirror diffusers' Cosmos3OmniPipeline registry
# (_EMBODIMENT_TO_DOMAIN_ID / _EMBODIMENT_TO_RAW_ACTION_DIM, diffusers 0.41.0).
# ``trained`` is MEASURED: per-slot L2 of the DomainAwareLinear output head on
# the pinned Edge revision splits the 32 slots into 10 trained (nonzero bias,
# weight L2 5.41..13.36) and 22 untrained (bias exactly 0, weight L2 4.916..4.945).
# The trained set equals exactly the 10 embodiments NVIDIA's model card lists.
@dataclass(frozen=True)
class Embodiment:
    name: str
    domain_id: int
    raw_action_dim: Optional[int]
    trained: bool


EDGE_ACTION_DIM = 64
EDGE_NUM_DOMAINS = 32
EDGE_TRAINED_DOMAIN_IDS: FrozenSet[int] = frozenset({1, 2, 3, 6, 7, 8, 12, 13, 15, 20})

_EMB: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("no_action", 0, None), ("av", 1, 9), ("camera_pose", 2, 9), ("hand_pose", 3, 57),
    ("pusht", 4, 2), ("libero", 5, None), ("umi", 6, 10), ("bridge_orig_lerobot", 7, 10),
    ("droid_lerobot", 8, 10), ("robomind-franka", 8, 10), ("galbot", 9, 30),
    ("robomind-franka-dual", 12, 20), ("robomind-ur", 13, 10), ("agibotworld", 15, 29),
    ("agibot_gear_gripper", 15, 29), ("agibot_gear_gripper_ext", 15, 29), ("fractal", 20, 10),
)
EDGE_EMBODIMENTS: Dict[str, Embodiment] = {
    n: Embodiment(n, d, r, d in EDGE_TRAINED_DOMAIN_IDS) for n, d, r in _EMB
}


def trained_embodiments() -> Tuple[str, ...]:
    return tuple(n for n, e in EDGE_EMBODIMENTS.items() if e.trained)


def validate_embodiment(name: str, *, allow_untrained: bool = False) -> Embodiment:
    """Resolve an Edge embodiment name, failing early and clearly.

    Raises ``ValueError`` for unknown names (e.g. ``so101``) and - unless
    ``allow_untrained`` - for registered names whose action head was never
    trained (``pusht``, ``libero``, ``galbot``, ``no_action``).
    """
    key = (name or "").strip()
    emb = EDGE_EMBODIMENTS.get(key) or EDGE_EMBODIMENTS.get(key.lower())
    if emb is None:
        raise ValueError(
            f"'{name}' is not a Cosmos3-Edge embodiment. Trained embodiments: "
            f"{', '.join(trained_embodiments())}. (so101/so100-class 6-DoF arms are not registered; "
            f"adding one means post-training a new domain slot.)"
        )
    if not emb.trained and not allow_untrained:
        raise ValueError(
            f"Cosmos3-Edge embodiment '{emb.name}' (domain {emb.domain_id}) is registered but its action head is "
            f"untrained on revision {EDGE_REVISION[:8]} (output bias == 0); outputs would be noise. Trained: "
            f"{', '.join(trained_embodiments())}. Pass allow_untrained=True to override."
        )
    return emb


__all__ = [
    "NANO", "EDGE", "NANO_POLICY_DROID", "EDGE_POLICY_DROID", "EDGE_REVISION",
    "Cosmos3Model", "CATALOG", "resolve_model_id", "get_model", "require_surface",
    "Embodiment", "EDGE_EMBODIMENTS", "EDGE_ACTION_DIM", "EDGE_NUM_DOMAINS",
    "EDGE_TRAINED_DOMAIN_IDS", "trained_embodiments", "validate_embodiment",
]
