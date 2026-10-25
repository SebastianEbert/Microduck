"""Microduck polite bow task, built on the ground-pick task.

Changes vs ground pick:
  * mouth stops ~8 cm above the floor instead of skimming it,
  * weaker "mouth points straight down" reward (a bow looks forward-ish),
  * longer pause at the bottom,
  * no simulated object carried in the mouth.

Phase profile (period 4 s); infer_policy.py ends the move at phase 0.7:
    [0.00, 0.25) down 1.0 s | [0.25, 0.40) hold 0.6 s | [0.40, 0.65) up 1.0 s | rest
"""

from dataclasses import replace

from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)

# ── Knobs ─────────────────────────────────────────────────────────────────────
BOW_PERIOD = 4.0          # must match --ground-pick-period at play time
DESCENT_END = 0.25
HOLD_END = 0.40
RISE_END = 0.65           # must stay < 0.7
MOUTH_HEIGHT = 0.08       # m above floor at the bottom (lower = deeper bow)
MOUTH_DOWN_WEIGHT = 0.5   # ground pick uses 2.0

# infer_policy.py hands control back to walking at phase 0.7.
assert 0.0 < DESCENT_END < HOLD_END < RISE_END < 0.7

_PHASE_KNOBS = {
    "descent_end": DESCENT_END,
    "hold_end": HOLD_END,
    "rise_end": RISE_END,
}


def make_microduck_bow_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=rough)

    # Re-time every phase-gated term. In the ground-pick cfg only rewards carry
    # phase params (events/curricula are step- or reset-based), but sweep all
    # three managers so a future upstream gated event/curriculum isn't missed.
    for terms in (cfg.rewards, cfg.events, cfg.curriculum):
        for term in terms.values():
            params = getattr(term, "params", None)
            if not params:
                continue
            for key, value in _PHASE_KNOBS.items():
                if key in params:
                    params[key] = value

    cfg.rewards["mouth_ground_proximity"].params["target_height"] = MOUTH_HEIGHT
    cfg.rewards["mouth_perpendicular_to_ground"].weight = MOUTH_DOWN_WEIGHT

    cfg.rewards.pop("mouth_payload_force", None)
    cfg.events.pop("sample_mouth_payload", None)

    cfg.commands["twist"].period = BOW_PERIOD
    return cfg


MicroduckBowRlCfg = replace(
    MicroduckGroundPickRlCfg,
    experiment_name="bow",
    run_name="bow",
    max_iterations=5_000,
)
