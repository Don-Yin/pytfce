"""Experiment configuration — single source of truth for all parameters.

Import this module from any experiment script to get consistent settings.
Edit here to change conditions globally.
"""

from __future__ import annotations

# ── phantom generation ─────────────────────────────────────────────
N_SUBJECTS = 80
SMOOTH_SIGMA = 1.5
SEED = 42
SHAPE_DEFAULT = (64, 64, 64)

# ── volume-size scaling ablation ───────────────────────────────────
VOLUME_SIZES = {
    "small": (32, 32, 32),
    "medium": (48, 48, 48),
    "default": (64, 64, 64),
}

# ── signal amplitude ablation ──────────────────────────────────────
AMPLITUDES = {
    "weak": 0.3,
    "medium": 0.5,
    "strong": 0.8,
    "very_strong": 1.2,
}

# ── signal shape ablation ──────────────────────────────────────────
# (imported at runtime to avoid circular deps)
SHAPE_BLOBS = {
    "small":     [{"c": (32, 32, 32), "r": (4, 4, 4)}],
    "large":     [{"c": (32, 32, 32), "r": (8, 8, 8)}],
    "elongated": [{"c": (32, 32, 32), "r": (3, 3, 12)}],
    "multi":     [{"c": (22, 32, 32), "r": (5, 5, 5)},
                  {"c": (42, 32, 32), "r": (3, 3, 3)},
                  {"c": (32, 22, 32), "r": (4, 4, 4)}],
}

# ── pTFCE core ─────────────────────────────────────────────────────
NH = 100
Z_EST_THR = 1.3
CONNECTIVITY = 6
LUT_DENSITY = 60

# ── TFCE permutation benchmarks ───────────────────────────────────
TFCE_PERM_COUNTS = [200, 500, 1000]

# ── parameter optimization ─────────────────────────────────────────
OPT_LR = 0.01
OPT_STEPS = 30
OPT_NH = 50
OPT_DIFFUSION_STEPS = 10
