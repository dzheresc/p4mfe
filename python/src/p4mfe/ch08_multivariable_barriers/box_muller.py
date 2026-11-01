"""Section 8.7: the Box-Muller method.

Z1 = sqrt(-2 ln U1) cos(2 pi U2), Z2 = sqrt(-2 ln U1) sin(2 pi U2) are
independent standard normals for independent uniform U1, U2: the inverse map
has |det J| = phi(z1) phi(z2) (change of variables, 8.3).
"""

from __future__ import annotations

import math

import numpy as np


# 8.7 -----------------------------------------------------------------------
def box_muller(u1, u2):
    rad = np.sqrt(-2 * np.log(u1))
    ang = 2 * math.pi * np.asarray(u2, float)
    return rad * np.cos(ang), rad * np.sin(ang)


def box_muller_inverse(z1, z2):
    u2 = np.arctan2(z2, z1) / (2 * math.pi)
    u2 = np.where(u2 < 0, u2 + 1, u2)
    return np.exp(-0.5 * (np.asarray(z1) ** 2 + np.asarray(z2) ** 2)), u2


def box_muller_samples(n: int, seed: int = 42) -> np.ndarray:
    """n standard normals from n/2 (rounded up) uniform pairs."""
    rng = np.random.default_rng(seed)
    m = (n + 1) // 2
    u1 = 1.0 - rng.random(m)  # (0, 1]
    z1, z2 = box_muller(u1, rng.random(m))
    out = np.empty(2 * m)
    out[0::2], out[1::2] = z1, z2
    return out[:n]
