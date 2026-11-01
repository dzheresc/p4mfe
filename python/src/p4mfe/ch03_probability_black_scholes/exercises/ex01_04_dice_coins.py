"""Exercises 1-4: a dice game; tosses until the first head and until k heads in a row.

1. P(sum = k) = n(k)/36 with n(k) = 6 - |k - 7|. The game is worth playing when
   w p - (1 - p) >= 0, i.e. w(k) >= 36/n(k) - 1: 35, 17, 11, 8, 6.2, 5 for
   k = 2..7 (symmetric about 7).
2. Geometric with p = 1/2: E[X] = 1/p = 2, var(X) = (1-p)/p^2 = 2.
3-4. E_k = E_{k-1} + 1 + (1-p) E_k  =>  E_k = (E_{k-1} + 1)/p, so
   E_k = 1/p + ... + 1/p^k = (1 - p^k)/((1-p) p^k). k = 2: (1+p)/p^2, 6 for a fair coin.

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex01_04_dice_coins.hpp
"""

from __future__ import annotations


def ex01_ways(k: int) -> int:
    if not 2 <= k <= 12:
        raise ValueError("need 2 <= k <= 12")
    return 6 - abs(k - 7)


def ex01_prob(k: int) -> float:
    return ex01_ways(k) / 36


def ex01_breakeven_win(k: int) -> float:
    return 36 / ex01_ways(k) - 1


def ex01_expected_gain(k: int, w: float) -> float:
    p = ex01_prob(k)
    return w * p - (1 - p)


def ex02_mean(p: float = 0.5) -> float:
    return 1 / p


def ex02_variance(p: float = 0.5) -> float:
    return (1 - p) / p**2


def ex03_expected_tosses(p: float = 0.5) -> float:
    return (1 + p) / p**2


def ex04_expected_tosses(k: int, p: float) -> float:
    if k < 0 or not 0 < p <= 1:
        raise ValueError("need k >= 0, 0 < p <= 1")
    if p == 1:
        return float(k)
    return (1 - p**k) / ((1 - p) * p**k)


def ex04_expected_tosses_recursive(k: int, p: float) -> float:
    e = 0.0
    for _ in range(k):
        e = (e + 1) / p
    return e


def ex04_expected_tosses_markov(k: int, p: float, tol: float = 1e-14) -> float:
    """E[N] = sum_n P(N > n), propagating the current run length (0..k-1) toss by toss."""
    state = [1.0] + [0.0] * (k - 1)
    expected = 0.0
    while True:
        alive = sum(state)
        expected += alive
        if alive < tol:
            return expected
        nxt = [0.0] * k
        for j, s in enumerate(state):
            nxt[0] += (1 - p) * s
            if j + 1 < k:
                nxt[j + 1] += p * s
        state = nxt
