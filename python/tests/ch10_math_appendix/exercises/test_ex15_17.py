import math

import pytest

from p4mfe.ch10_math_appendix.exercises.ex15_17_nested_limits import (
    EX17_X_MAX,
    ex15_iterate,
    ex15_limit,
    ex16_iterate,
    ex16_limit,
    ex17_tower,
    ex17_x_for,
)

GOLDEN = (1 + math.sqrt(5)) / 2


def test_ex15_nested_radical():
    assert ex15_limit(2.0) == 2.0
    assert ex15_limit(6.0) == 3.0
    assert ex15_limit(1.0) == pytest.approx(GOLDEN, abs=1e-15)
    for a in (0.01, 0.5, 1.0, 2.0, 6.0, 100.0):
        limit = ex15_limit(a)
        prev = 0.0
        for n in range(1, 11):
            x = ex15_iterate(a, n)
            assert prev < x < limit
            prev = x
        assert ex15_iterate(a, 200) == pytest.approx(limit, rel=1e-13)
    with pytest.raises(ValueError):
        ex15_limit(0.0)


def test_ex16_continued_fraction():
    assert ex16_limit(1.0) == pytest.approx(GOLDEN, abs=1e-15)
    assert ex16_limit(2.0) == pytest.approx(1 + math.sqrt(2), abs=1e-15)
    for a in (0.1, 0.5, 1.0, 2.0, 10.0):
        limit = ex16_limit(a)
        assert limit == pytest.approx(a + 1 / limit, rel=1e-14)
        for n in (1, 3, 5):
            assert ex16_iterate(a, n) < limit < ex16_iterate(a, n + 1)
        assert ex16_iterate(a, 2000) == pytest.approx(limit, rel=1e-13)


def test_ex17_tower_equal_to_two():
    x = ex17_x_for(2.0)
    assert x == pytest.approx(math.sqrt(2), abs=1e-15)
    assert ex17_x_for(4.0) == pytest.approx(math.sqrt(2), abs=1e-15)
    prev = 0.0
    for n in range(1, 31):
        t = ex17_tower(x, n)
        assert prev < t < 2
        prev = t
    assert ex17_tower(x, 200) == pytest.approx(2, abs=1e-13)


def test_ex17_largest_x_and_b():
    assert ex17_x_for(math.e) == pytest.approx(EX17_X_MAX, abs=1e-15)
    for b in (0.5, 1.0, 2.0, 2.7, 2.72, 3.0, 4.0, 10.0):
        assert ex17_x_for(b) < EX17_X_MAX
    n = 20000
    t = ex17_tower(EX17_X_MAX, n)
    assert math.e - 1e-3 < t < math.e
    assert (math.e - t) * n / (2 * math.e) == pytest.approx(1, abs=0.05)
    assert ex17_tower(EX17_X_MAX + 1e-3, 100000) == math.inf
    below = ex17_tower(1.44, 100000)
    assert below < math.e
    assert 1.44**below == pytest.approx(below, abs=1e-12)
    half = ex17_tower(0.5, 200)
    assert 0.5**half == pytest.approx(half, abs=1e-14)
    assert half < 1
