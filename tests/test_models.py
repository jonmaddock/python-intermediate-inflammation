"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ],
)
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean

    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


# TODO(lesson-robust) Implement tests for the other statistical functions


@pytest.mark.parametrize(
    "test, expected",
    [([[1, 2], [3, 4], [5, 6]], [5, 6]), ([[1, 9], [3, 7], [3, 4]], [3, 9])],
)
def test_daily_max(test, expected):
    from inflammation.models import daily_max

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [([[1, 2], [3, 4], [5, 6]], [1, 2]), ([[12, 4], [3, 4], [6, 9]], [3, 4])],
)
def test_daily_min(test, expected):
    from inflammation.models import daily_min

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello", "there"], ["General", "Kenobi"]])
