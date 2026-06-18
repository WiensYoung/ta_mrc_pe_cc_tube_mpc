"""Tests for utility functions (math, coordinates, I/O)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.utils.math_utils import (
    clamp_angle_diff,
    clip_value,
    kappa_epsilon,
    max_eigenvalue,
    safe_divide,
    sigmoid,
)
from ta_mrc_pe_cc_tube_mpc.utils.coordinates import (
    body_to_world,
    distance,
    normalize_angle_deg,
    normalize_angle_rad,
    relative_bearing,
    rotation_matrix_2d,
    world_to_body,
)
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import deep_merge, resolve_path


class TestMathUtils:
    def test_clip_value(self):
        assert clip_value(5, 0, 10) == 5
        assert clip_value(-1, 0, 10) == 0
        assert clip_value(15, 0, 10) == 10

    def test_safe_divide(self):
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0  # zero denominator
        # denominator below default eps (1e-12) is treated as zero
        assert safe_divide(10, 0.5) == 20.0

    def test_kappa_epsilon_known_value(self):
        k = kappa_epsilon(0.05)
        assert k == pytest.approx(2.4477, rel=0.01)

    def test_kappa_epsilon_monotonic(self):
        k1 = kappa_epsilon(0.10)
        k2 = kappa_epsilon(0.05)
        k3 = kappa_epsilon(0.01)
        assert k1 < k2 < k3

    def test_max_eigenvalue_diagonal(self):
        S = np.diag([4.0, 9.0])
        assert max_eigenvalue(S) == pytest.approx(9.0)

    def test_max_eigenvalue_2d_symmetric(self):
        S = np.array([[2.0, 1.0], [1.0, 2.0]])
        lam = max_eigenvalue(S)
        assert lam == pytest.approx(3.0)

    def test_sigmoid(self):
        assert sigmoid(0) == 0.5
        assert sigmoid(10) > 0.99
        assert sigmoid(-10) < 0.01

    def test_clamp_angle_diff(self):
        # shortest signed diff: target - current, wrapped to [-pi, pi]
        diff = clamp_angle_diff(np.deg2rad(350), np.deg2rad(10))
        assert diff == pytest.approx(np.deg2rad(-20), rel=0.01)
        diff = clamp_angle_diff(np.deg2rad(10), np.deg2rad(350))
        assert diff == pytest.approx(np.deg2rad(20), rel=0.01)
        assert clamp_angle_diff(0.5, 0.5) == 0.0


class TestCoordinates:
    def test_body_to_world_zero_heading(self):
        result = body_to_world(5.0, 0.0, 0.0)
        np.testing.assert_allclose(result, [5.0, 0.0])

    def test_body_to_world_90deg_heading(self):
        result = body_to_world(5.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(result, [0.0, 5.0], atol=1e-10)

    def test_world_to_body_roundtrip(self):
        v_world = np.array([3.0, 4.0])
        psi = 0.5
        v_body = world_to_body(*v_world, psi)
        v_world2 = body_to_world(*v_body, psi)
        np.testing.assert_allclose(v_world2, v_world, atol=1e-10)

    def test_distance(self):
        assert distance(np.array([0, 0]), np.array([3, 4])) == 5.0

    def test_normalize_angle_rad(self):
        assert normalize_angle_rad(0) == 0.0
        # Normalizes to [-pi, pi); both +pi and -pi map to -pi
        assert abs(normalize_angle_rad(3 * np.pi)) == pytest.approx(np.pi)
        assert abs(normalize_angle_rad(-np.pi)) == pytest.approx(np.pi)

    def test_normalize_angle_deg(self):
        assert normalize_angle_deg(0) == 0.0
        assert normalize_angle_deg(370) == 10.0
        # Normalizes to [-180, 180), so -10 stays at -10
        assert normalize_angle_deg(-10) == -10.0
        assert normalize_angle_deg(-190) == 170.0

    def test_rotation_matrix_2d(self):
        R = rotation_matrix_2d(0)
        np.testing.assert_allclose(R, np.eye(2))
        R90 = rotation_matrix_2d(np.pi / 2)
        np.testing.assert_allclose(R90 @ [1, 0], [0, 1], atol=1e-10)

    def test_relative_bearing(self):
        own_pos = np.array([0.0, 0.0])
        # Returns degrees [0, 360)
        assert relative_bearing(own_pos, 0.0, np.array([10.0, 0.0])) == pytest.approx(0.0)
        # Target directly abeam to port → 90 deg
        assert relative_bearing(own_pos, 0.0, np.array([0.0, 10.0])) == pytest.approx(90.0)


class TestIOUtils:
    def test_deep_merge_nested(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}}
        override = {"b": {"x": 99}, "c": 3}
        result = deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["x"] == 99
        assert result["b"]["y"] == 2
        assert result["c"] == 3

    def test_deep_merge_empty_override(self):
        base = {"a": 1}
        assert deep_merge(base, {}) == {"a": 1}

    def test_resolve_path(self):
        result = resolve_path("/base/dir", "sub/file.txt")
        # os.path.join handles platform separators
        expected = os.path.join("/base/dir", "sub/file.txt")
        assert result == expected or result.replace("\\", "/") == expected.replace("\\", "/")
