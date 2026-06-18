"""Tests for oriented bounding-box collision geometry.

Validates that collision detection uses hull geometry, not
isotropic centroid-distance thresholds.
"""

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.evaluation.geometry import (
    oriented_box_corners,
    polygons_overlap_sat,
    hulls_overlap_oriented_rect,
    centroid_distance,
)


class TestOrientedBoxCorners:
    """Corner computation for oriented rectangles."""

    def test_zero_heading(self):
        """Box at origin, heading 0 (east)."""
        corners = oriented_box_corners(0.0, 0.0, 0.0, 10.0, 4.0)
        assert corners.shape == (4, 2)
        # Front-right (starboard bow): (5, -2)
        np.testing.assert_array_almost_equal(corners[0], [5.0, -2.0])
        # Front-left (port bow): (5, 2)
        np.testing.assert_array_almost_equal(corners[1], [5.0, 2.0])
        # Aft-left (port quarter): (-5, 2)
        np.testing.assert_array_almost_equal(corners[2], [-5.0, 2.0])
        # Aft-right (starboard quarter): (-5, -2)
        np.testing.assert_array_almost_equal(corners[3], [-5.0, -2.0])

    def test_90_deg_heading(self):
        """Box at origin, heading 90 deg (north)."""
        corners = oriented_box_corners(0.0, 0.0, np.pi / 2, 10.0, 4.0)
        # Front-right → bow direction is north, starboard is east
        np.testing.assert_array_almost_equal(corners[0], [2.0, 5.0], decimal=4)

    def test_offset_center(self):
        """Box at non-zero center."""
        corners = oriented_box_corners(100.0, 50.0, 0.0, 20.0, 6.0)
        center = np.mean(corners, axis=0)
        np.testing.assert_array_almost_equal(center, [100.0, 50.0])


class TestPolygonsOverlapSAT:
    """Separating Axis Theorem overlap tests."""

    def test_separated_boxes_no_overlap(self):
        """Two boxes far apart should not overlap."""
        box_a = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        box_b = np.array([[5., 0.], [7., 0.], [7., 1.], [5., 1.]])
        assert not polygons_overlap_sat(box_a, box_b)

    def test_overlapping_boxes(self):
        """Two overlapping boxes."""
        box_a = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        box_b = np.array([[1., 0.5], [3., 0.5], [3., 1.5], [1., 1.5]])
        assert polygons_overlap_sat(box_a, box_b)

    def test_just_touching_edge(self):
        """Boxes sharing an edge overlap (SAT treats boundary-touch as overlap).

        This is a conservative choice: boundary contact is treated as overlap
        for safety-critical collision detection. Use safety_padding to control
        the effective overlap threshold.
        """
        box_a = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        box_b = np.array([[2., 0.], [4., 0.], [4., 1.], [2., 1.]])
        # SAT: touching edge = overlapping projections = overlap
        assert polygons_overlap_sat(box_a, box_b)

    def test_rotated_overlap(self):
        """Rotated boxes that overlap."""
        # Box A at 0 deg, Box B at 45 deg, overlapping
        box_a = oriented_box_corners(0., 0., 0., 10., 4.)
        box_b = oriented_box_corners(3., 1., np.pi / 6, 8., 3.)
        assert polygons_overlap_sat(box_a, box_b)

    def test_rotated_no_overlap(self):
        """Rotated boxes that don't overlap.

        Ship A (0,0) heading 0, 10x4. Ship B (13, 0) heading pi/2, 10x4.
        Ship A: x=[-5,5], y=[-2,2]. Ship B: x=[11,15], y=[-5,5].
        B's x range [11,15] does NOT overlap A's x range [-5,5].
        """
        box_a = oriented_box_corners(0., 0., 0., 10., 4.)
        box_b = oriented_box_corners(13., 0., np.pi / 2, 10., 4.)
        assert not polygons_overlap_sat(box_a, box_b)


class TestHullsOverlapOrientedRect:
    """Realistic ship hull overlap scenarios."""

    def test_parallel_close_but_not_overlapping(self):
        """Parallel ships at close range with non-overlapping hulls."""
        # Ship A: heading east at (0, 0), 200m long, 30m beam
        # Ship B: heading east at (0, 50), 180m long, 25m beam
        assert not hulls_overlap_oriented_rect(
            0., 0., 0., 200., 30.,
            0., 50., 0., 180., 25.,
        )

    def test_head_on_overlap(self):
        """Head-on collision: hulls overlap."""
        # Ship A heading east at (0, 0)
        # Ship B heading west at (5, 0) — bow-to-bow overlap
        assert hulls_overlap_oriented_rect(
            0., 0., 0., 100., 20.,
            5., 0., np.pi, 100., 20.,
        )

    def test_crossing_overlap(self):
        """Crossing collision: hulls overlap."""
        # Ship A heading east, Ship B heading north, intersection
        assert hulls_overlap_oriented_rect(
            0., 0., 0., 100., 20.,
            50., -30., np.pi / 2, 80., 15.,
        )

    def test_centroid_close_but_no_overlap(self):
        """Centroids close but hulls don't overlap due to beam separation.

        Ship A (0,0) heading 0, 50x10 → x=[-25,25], y=[-5,5].
        Ship B (0, 50) heading π/2, 50x10 → x=[-5,5], y=[25,75].
        y ranges don't overlap: A=[-5,5], B=[25,75].
        x ranges do overlap at [-5,5] but since y is separated → no overlap.
        """
        assert not hulls_overlap_oriented_rect(
            0., 0., 0., 50., 10.,
            0., 50., np.pi / 2, 50., 10.,
        )

    def test_centroid_close_with_overlap(self):
        """Same centroids but wider beam and zero separation → overlap."""
        assert hulls_overlap_oriented_rect(
            0., 0., 0., 50., 25.,
            0., 10., 0., 50., 25.,
        )

    def test_safety_padding_expands_hull(self):
        """Safety padding makes non-collision into collision.

        Two ships separated by exactly the gap between their hulls.
        Without padding: no overlap. With padding: overlap.
        Ship A (0,0) heading 0, 50x20 → x=[-25,25], y=[-10,10].
        Ship B (55,0) heading 0, 50x20 → x=[30,80], y=[-10,10].
        Gap = 30-25 = 5m (no overlap without padding).
        """
        # With 0 padding: no overlap (gap = 5m)
        assert not hulls_overlap_oriented_rect(
            0., 0., 0., 50., 20.,
            55., 0., 0., 50., 20.,
            safety_padding=0.0,
        )
        # With 3m padding: expanded hulls overlap (gap = 5m < 2*3=6m)
        assert hulls_overlap_oriented_rect(
            0., 0., 0., 50., 20.,
            55., 0., 0., 50., 20.,
            safety_padding=3.0,
        )

    def test_safety_domain_not_collision(self):
        """Safety-domain violation (with padding) != collision (no padding).

        Ship A (0,0) heading 0, 100x20 → x=[-50,50], y=[-10,10].
        Ship B (60, 15) heading 0, 100x20 → x=[10,110], y=[5,25].
        Without padding: y ranges A=[-10,10], B=[5,25] → overlap at [5,10].
        x ranges A=[-50,50], B=[10,110] → overlap at [10,50].
        → Both overlap → hulls overlap. Not a good test case.

        Try: Ship B offset enough in y that beams don't overlap without padding.
        Ship A (0,0) heading 0, 100x20 → x=[-50,50], y=[-10,10].
        Ship B (0, 25) heading 0, 100x20 → x=[-50,50], y=[15,35].
        y gap: 10 to 15 = 5m gap. No collision! Domain with 5m padding: overlap!
        """
        # No collision: 5m gap between hulls
        is_collision = hulls_overlap_oriented_rect(
            0., 0., 0., 100., 20.,
            0., 25., 0., 100., 20.,
            safety_padding=0.0,
        )
        # Domain violation: 5m padding fills 5m gap → overlap
        is_domain_violation = hulls_overlap_oriented_rect(
            0., 0., 0., 100., 20.,
            0., 25., 0., 100., 20.,
            safety_padding=3.0,
        )

        assert not is_collision, "Should not be a collision (hulls don't touch)"
        assert is_domain_violation, "Should be a domain violation (hulls + padding overlap)"


class TestLegacyThresholdReplaced:
    """Verify that the legacy centroid distance is NOT used for collision."""

    def test_legacy_threshold_would_misclassify(self):
        """Centroid distance < 0.25*(L1+L2) but hulls don't overlap.

        Two long, narrow ships, parallel, close together.
        Centroid distance < 0.25*(L1+L2) = 0.25*400 = 100m.
        At d=60m, legacy would call collision. But oriented check: no overlap.
        """
        legacy_threshold = 0.25 * (200.0 + 200.0)  # 100m
        d_centroid = 60.0  # less than threshold

        # Parallel ships, 60m apart: hulls at centered, beam=20m each
        # Half beams: 10m + 10m = 20m. At 60m separation, hulls don't touch.
        is_legacy_collision = d_centroid < legacy_threshold  # True!
        is_oriented_collision = hulls_overlap_oriented_rect(
            0., 0., 0., 200., 20.,
            0., d_centroid, 0., 200., 20.,
            safety_padding=0.0,
        )

        assert is_legacy_collision, "Legacy threshold says collision (this is the bug)"
        assert not is_oriented_collision, (
            "Oriented check correctly says no collision "
            f"(parallel ships at d={d_centroid}m)"
        )

    def test_centroid_distance_only(self):
        """Validate that centroid_distance() exists and works."""
        d = centroid_distance(0., 0., 30., 40.)
        assert abs(d - 50.0) < 1e-6
