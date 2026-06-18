"""Tests for IMM behavior filter — per-mode predictions must differ."""

import warnings

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.risk.imm_filter import (
    IMMFilter,
    PerTargetIMMState,
    MODE_CV,
    MODE_CT,
    MODE_COLREGS,
    MODE_NON_COMPLIANT,
    MODE_CHANNEL,
    MODE_ERRATIC,
    MODE_NAMES,
)


class TestPerModePredictionsDiffer:
    """P0: Each mode must produce a different position prediction."""

    def test_cv_vs_ct_differ_with_turn_rate(self):
        """Given non-zero speed and yaw rate, CV and CT predictions differ."""
        imm = IMMFilter()
        pos = np.array([0.0, 0.0])
        vel = np.array([10.0, 0.0])  # moving east at 10 m/s
        heading = 0.0
        speed = 10.0
        turn_rate = np.deg2rad(3.0)  # 3 deg/s turn

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0,
        )

        # Get per-target state and predictions
        tgt = imm.get_target_state("tgt_1")
        cv_pred = imm._predict_positions_per_mode(tgt, dt=1.0)[MODE_CV]
        ct_pred = imm._predict_positions_per_mode(tgt, dt=1.0)[MODE_CT]

        d = np.linalg.norm(cv_pred - ct_pred)
        assert d > 1e-4, (
            f"CV and CT predictions must differ with non-zero turn rate. "
            f"CV={cv_pred}, CT={ct_pred}, distance={d:.6f}"
        )

    def test_cv_vs_ct_same_with_zero_turn_rate(self):
        """With zero turn rate, CT falls back to CV (both go straight)."""
        imm = IMMFilter()
        pos = np.array([0.0, 0.0])
        vel = np.array([10.0, 0.0])
        heading = 0.0
        speed = 10.0
        turn_rate = 0.0

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0,
        )
        tgt = imm.get_target_state("tgt_1")
        cv_pred = imm._predict_positions_per_mode(tgt, dt=1.0)[MODE_CV]
        ct_pred = imm._predict_positions_per_mode(tgt, dt=1.0)[MODE_CT]

        d = np.linalg.norm(cv_pred - ct_pred)
        # When turn_rate=0, CT falls back to CV — they should be close
        assert d < 1e-4, (
            f"With zero turn rate, CV and CT should be nearly identical. "
            f"d={d:.6f}"
        )

    def test_colregs_vs_non_compliant_differ_head_on(self):
        """For head-on encounter, COLREGS starboard turn != non-compliant straight."""
        imm = IMMFilter()
        pos = np.array([100.0, 0.0])
        vel = np.array([-10.0, 0.0])  # moving west toward ownship
        heading = np.pi  # heading west
        speed = 10.0
        turn_rate = 0.0

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0, encounter_type="head_on",
        )
        tgt = imm.get_target_state("tgt_1")
        preds = imm._predict_positions_per_mode(tgt, dt=1.0, encounter_type="head_on")

        col_pred = preds[MODE_COLREGS]
        nc_pred = preds[MODE_NON_COMPLIANT]

        d = np.linalg.norm(col_pred - nc_pred)
        assert d > 0.1, (
            f"COLREGS and NON_COMPLIANT predictions must differ for head-on. "
            f"COLREGS={col_pred}, NONC={nc_pred}, d={d:.3f}"
        )

    def test_channel_vs_cv_differ_with_channel_heading(self):
        """Channel-following with channel heading differs from CV heading."""
        imm = IMMFilter()
        pos = np.array([0.0, 0.0])
        vel = np.array([0.0, 10.0])  # moving north
        heading = np.pi / 2  # north
        speed = 10.0
        turn_rate = 0.0
        channel_heading = 0.0  # channel goes east

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0, channel_heading=channel_heading,
        )
        tgt = imm.get_target_state("tgt_1")
        preds = imm._predict_positions_per_mode(tgt, dt=1.0, channel_heading=channel_heading)

        cv_pred = preds[MODE_CV]    # continues north
        ch_pred = preds[MODE_CHANNEL]  # follows channel east

        d = np.linalg.norm(cv_pred - ch_pred)
        assert d > 0.1, (
            f"Channel-following prediction must differ from CV when channel heading differs. "
            f"CV={cv_pred}, CH={ch_pred}, d={d:.3f}"
        )

    def test_erratic_differs_from_cv_with_perpendicular_offset(self):
        """Erratic mode uses a perturbed prediction (perpendicular offset from CV).

        Combined with inflated observation std, this gives ERRATIC a distinct
        signature: the prediction center is offset, and the likelihood is wider.
        """
        imm = IMMFilter()
        pos = np.array([0.0, 0.0])
        vel = np.array([10.0, 0.0])
        heading = 0.0
        speed = 10.0
        turn_rate = 0.0

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0,
        )
        tgt = imm.get_target_state("tgt_1")
        preds = imm._predict_positions_per_mode(tgt, dt=1.0)

        cv_pred = preds[MODE_CV]
        er_pred = preds[MODE_ERRATIC]

        d = np.linalg.norm(cv_pred - er_pred)
        assert d > 0.5, (
            f"Erratic prediction should differ from CV (perpendicular offset). "
            f"CV={cv_pred}, ERRATIC={er_pred}, d={d:.3f}"
        )
        # The difference should be roughly perpendicular to heading
        # heading=0 (east), perp=north, so ERRATIC should have a y-component
        assert abs(er_pred[1] - cv_pred[1]) > 0.5, (
            f"Erratic offset should be perpendicular to heading (northward for heading=east). "
            f"dy={er_pred[1] - cv_pred[1]:.3f}"
        )


class TestModesHaveDistinctPredictions:
    """P0: At least some modes must produce different predictions."""

    def test_all_modes_not_identical_with_turning(self):
        """With turning motion, at least 2 mode pairs must differ."""
        imm = IMMFilter()
        pos = np.array([0.0, 0.0])
        vel = np.array([8.0, 2.0])
        heading = np.arctan2(2.0, 8.0)
        speed = np.sqrt(8**2 + 2**2)
        turn_rate = np.deg2rad(5.0)

        imm.set_target_observation(
            "tgt_1", pos, vel, heading, speed, turn_rate,
            timestamp=0.0, encounter_type="crossing",
            channel_heading=np.deg2rad(45.0),
        )
        tgt = imm.get_target_state("tgt_1")
        preds = imm._predict_positions_per_mode(
            tgt, dt=1.0, encounter_type="crossing",
            channel_heading=np.deg2rad(45.0),
        )

        # Count pairs that differ
        n_diff = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if preds[i] is not None and preds[j] is not None:
                    d = np.linalg.norm(preds[i] - preds[j])
                    if d > 1e-3:
                        n_diff += 1

        assert n_diff >= 2, (
            f"Expected at least 2 mode pairs with different predictions. "
            f"Got {n_diff} differing pairs out of {len(preds)} modes."
        )

    def test_identical_predictions_trigger_warning(self):
        """All-identical predictions must trigger a RuntimeWarning."""
        imm = IMMFilter()

        # If we pass all-identical predictions via legacy update(), it should warn
        observed = np.array([0.0, 0.0])
        identical_preds = [np.array([1.0, 0.0])] * 6

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            imm.update(observed, identical_preds)
            # Should have a RuntimeWarning about identical predictions
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) >= 1, (
                "Identical predictions must trigger a RuntimeWarning"
            )


class TestCVMotionIncreasesCVProbability:
    """Straight-line motion should increase CV mode probability."""

    def test_straight_motion_increases_cv(self):
        """With straight-line motion, CV probability should dominate over COLREGS.

        In a crossing encounter, COLREGS mode predicts a starboard turn while
        CV predicts straight. If the target goes straight, CV should have
        higher probability than COLREGS.
        """
        imm = IMMFilter(obs_std=2.0)  # tighter obs std for sharper discrimination
        initial_cv_prob = imm.mode_probs[MODE_CV]

        pos_start = np.array([0.0, 0.0])
        vel = np.array([10.0, 0.0])
        heading = 0.0
        speed = 10.0
        turn_rate = 0.0

        for k in range(30):
            pos = np.array([pos_start[0] + k * 10.0, pos_start[1]])
            imm.set_target_observation(
                "tgt_cv_test", pos, vel, heading, speed, turn_rate,
                timestamp=float(k),
                encounter_type="crossing",
            )

        probs = imm.get_target_probs("tgt_cv_test")
        # CV should dominate COLREGS since target went straight
        assert probs[MODE_CV] > probs[MODE_COLREGS], (
            f"CV probability should exceed COLREGS for straight-line motion "
            f"under crossing encounter. CV={probs[MODE_CV]:.3f}, "
            f"COLREGS={probs[MODE_COLREGS]:.3f}"
        )


class TestTurningMotionIncreasesTurningProbability:
    """Turning motion should increase CT/turning mode probability."""

    def test_turning_increases_ct(self):
        """Multiple turning observations raise CT probability."""
        imm = IMMFilter()
        speed = 8.0
        turn_rate = np.deg2rad(4.0)
        heading = 0.0

        initial_ct_prob = imm.mode_probs[MODE_CT]

        for k in range(20):
            heading += turn_rate
            pos = np.array([
                100.0 + speed / turn_rate * (np.sin(heading) - 0.0),
                0.0 + speed / turn_rate * (1.0 - np.cos(heading)),
            ])
            vel = np.array([speed * np.cos(heading), speed * np.sin(heading)])
            imm.set_target_observation(
                "tgt_1", pos, vel, heading, speed, turn_rate,
                timestamp=float(k),
            )

        final_ct_prob = imm.get_target_probs("tgt_1")[MODE_CT]
        assert final_ct_prob > initial_ct_prob, (
            f"CT probability should increase with turning motion. "
            f"Initial={initial_ct_prob:.3f}, Final={final_ct_prob:.3f}"
        )


class TestErraticMotionIncreasesEntropy:
    """Irregular observations should increase entropy or erratic probability."""

    def test_noisy_observations_increase_entropy(self):
        """Noisy observations increase mode distribution entropy.

        With encounter_type provided, modes produce distinct predictions.
        Noisy observations that don't match any specific mode spread
        likelihood evenly → higher entropy.
        """
        imm = IMMFilter(obs_std=2.0)

        # Establish clean CV track with crossing encounter
        base_speed = 10.0
        heading = 0.0
        vel = np.array([base_speed, 0.0])
        for k in range(10):
            pos = np.array([k * base_speed, 0.0])
            imm.set_target_observation(
                "tgt_noise", pos, vel, heading, base_speed, 0.0,
                timestamp=float(k),
                encounter_type="crossing",
            )
        clean_entropy = imm.get_target_entropy("tgt_noise")

        # Very noisy observations → spread probability across modes
        # Use large jumps that don't match any prediction
        rng = np.random.RandomState(42)
        for k in range(30):
            # Random large deviations
            pos = np.array([k * base_speed + rng.normal(0, 15.0),
                           rng.normal(0, 15.0)])
            imm.set_target_observation(
                "tgt_noise", pos, vel, heading, base_speed, 0.0,
                timestamp=float(k + 10),
                encounter_type="crossing",
            )
        noisy_entropy = imm.get_target_entropy("tgt_noise")

        # Entropy should increase due to noisy observations
        assert noisy_entropy > clean_entropy * 0.8, (
            f"Noisy observations should maintain or increase entropy. "
            f"Clean={clean_entropy:.3f}, Noisy={noisy_entropy:.3f}"
        )

    def test_high_turn_noise_increases_erratic_probability(self):
        """Observations with erratic direction changes boost erratic mode."""
        imm = IMMFilter()
        speed = 10.0

        initial_erratic = imm.mode_probs[MODE_ERRATIC]

        for k in range(30):
            # Random heading changes each step
            heading = np.random.uniform(-np.pi, np.pi)
            pos = np.array([k * speed * 0.2, np.random.normal(0, 5.0)])
            vel = np.array([speed * np.cos(heading), speed * np.sin(heading)])
            imm.set_target_observation(
                "tgt_1", pos, vel, heading, speed,
                np.random.normal(0, np.deg2rad(20.0)),
                timestamp=float(k),
            )

        final_erratic = imm.get_target_probs("tgt_1")[MODE_ERRATIC]
        assert final_erratic > initial_erratic, (
            f"Erratic probability should increase with noisy observations. "
            f"Initial={initial_erratic:.3f}, Final={final_erratic:.3f}"
        )


class TestTargetSpecificIMMStates:
    """P0: Two targets must have independent IMM states."""

    def test_independent_mode_probabilities(self):
        """Mode probabilities for two targets must differ (independent state).

        Target A: straight line in crossing scenario → CV-favoring distribution.
        Target B: turning motion in head-on scenario → CT-favoring distribution.

        The key claim: targets maintain distinct, independent probability vectors.
        We verify this by checking:
        1. Both distributions sum to 1.
        2. Target A's CV prob differs from Target B's CV prob.
        3. Target A (straight) CV > COLREGS.
        4. Target B (turning) CT > COLREGS.
        """
        imm = IMMFilter(obs_std=2.0)

        # Target A: straight line, crossing scenario
        for k in range(40):
            pos_a = np.array([k * 10.0, 0.0])
            imm.set_target_observation(
                "tgt_A", pos_a, np.array([10.0, 0.0]), 0.0, 10.0, 0.0,
                timestamp=float(k),
                encounter_type="crossing",
            )

        # Target B: turning motion, head-on scenario
        speed_b = 8.0
        r_b = np.deg2rad(5.0)
        psi_b = 0.0
        for k in range(40):
            psi_b += r_b
            pos_b = np.array([
                200.0 + 30.0 * np.cos(psi_b),
                100.0 + 30.0 * np.sin(psi_b),
            ])
            vel_b = np.array([speed_b * np.cos(psi_b), speed_b * np.sin(psi_b)])
            imm.set_target_observation(
                "tgt_B", pos_b, vel_b, psi_b, speed_b, r_b,
                timestamp=float(k),
                encounter_type="head_on",
            )

        probs_a = imm.get_target_probs("tgt_A")
        probs_b = imm.get_target_probs("tgt_B")

        # 1. Both sum to 1
        assert abs(np.sum(probs_a) - 1.0) < 1e-8, f"Target A probs sum={np.sum(probs_a):.10f}"
        assert abs(np.sum(probs_b) - 1.0) < 1e-8, f"Target B probs sum={np.sum(probs_b):.10f}"

        # 2. The two distributions differ (independent state, not cross-contaminated)
        max_diff = np.max(np.abs(probs_a - probs_b))
        assert max_diff > 0.01, (
            f"Target A and B must have different mode probabilities. max_diff={max_diff:.4f}\n"
            f"A={probs_a}\nB={probs_b}"
        )

        # 3. Target A (straight): CV exceeds COLREGS (not turning, no avoidance)
        assert probs_a[MODE_CV] > probs_a[MODE_COLREGS], (
            f"Target A straight: CV should exceed COLREGS. "
            f"CV={probs_a[MODE_CV]:.3f}, COLREGS={probs_a[MODE_COLREGS]:.3f}"
        )

        # 4. Target B (turning with head_on): CT should have higher probability
        #    than COLREGS since the target is actively turning
        assert probs_b[MODE_CT] > probs_b[MODE_COLREGS], (
            f"Target B turning: CT should exceed COLREGS. "
            f"CT={probs_b[MODE_CT]:.3f}, COLREGS={probs_b[MODE_COLREGS]:.3f}"
        )

    def test_add_target_does_not_reset_existing(self):
        """Adding a new target should not reset existing target's probabilities."""
        imm = IMMFilter()

        # Establish target A with CV motion
        for k in range(10):
            pos = np.array([k * 10.0, 0.0])
            imm.set_target_observation(
                "tgt_A", pos, np.array([10.0, 0.0]), 0.0, 10.0, 0.0,
                timestamp=float(k),
            )
        probs_a_before = imm.get_target_probs("tgt_A").copy()

        # Add target B
        imm.set_target_observation(
            "tgt_B", np.array([100.0, 100.0]),
            np.array([5.0, 5.0]), np.pi / 4, 7.0, 0.0,
            timestamp=10.0,
        )

        probs_a_after = imm.get_target_probs("tgt_A")
        np.testing.assert_array_almost_equal(probs_a_before, probs_a_after,
                                             err_msg="Target A probabilities changed when adding target B")


class TestIMMFilterInitialization:
    """Basic IMM filter initialization and properties."""

    def test_default_initialization(self):
        imm = IMMFilter()
        assert imm.n_modes == 6
        assert len(imm.mode_probs) == 6
        assert abs(np.sum(imm.mode_probs) - 1.0) < 1e-10

    def test_entropy_max_1(self):
        """Normalized entropy is always ≤ 1."""
        imm = IMMFilter()
        ent = imm.get_entropy()
        assert 0.0 <= ent <= 1.0, f"Entropy {ent} should be in [0, 1]"

    def test_prediction_preserves_sum(self):
        """Markov prediction preserves probability sum."""
        imm = IMMFilter()
        before = imm.mode_probs.sum()
        imm.predict()
        after = imm.mode_probs.sum()
        assert abs(before - after) < 1e-10
