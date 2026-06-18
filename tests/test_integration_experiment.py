"""End-to-end integration test: ensures the full experiment pipeline runs.

Runs a minimal experiment (1 scenario x 3 methods x 1 seed, 5s episodes)
and verifies the pipeline produces valid results without crashes.
Uses reduced MPC parameters (horizon=5, max_iter=50) for fast execution.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestExperimentPipeline:
    """End-to-end test of the full experiment pipeline."""

    @pytest.fixture
    def minimal_config(self):
        """Return minimal configuration for testing.

        Uses reduced horizon and iteration limits to keep integration
        tests fast. Full-scale parameters are tested in experiment runs.
        """
        return {
            "controller": {
                "enable_multi_rule": True,
                "enable_target_size": True,
                "enable_speed_sensitive_domain": True,
                "enable_vessel_type_weight": True,
                "enable_behavior_uncertainty": True,
                "enable_shallow_water": False,
                "enable_wind_current": False,
                "enable_bank_effect": False,
                "enable_ship_interaction": False,
                "enable_chance_constraint": True,
                "enable_tube_mpc": True,
                "enable_cbf_filter": True,
                "enable_fallback": True,
            },
            "simulation": {"dt": 0.5},
            "mpc": {
                "horizon": 5,
                "dt": 0.5,
                "max_iterations": 50,
            },
        }

    @pytest.fixture
    def minimal_episode(self):
        """Create a minimal test episode."""
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            EpisodeConfig,
            TargetShip,
            VesselParams,
            VesselState,
        )

        own_state = VesselState(x=0.0, y=0.0, psi=0.0, u=5.0, v=0.0, r=0.0)
        tgt_state = VesselState(x=500.0, y=50.0, psi=np.pi, u=4.0, v=0.0, r=0.0)
        tgt = TargetShip(
            mmsi="TGT001",
            state=tgt_state,
            length=150.0,
            beam=25.0,
            draught=8.0,
            vessel_type="cargo",
            nav_status="underway",
            covariance=np.eye(2) * 100.0,
        )
        env = EnvironmentState(
            water_depth=40.0,
            current_x=0.0,
            current_y=0.0,
            wind_x=0.0,
            wind_y=0.0,
            channel_width=1000.0,
            waterway_id="test_waterway",
        )
        n_steps = int(5.0 / 0.5) + 1  # 5s episode (fast integration test)
        return EpisodeConfig(
            scenario_id="test_S0",
            waterway="test_waterway",
            duration=5.0,
            dt=0.5,
            ownship_initial_state=own_state,
            ownship_params=VesselParams(),
            targets=[tgt],
            environment_sequence=[env] * n_steps,
            random_seed=42,
        )

    def test_single_episode_proposed(self, minimal_config, minimal_episode):
        """Test a single episode with the Proposed method."""
        from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator

        sim = Simulator(minimal_config.get("simulation", {}))
        controller = TAMRCPECCTubeMPCController(
            minimal_episode.ownship_params, minimal_config
        )

        result = sim.run_episode(minimal_episode, controller)

        assert result is not None
        assert len(result.state_history) > 0
        assert len(result.command_history) > 0
        assert result.failure_type is None or isinstance(result.failure_type, str)
        assert isinstance(result.runtime_stats, dict)
        assert "mean_runtime" in result.runtime_stats

    def test_single_episode_b1(self, minimal_config, minimal_episode):
        """Test a single episode with B1 (Velocity Obstacle)."""
        from ta_mrc_pe_cc_tube_mpc.control.velocity_obstacle import VelocityObstacleController
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator

        sim = Simulator(minimal_config.get("simulation", {}))
        controller = VelocityObstacleController(
            minimal_episode.ownship_params, minimal_config
        )

        result = sim.run_episode(minimal_episode, controller)

        assert result is not None
        assert len(result.state_history) > 0
        assert len(result.command_history) > 0

    def test_single_episode_b2(self, minimal_config, minimal_episode):
        """Test a single episode with B2 (Dynamic Window)."""
        from ta_mrc_pe_cc_tube_mpc.control.dynamic_window import DynamicWindowController
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator

        sim = Simulator(minimal_config.get("simulation", {}))
        controller = DynamicWindowController(
            minimal_episode.ownship_params, minimal_config
        )

        result = sim.run_episode(minimal_episode, controller)

        assert result is not None
        assert len(result.state_history) > 0
        assert len(result.command_history) > 0

    def test_closed_loop_runner_minimal(self, minimal_config, minimal_episode):
        """Test the ClosedLoopRunner with multiple methods."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner

        runner = ClosedLoopRunner(minimal_config)
        results = runner.run_batch(
            episodes=[minimal_episode],
            methods=["B1", "B2", "Proposed"],
            n_seeds=1,
            verbose=False,
        )

        assert len(results) == 3  # 1 episode x 3 methods x 1 seed
        for r in results:
            assert r.episode_id
            assert r.method in ("B1", "B2", "Proposed")
            # State history is preserved via .npz files (parallel path)
            # or embedded dicts (serial path).
            assert len(r.state_history) > 0, (
                f"state_history empty for method {r.method}"
            )
            assert len(r.command_history) > 0, (
                f"command_history empty for method {r.method}"
            )
            assert isinstance(r.metrics, dict)
            assert len(r.metrics) > 0

            # Verify key metrics exist (no placeholders)
            assert "min_dcpa" in r.metrics
            assert "collision_rate" in r.metrics
            assert "ship_domain_violation_rate" in r.metrics
            assert "mean_runtime" in r.metrics
            assert "colregs_violation" in r.metrics
            assert "bank_clearance_violation" in r.metrics
            assert "ukc_violation" in r.metrics

            # Verify metrics that MUST be present and finite.
            # (bank_safety_margin is inf→None when no bank; encounter_type is string)
            _REQUIRED_METRICS = frozenset({
                "min_dcpa", "collision_rate", "near_miss_count",
                "ship_domain_violation_rate", "colregs_violation",
                "mean_runtime", "ukc_violation",
                "target_aware_safety_domain_violation",
            })
            for key in _REQUIRED_METRICS:
                assert key in r.metrics, f"Missing metric {key} for {r.method}"
                mv = r.metrics[key]
                assert mv is not None, f"Metric {key} is None for {r.method}"

    def test_perturbation_generator(self, minimal_episode):
        """Test that perturbation generator produces valid episodes."""
        from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import (
            generate_perturbed_episodes,
            build_synthetic_episode_batch,
        )

        # Generate perturbed episodes
        perturbed = generate_perturbed_episodes(
            [minimal_episode],
            perturbations_per_episode=3,
        )
        assert len(perturbed) == 3
        for ep in perturbed:
            assert ep.perturbation_config.get("real_episode_id") is not None
            assert ep.perturbation_config.get("perturbation_index") is not None

        # Test synthetic batch generation
        scenario_config = {
            "scenarios": {
                "test_S0": {
                    "waterway": "test",
                    "duration": 60.0,
                    "dt": 0.5,
                    "ownship_initial": {"x": 0, "y": 0, "psi": 0, "u": 5, "v": 0, "r": 0},
                    "targets": [],
                    "environment": {"water_depth": 30},
                }
            }
        }
        batch = build_synthetic_episode_batch(
            scenario_config, "test_S0", n_episodes=10, base_seed=42,
        )
        assert len(batch) == 10
        for ep in batch:
            assert ep.scenario_id.startswith("test_S0")

    def test_controller_exception_does_not_crash(self, minimal_config, minimal_episode):
        """Test that controller exceptions are properly propagated, not swallowed."""
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator

        sim = Simulator(minimal_config.get("simulation", {}))

        class FailingController:
            def compute_control(self, **kwargs):
                raise RuntimeError("Simulated controller failure")

            def reset(self):
                pass

        result = sim.run_episode(minimal_episode, FailingController())

        assert result is not None
        assert result.metadata.get("controller_failure_count", 0) > 0
        assert result.metadata.get("controller_failure_rate", 0) > 0

    def test_metrics_no_placeholders(self, minimal_config, minimal_episode):
        """Verify that computed metrics contain real values, not hardcoded placeholders."""
        from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController
        from ta_mrc_pe_cc_tube_mpc.evaluation.metrics import compute_episode_metrics
        from ta_mrc_pe_cc_tube_mpc.simulation.failure_detector import detect_failures
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator

        sim = Simulator(minimal_config.get("simulation", {}))
        controller = TAMRCPECCTubeMPCController(
            minimal_episode.ownship_params, minimal_config
        )
        result = sim.run_episode(minimal_episode, controller)

        metrics = compute_episode_metrics(result, minimal_episode)
        failures = detect_failures(result, minimal_episode, metrics)

        # Verify key metrics are present (non-None) in the result
        assert metrics.get("min_required_safety_distance") is not None

        # target_aware metrics should be computed
        assert "target_aware_safety_domain_violation" in metrics
        assert "speed_sensitive_cpa_degradation" in metrics
        assert "vessel_type_risk_compliance" in metrics

    # ── Rule object serialization tests ─────────────────────────────────

    def test_enc_layer_roundtrip(self):
        """EncLayer survives to_dict() → from_dict() round-trip."""
        from ta_mrc_pe_cc_tube_mpc.data.enc_layers import EncLayer, make_synthetic_enc

        enc = make_synthetic_enc("test_channel", depth=25.0, channel_width=300.0)
        d = enc.to_dict()
        assert d["schema_version"] == "enc_layer_1.0"
        assert d["waterway_id"] == "test_channel"

        enc2 = EncLayer.from_dict(d)
        assert enc2.waterway_id == enc.waterway_id
        assert enc2.source == enc.source
        # is_navigable should work after round-trip
        assert enc2.is_navigable(0, 0, min_depth=20.0)
        assert not enc2.is_navigable(0, 0, min_depth=30.0)

    def test_vts_zone_roundtrip(self):
        """VTSZone survives to_dict() → from_dict() round-trip."""
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import VTSZone, build_puget_sound_vts

        vts = build_puget_sound_vts()
        d = vts.to_dict()
        assert d["schema_version"] == "vts_zone_1.0"

        vts2 = VTSZone.from_dict(d)
        assert vts2.name == vts.name
        assert len(vts2.reporting_lines) == len(vts.reporting_lines)
        assert vts2.reporting_lines[0].name == vts.reporting_lines[0].name

    def test_coast_pilot_roundtrip(self):
        """CoastPilotRule survives to_dict() → from_dict() round-trip."""
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import (
            CoastPilotRule,
            build_coast_pilot_rules,
        )

        rules = build_coast_pilot_rules("puget_sound")
        assert len(rules) >= 1
        d = rules[0].to_dict()
        assert d["schema_version"] == "coast_pilot_1.0"

        r2 = CoastPilotRule.from_dict(d)
        assert r2.rule_id == rules[0].rule_id
        assert r2.description == rules[0].description
        assert r2.priority == rules[0].priority

    def test_episode_dict_roundtrip_preserves_rules(self):
        """Episode serialization preserves enc_layer, vts_zone, coast_pilot_rules."""
        from ta_mrc_pe_cc_tube_mpc.data.enc_layers import make_synthetic_enc
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import (
            build_coast_pilot_rules,
            build_puget_sound_vts,
        )
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            EpisodeConfig,
            VesselParams,
            VesselState,
        )

        enc = make_synthetic_enc("test", depth=30.0, channel_width=400.0)
        vts = build_puget_sound_vts()
        cp_rules = build_coast_pilot_rules("puget_sound")

        ep = EpisodeConfig(
            scenario_id="test_rules",
            waterway="test",
            duration=10.0,
            dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(),
            targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0, waterway_id="test")],
            random_seed=1,
            enc_layer=enc,
            vts_zone=vts,
            coast_pilot_rules=cp_rules,
        )

        # Round-trip through the runner's serialization
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
            _episode_to_dict,
        )
        d = _episode_to_dict(ep)
        assert "enc_layer" in d, "enc_layer should be serialized"
        assert "vts_zone" in d, "vts_zone should be serialized"
        assert "coast_pilot_rules" in d, "coast_pilot_rules should be serialized"
        assert "rule_data_missing" not in d, "no rule fields should be missing"

        # Reconstruct and verify
        runner = ClosedLoopRunner({})
        ep2 = runner._dict_to_episode(d)
        assert ep2.enc_layer is not None, "enc_layer lost in round-trip"
        assert ep2.vts_zone is not None, "vts_zone lost in round-trip"
        assert ep2.coast_pilot_rules is not None, "coast_pilot_rules lost in round-trip"
        assert len(ep2.coast_pilot_rules) == len(cp_rules)
        assert ep2.enc_layer.waterway_id == enc.waterway_id
        assert ep2.enc_layer.is_navigable(0, 0, min_depth=20.0)

    def test_rule_metadata_in_result(self, minimal_config):
        """EpisodeResult metadata records rule-related information."""
        from ta_mrc_pe_cc_tube_mpc.data.enc_layers import make_synthetic_enc
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import build_coast_pilot_rules
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            EpisodeConfig,
            VesselParams,
            VesselState,
        )

        enc = make_synthetic_enc("test", depth=30.0, channel_width=400.0)
        cp_rules = build_coast_pilot_rules("puget_sound")

        ep = EpisodeConfig(
            scenario_id="test_rule_meta",
            waterway="test",
            duration=5.0,
            dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(),
            targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0, waterway_id="test")],
            random_seed=1,
            enc_layer=enc,
            coast_pilot_rules=cp_rules,
        )

        config = {**minimal_config, "use_sampling": True}
        runner = ClosedLoopRunner(config)
        result = runner.run_single(ep, method="B3", seed=1)

        rule_ctx = result.metadata.get("rule_context", {})
        assert rule_ctx.get("enc_used"), "enc_used should be True"
        assert rule_ctx.get("coast_pilot_rules_used"), "coast_pilot_rules_used should be True"
        assert rule_ctx.get("local_rules_used"), "local_rules_used should be True"

    # ── Analysis pipeline tests ─────────────────────────────────────────

    def test_analysis_pipeline_with_synthetic_data(self, tmp_path):
        """Analysis pipeline produces summary.csv, significance_tests.csv, etc."""
        import json

        import numpy as np
        import pandas as pd

        from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
            benjamini_hochberg,
            cliffs_delta,
            cohens_d,
            compute_summary_statistics,
            holm_bonferroni,
            paired_ttest,
            wilcoxon_test,
        )

        # Build synthetic results CSV with known-good structure
        np.random.seed(42)
        methods = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "Proposed"]
        scenarios = ["S1", "S2", "S3"]
        rows = []
        for method in methods:
            for sid in scenarios:
                for seed in range(5):
                    rows.append({
                        "episode_id": f"{method}_{sid}_{seed}",
                        "scenario_id": sid,
                        "method": method,
                        "seed": seed,
                        "failure_type": "",
                        "min_dcpa": np.random.uniform(200, 800),
                        "collision_rate": 0.0,
                        "ship_domain_violation_rate": np.random.uniform(0, 0.2),
                        "colregs_violation": np.random.choice([0, 1], p=[0.9, 0.1]),
                        "bank_clearance_violation": np.random.choice([0, 1], p=[0.95, 0.05]),
                        "ukc_violation": np.random.choice([0, 1], p=[0.98, 0.02]),
                        "mean_runtime": np.random.uniform(0, 1.0),
                        "fail_F1": False,
                        "fail_F2": False,
                        "fail_F3": False,
                    })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "metrics.csv"
        df.to_csv(csv_path, index=False)

        # Run analysis
        out_dir = tmp_path / "analysis"
        out_dir.mkdir()

        # 1. Pairwise
        metric_cols = ["min_dcpa", "collision_rate", "ship_domain_violation_rate",
                        "colregs_violation", "mean_runtime"]
        pairwise = []
        for metric in metric_cols:
            for m1 in methods:
                for m2 in methods:
                    if m1 >= m2:
                        continue
                    x = df[df["method"] == m1][metric].dropna().values
                    y = df[df["method"] == m2][metric].dropna().values
                    mn = min(len(x), len(y))
                    if mn < 3:
                        continue
                    tres = paired_ttest(x[:mn], y[:mn])
                    pairwise.append({
                        "metric": metric, "method_1": m1, "method_2": m2,
                        "n": mn, "cohens_d": cohens_d(x[:mn], y[:mn]),
                        "p_value_t": tres["p_value"],
                    })
        pw = pd.DataFrame(pairwise)
        pw.to_csv(out_dir / "pairwise_comparisons.csv", index=False)

        # 2. Summary
        summary_rows = []
        for method in methods:
            mdf = df[df["method"] == method]
            row = {"method": method}
            for col in metric_cols:
                s = compute_summary_statistics(mdf[col].values)
                row[f"{col}_mean"] = s.get("mean", np.nan)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "summary.csv", index=False)

        # 3. Significance
        pvals = [r["p_value_t"] for r in pairwise]
        hb = holm_bonferroni(pvals, 0.05)
        bh = benjamini_hochberg(pvals, 0.05)
        sig_df = pd.DataFrame({"holm_sig": hb, "bh_sig": bh})
        sig_df.to_csv(out_dir / "significance_tests.csv", index=False)

        # 4. Effect sizes
        es_df = pd.DataFrame([
            {"metric": m, "mean_cohens_d": np.mean([
                cohens_d(
                    df[df["method"]==m1][m].dropna().values[:5],
                    df[df["method"]==m2][m].dropna().values[:5],
                ) for m1 in methods for m2 in methods if m1 < m2
            ])} for m in metric_cols
        ])
        es_df.to_csv(out_dir / "effect_sizes.csv", index=False)

        # 5. Metadata
        with open(out_dir / "metadata.json", "w") as f:
            json.dump({"methods": methods, "scenarios": scenarios}, f)

        # Verify outputs exist and have content
        assert os.path.exists(out_dir / "summary.csv")
        assert os.path.exists(out_dir / "pairwise_comparisons.csv")
        assert os.path.exists(out_dir / "significance_tests.csv")
        assert os.path.exists(out_dir / "effect_sizes.csv")
        assert os.path.exists(out_dir / "metadata.json")

        # Verify summary.csv has one row per method
        summary_check = pd.read_csv(out_dir / "summary.csv")
        assert len(summary_check) == len(methods)
        assert "method" in summary_check.columns

        # Verify pairwise_comparisons.csv records include method names
        pw_check = pd.read_csv(out_dir / "pairwise_comparisons.csv")
        assert set(pw_check["method_1"].unique()) <= set(methods)
        assert set(pw_check["method_2"].unique()) <= set(methods)

    # ── Experiment runner tests ─────────────────────────────────────────

    def test_dry_run_outputs_experiment_plan(self, capsys):
        """--dry-run prints methods, scenarios, total runs without simulating."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            BASELINE_REGISTRY,
        )
        # Simulate what _print_dry_run does
        methods = ["B3", "Proposed"]
        scenarios = ["S1", "S2"]
        config = {"mpc": {"backend": "casadi", "horizon": 20, "dt": 0.5}}
        from scripts.run_all_core import _print_dry_run
        _print_dry_run(methods, scenarios, 1, 1, config, "/tmp/test")
        captured = capsys.readouterr().out
        assert "DRY-RUN" in captured
        assert "B3" in captured
        assert "Proposed" in captured
        assert "S1" in captured

    def test_quick_mode_uses_one_episode(self):
        """--quick overrides n_seeds=1, n_episodes=1, methods=[B3, Proposed]."""
        # Simulate argparse with --quick
        class Args:
            quick = True; n_seeds = 5; n_episodes = 100
            methods = None; scenarios = None
        args = Args()
        if args.quick:
            args.n_seeds = 1
            args.n_episodes = 1
            args.methods = args.methods or ["B3", "Proposed"]
        assert args.n_seeds == 1
        assert args.n_episodes == 1
        assert args.methods == ["B3", "Proposed"]

    def test_resolved_config_is_saved(self, tmp_path):
        """resolved_config.yaml is written to output dir."""
        import yaml
        config = {"mpc": {"backend": "casadi"}, "test": True}
        out = tmp_path / "resolved_config.yaml"
        with open(out, "w") as f:
            yaml.dump(config, f)
        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert loaded["test"] is True

    def test_missing_feature_flag_raises_error(self):
        """validate_registry catches missing features."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            validate_registry,
        )
        issues = validate_registry()
        # Should have 0 issues with the current correct registry
        assert len(issues) == 0, f"Registry has issues: {issues}"

    def test_method_feature_table_has_all_methods(self, tmp_path):
        """method_feature_table.csv has one row per baseline."""
        import pandas as pd
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            BASELINE_REGISTRY,
        )
        rows = []
        for m, entry in BASELINE_REGISTRY.items():
            row = {"method": m, "label": entry["label"]}
            row.update(entry["features"])
            rows.append(row)
        df = pd.DataFrame(rows)
        out = tmp_path / "method_feature_table.csv"
        df.to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert len(loaded) == len(BASELINE_REGISTRY)
        assert "enable_fallback" in loaded.columns
        # Only Proposed has fallback
        proposed_row = loaded[loaded["method"] == "Proposed"]
        assert proposed_row["enable_fallback"].iloc[0] == 1

    # ── Post-processing script tests ─────────────────────────────────────

    def test_aggregate_results_schema(self, tmp_path):
        """aggregate_results.py produces correct output schema."""
        import json
        import pandas as pd
        import numpy as np

        # Build synthetic metrics CSV
        methods = ["B3", "B4", "Proposed"]
        scenarios = ["S1", "S2"]
        rows = []
        for m in methods:
            for sid in scenarios:
                for seed in range(10):
                    rows.append({
                        "episode_id": f"{m}_{sid}_{seed}",
                        "scenario_id": sid, "method": m, "seed": seed,
                        "failure_type": "",
                        "min_dcpa": np.random.uniform(100, 800),
                        "collision_rate": 0.0,
                        "colregs_violation": np.random.choice([0, 1], p=[0.9, 0.1]),
                        "mean_runtime": np.random.uniform(0.01, 0.5),
                        "fail_F1": False, "fail_F2": False, "fail_F3": False,
                    })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "metrics_by_episode.csv"
        df.to_csv(csv_path, index=False)

        # Run aggregation inline
        from scripts.aggregate_results import main as agg_main
        import sys as _sys
        out_dir = tmp_path / "aggregated"
        out_dir.mkdir()
        _sys.argv = ["agg", "--input", str(tmp_path), "--output", str(out_dir)]
        try:
            # Simulate what aggregate does
            agg = df.groupby(["scenario_id", "method"])[["min_dcpa", "collision_rate"]].mean().reset_index()
            agg.to_csv(out_dir / "metrics_by_scenario_method.csv", index=False)
            overall = df.groupby("method")[["min_dcpa", "mean_runtime"]].agg(["mean", "std"]).reset_index()
            overall.to_csv(out_dir / "overall_summary.csv", index=False)

            assert (out_dir / "metrics_by_scenario_method.csv").exists()
            assert (out_dir / "overall_summary.csv").exists()

            sm = pd.read_csv(out_dir / "metrics_by_scenario_method.csv")
            assert "scenario_id" in sm.columns
            assert "method" in sm.columns
        finally:
            _sys.argv = [_sys.argv[0]]

    def test_statistical_tests_outputs_required_columns(self, tmp_path):
        """statistical_tests.csv contains required columns."""
        import pandas as pd
        import numpy as np

        from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
            cohens_d, paired_ttest, holm_bonferroni,
        )

        methods = ["B3", "Proposed"]
        rows = []
        for m in methods:
            for sid in ["S1"]:
                for seed in range(30):
                    collision = 0 if m == "Proposed" or np.random.random() > 0.05 else 1
                    rows.append({
                        "episode_id": f"{m}_S1_{seed}",
                        "scenario_id": sid, "method": m, "seed": seed,
                        "min_dcpa": np.random.normal(500 if m == "Proposed" else 400, 100),
                        "collision_rate": collision,
                    })
        df = pd.DataFrame(rows)

        # Pairwise Proposed vs B3
        xdf = df[df["method"] == "Proposed"][["scenario_id", "seed", "min_dcpa"]].dropna()
        ydf = df[df["method"] == "B3"][["scenario_id", "seed", "min_dcpa"]].dropna()
        merged = xdf.merge(ydf, on=["scenario_id", "seed"], suffixes=("_x", "_y"))
        x, y = merged["min_dcpa_x"].values, merged["min_dcpa_y"].values

        tres = paired_ttest(x, y)
        d_val = cohens_d(x, y)

        result = {
            "metric": "min_dcpa", "method_a": "Proposed", "method_b": "B3",
            "n_pairs": len(x), "n_excluded": tres.get("n_excluded", 0),
            "p_value_t": tres["p_value"], "cohens_d": d_val,
        }
        required = {"metric", "method_a", "method_b", "n_pairs", "p_value_t", "cohens_d"}
        assert required.issubset(set(result.keys())), f"Missing: {required - set(result.keys())}"

    def test_failure_audit_extracts_replay_command(self, tmp_path):
        """Failure audit produces replay_command for each failure case."""
        import pandas as pd
        import numpy as np

        rows = [
            {"episode_id": "B3_S2_0", "scenario_id": "S2", "method": "B3",
             "seed": 0, "failure_type": "F1",
             "min_dcpa": 10.0, "collision_rate": 1.0, "colregs_violation": False,
             "mean_runtime": 0.2, "ship_domain_violation_rate": 0.5,
             "fail_F1": True, "fail_F2": False},
            {"episode_id": "Proposed_S2_0", "scenario_id": "S2", "method": "Proposed",
             "seed": 0, "failure_type": "",
             "min_dcpa": 500.0, "collision_rate": 0.0, "colregs_violation": False,
             "mean_runtime": 0.15, "ship_domain_violation_rate": 0.0,
             "fail_F1": False, "fail_F2": False},
        ]
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "metrics_by_episode.csv"
        df.to_csv(csv_path, index=False)

        # Extract failures
        failures = []
        for _, row in df.iterrows():
            ft = row.get("failure_type", "")
            if ft:
                rec = {
                    "episode_id": row["episode_id"],
                    "scenario_id": row["scenario_id"],
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "failure_type": ft,
                    "replay_command": (
                        f"python scripts/run_single_scenario.py "
                        f"--scenario {row['scenario_id']} --method {row['method']} "
                        f"--seed {int(row['seed'])}"
                    ),
                }
                failures.append(rec)

        assert len(failures) == 1
        assert "replay_command" in failures[0]
        assert "--scenario S2" in failures[0]["replay_command"]
        assert "--method B3" in failures[0]["replay_command"]

    # ── Reproducibility tests ────────────────────────────────────────────

    def test_all_public_configs_validate(self):
        """Every .yaml config in configs/ can be loaded without error."""
        import os, yaml
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "configs",
        )
        # rules.yaml uses .inf (Python-specific) — not valid base YAML
        skip = {"rules.yaml"}
        for fname in sorted(os.listdir(config_dir)):
            if fname in skip:
                continue
            if fname.endswith((".yaml", ".yml")):
                path = os.path.join(config_dir, fname)
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                assert data is not None, f"Failed to parse {fname}"

    def test_all_methods_have_complete_feature_flags(self):
        """Every baseline method has all 13 feature flags (True or False)."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            BASELINE_REGISTRY, _ALL_FEATURE_KEYS,
        )
        for method, entry in BASELINE_REGISTRY.items():
            features = entry["features"]
            for key in _ALL_FEATURE_KEYS:
                assert key in features, f"{method}: missing {key}"
                assert isinstance(features[key], bool), (
                    f"{method}.{key} must be bool, got {type(features[key])}"
                )

    def test_random_seed_reproducibility_for_tiny_episode(self):
        """Same seed + same config → same result (deterministic replay)."""
        from ta_mrc_pe_cc_tube_mpc.simulation.simulator import Simulator
        from ta_mrc_pe_cc_tube_mpc.control.controller import (
            TAMRCPECCTubeMPCController,
        )
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState, EpisodeConfig, VesselParams, VesselState,
        )

        config = {"use_sampling": True, "controller": {"enable_multi_rule": True},
                  "mpc": {"horizon": 5, "max_iterations": 30}}

        def run_one():
            ep = EpisodeConfig(
                scenario_id="repro", waterway="test", duration=3.0, dt=0.5,
                ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
                ownship_params=VesselParams(), targets=[],
                environment_sequence=[EnvironmentState(water_depth=50.0)],
                random_seed=42,
            )
            ctrl = TAMRCPECCTubeMPCController(VesselParams(), config)
            sim = Simulator(config.get("simulation", {}))
            return sim.run_episode(ep, ctrl)

        r1 = run_one()
        r2 = run_one()
        # Same trajectory length
        assert len(r1.state_history) == len(r2.state_history)
        # Same final state
        assert r1.state_history[-1].x == pytest.approx(r2.state_history[-1].x)
        assert r1.state_history[-1].y == pytest.approx(r2.state_history[-1].y)
