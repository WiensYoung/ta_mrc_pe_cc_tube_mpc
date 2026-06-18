"""Closed-loop experiment runner with parallelization and checkpoint/resume.

Orchestrates running multiple episodes across scenarios, methods, and seeds.
Supports:
- Multiprocessing parallel execution
- Checkpoint/resume for long-running experiments
- Progress persistence to disk
"""

import hashlib
import json
import multiprocessing
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..control.controller import TAMRCPECCTubeMPCController
from ..control.dynamic_window import DynamicWindowController
from ..control.velocity_obstacle import VelocityObstacleController
from ..types import EpisodeConfig, EpisodeResult, VesselParams
from ..utils.io_utils import load_json, save_json
from ..utils.logging_utils import get_logger
from .simulator import Simulator

logger = get_logger(__name__)

# Checkpoint file template
CHECKPOINT_SUFFIX = "_checkpoint.json"


def _pickle_result(result: EpisodeResult, trajectory_dir: str = None) -> dict:
    """Serialize EpisodeResult to a JSON-safe dict for checkpointing.

    When *trajectory_dir* is given, state/command/target histories are
    saved to ``{trajectory_dir}/{episode_id}.npz`` and the returned dict
    records the file path in ``trajectory_file``.  When *trajectory_dir*
    is None (serial / small-batch mode), the full histories are included
    directly under ``state_history``, ``command_history``, and
    ``target_histories`` keys (as picklable lists of dataclass dicts).
    """
    import dataclasses as _dc

    # Track metrics that were NaN/Inf (become None via _safe_float).
    # Only numeric values go through _safe_float; strings, bools, etc.
    # are passed through unchanged.
    invalid_metrics = []
    safe_metrics = {}
    for k, v in result.metrics.items():
        if isinstance(v, (int, float, bool)) or v is None:
            fv = _safe_float(v)
            safe_metrics[k] = fv
            if fv is None and v is not None and not (isinstance(v, float) and np.isnan(v)):
                invalid_metrics.append(k)
        else:
            # Non-numeric (strings, lists, dicts) — keep as-is
            safe_metrics[k] = v

    d = {
        "episode_id": result.episode_id,
        "scenario_id": result.scenario_id,
        "method": result.method,
        "seed": result.seed,
        "failure_type": result.failure_type or "",
        "failure_flags": {k: v for k, v in result.failure_flags.items()},
        "metrics": safe_metrics,
        "runtime_stats": {k: _safe_float(v) for k, v in result.runtime_stats.items()},
        "metadata": _pickle_metadata(result.metadata),
    }
    if invalid_metrics:
        d["invalid_metric_fields"] = invalid_metrics

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
        traj_path = os.path.join(trajectory_dir, f"{result.episode_id}.npz")
        _save_trajectory_npz(traj_path, result)
        d["trajectory_file"] = traj_path
        d["trajectory_format"] = "npz"
    else:
        # Keep in-memory for small / integration runs
        d["state_history"] = [_dc.asdict(s) for s in result.state_history]
        d["command_history"] = [_dc.asdict(c) for c in result.command_history]
        d["target_histories"] = [
            [_dc.asdict(ts) for ts in th] for th in result.target_histories
        ]
        d["trajectory_format"] = "inline"

    # ── Shape metadata (always present) ─────────────────────────────────
    try:
        from .trajectory_io import trajectory_shape_meta
        d["_trajectory_meta"] = trajectory_shape_meta(result)
    except ImportError:
        d["_trajectory_meta"] = {"warning": "trajectory_io module not available"}
    return d


def _save_trajectory_npz(path: str, result: EpisodeResult):
    """Save state/command/target histories to a compressed .npz file."""
    state_arrays = {}
    if result.state_history:
        arr = np.array([[s.x, s.y, s.psi, s.u, s.v, s.r, s.timestamp]
                        for s in result.state_history])
        state_arrays["own_state"] = arr
    if result.command_history:
        arr = np.array([[c.rudder, c.propeller, int(c.cbf_modified), int(c.fallback_active)]
                        for c in result.command_history])
        state_arrays["own_command"] = arr
    for i, th in enumerate(result.target_histories):
        if th:
            arr = np.array([[s.x, s.y, s.psi, s.u, s.v, s.r, s.timestamp]
                            for s in th])
            state_arrays[f"target_{i}_state"] = arr
    np.savez_compressed(path, **state_arrays)


def _load_trajectory_npz(path: str) -> dict:
    """Load trajectories from .npz back into a dict of numpy arrays."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _pickle_metadata(meta: dict) -> dict:
    """Strip non-JSON-serializable values from metadata dict."""
    safe = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, dict):
            safe[k] = _pickle_metadata(v)
        elif isinstance(v, (list, tuple)):
            safe[k] = [
                _pickle_metadata(x) if isinstance(x, dict) else
                str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
                for x in v
            ]
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        else:
            safe[k] = str(v)
    return safe


def _safe_float(v, allow_nan: bool = True):
    """Convert value to a JSON-safe float.

    Args:
        v: Input value.
        allow_nan: When True, NaN → None (JSON null), Inf → None with
            magnitude preserved in metadata.  When False (strict mode),
            NaN/Inf raise ValueError.

    Returns:
        Float value, or None for NaN/Inf when *allow_nan* is True.

    .. important::
        NaN and Inf are **never** silently converted to 0.0.  The old
        behaviour (``return 0.0`` for NaN/Inf) was a statistical error
        that distorted safety metrics and hypothesis tests.
    """
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        fv = float(v)
        if np.isnan(fv):
            if allow_nan:
                return None   # JSON null — detectable, not silently zero
            raise ValueError("NaN in metric value with allow_nan=False")
        if np.isinf(fv):
            if allow_nan:
                return None   # preserve as null with sign in metadata
            raise ValueError("Inf in metric value with allow_nan=False")
        return fv
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _run_single_worker(args):
    """Worker function for multiprocessing. Unpacks args and runs a single episode.

    Args:
        args: (task_id, ep_dict, method, controller_config, seed, config_dict)

    Returns:
        (task_id, EpisodeResult) tuple for task tracking in parallel mode.
    """
    # Ensure package is importable in spawned subprocesses (Windows default)
    import sys as _sys
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    (task_id, ep_dict, method, controller_config, seed, config_dict, trajectory_dir) = args

    # Reconstruct episode from dict using _dict_to_episode, which properly
    # restores enc_layer, vts_zone, and coast_pilot_rules that were
    # serialised by _episode_to_dict. The old manual construction silently
    # dropped all three, disabling ENC/VTS/CoastPilot in parallel mode.
    runner = ClosedLoopRunner(config_dict)
    ep = runner._dict_to_episode(ep_dict)

    try:
        result = runner.run_single(ep, method, controller_config, seed)
        # Save heavy trajectory data to .npz file, return only the path + scalars.
        # This keeps the multiprocessing pipe payload small (~2 KB) while
        # preserving trajectories for later replotting / failure analysis.
        return (task_id, _pickle_result(result, trajectory_dir=trajectory_dir))
    except BaseException as exc:
        # Catch BaseException (not just Exception) to handle
        # SystemExit, KeyboardInterrupt, and C-level crashes that
        # manifest as RuntimeError in the worker process.
        import traceback as _tb
        # Avoid re-serializing the full traceback object — format it to string
        tb_str = _tb.format_exc() if isinstance(exc, Exception) else str(exc)
        return (task_id, {
            "episode_id": ep_dict.get("scenario_id", "unknown"),
            "scenario_id": ep_dict.get("scenario_id", "unknown"),
            "method": method,
            "seed": seed,
            "failure_type": "worker_crash",
            "failure_flags": {},
            "metrics": {},
            "runtime_stats": {"worker_error": str(exc), "traceback": tb_str},
            "metadata": {},
        })


def _reconstruct_targets(targets_data: list) -> list:
    from ..types import TargetShip, VesselState
    result = []
    for td in targets_data:
        state_data = td.get("state", {})
        state = VesselState(**state_data) if state_data else VesselState()
        cov_data = td.get("covariance")
        covariance = np.array(cov_data) if cov_data is not None else None
        tgt = TargetShip(
            mmsi=td.get("mmsi", ""),
            state=state,
            length=td.get("length", 100.0),
            beam=td.get("beam", 15.0),
            draught=td.get("draught", 5.0),
            vessel_type=td.get("vessel_type", "cargo"),
            nav_status=td.get("nav_status", "underway"),
            cog=td.get("cog", 0.0),
            sog=td.get("sog", 0.0),
            heading=td.get("heading"),
            covariance=covariance,
            ais_quality=td.get("ais_quality", {}),
        )
        result.append(tgt)
    return result


def _reconstruct_env_seq(env_data: list) -> list:
    from ..types import EnvironmentState
    result = []
    for ed in env_data:
        result.append(EnvironmentState(**ed))
    return result


def _episode_to_dict(ep: EpisodeConfig) -> dict:
    """Serialize EpisodeConfig to a JSON-safe dict for multiprocessing.

    """
    import dataclasses
    import warnings as _w

    d = {
        "scenario_id": ep.scenario_id,
        "waterway": ep.waterway,
        "duration": ep.duration,
        "dt": ep.dt,
        "ownship_initial_state": dataclasses.asdict(ep.ownship_initial_state),
        "ownship_params": dataclasses.asdict(ep.ownship_params),
        "targets": [_target_to_dict(t) for t in ep.targets],
        "environment_sequence": [dataclasses.asdict(e) for e in ep.environment_sequence],
        "random_seed": ep.random_seed,
        "perturbation_config": ep.perturbation_config,
    }

    # ── Rule objects — serialise via to_dict() ─────────────────────────
    missing_fields = []
    if ep.enc_layer is not None:
        try:
            d["enc_layer"] = ep.enc_layer.to_dict()
        except Exception as exc:
            _w.warn(f"Cannot serialise enc_layer: {exc}", RuntimeWarning)
            missing_fields.append("enc_layer")
    if ep.vts_zone is not None:
        try:
            d["vts_zone"] = ep.vts_zone.to_dict()
        except Exception as exc:
            _w.warn(f"Cannot serialise vts_zone: {exc}", RuntimeWarning)
            missing_fields.append("vts_zone")
    if ep.coast_pilot_rules:
        try:
            d["coast_pilot_rules"] = [r.to_dict() for r in ep.coast_pilot_rules]
        except Exception as exc:
            _w.warn(f"Cannot serialise coast_pilot_rules: {exc}", RuntimeWarning)
            missing_fields.append("coast_pilot_rules")

    if missing_fields:
        d["rule_data_missing"] = missing_fields
    return d


def _target_to_dict(tgt) -> dict:
    import dataclasses
    cov = None
    if tgt.covariance is not None:
        cov = tgt.covariance.tolist() if hasattr(tgt.covariance, "tolist") else None
    return {
        "mmsi": tgt.mmsi,
        "state": dataclasses.asdict(tgt.state),
        "length": tgt.length,
        "beam": tgt.beam,
        "draught": tgt.draught,
        "vessel_type": tgt.vessel_type,
        "nav_status": tgt.nav_status,
        "cog": tgt.cog,
        "sog": tgt.sog,
        "heading": tgt.heading,
        "covariance": cov,
        "ais_quality": tgt.ais_quality,
    }


class ClosedLoopRunner:
    """Runs closed-loop simulation experiments with parallelization support."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.sim = Simulator(config.get("simulation", {}))
        self._csv_saved_count = 0  # track incremental CSV saves to prevent duplicates

    def run_single(
        self,
        episode: EpisodeConfig,
        method: str = "Proposed",
        controller_config: dict = None,
        seed: int = None,
    ) -> EpisodeResult:
        """Run a single episode with a specific method."""
        if seed is not None:
            episode.random_seed = seed
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(episode.random_seed)

        controller = self._build_controller(method, episode.ownship_params, controller_config)

        result = self.sim.run_episode(episode, controller, rng)
        result.method = method

        from ..evaluation.metrics import compute_episode_metrics
        from ..evaluation.safety_evaluator import evaluate_episode_safety
        from .failure_detector import detect_failures

        # Legacy metrics (fast, broad coverage)
        metrics = compute_episode_metrics(result, episode)
        # Unified safety evaluator — single source of truth for paper metrics
        safety = evaluate_episode_safety(
            own_states=result.state_history,
            target_histories=result.target_histories,
            own_params=episode.ownship_params,
            target_params_list=episode.targets,
            command_history=result.command_history,
            metadata=result.metadata,
            env_sequence=episode.environment_sequence,
        )
        # Merge: legacy metrics as base, safety evaluator fields override where present.
        # L3 fix (2026-06-10): Detect and log conflicts where legacy and safety
        # evaluator disagree significantly (threshold >1% relative difference).
        safety_dict = safety.to_dict()
        _conflict_keys = []
        for k in safety_dict:
            if k in metrics and isinstance(metrics[k], (int, float)) and isinstance(safety_dict[k], (int, float)):
                legacy_v = float(metrics[k])
                safety_v = float(safety_dict[k])
                if legacy_v != 0.0 and abs(legacy_v - safety_v) / max(abs(legacy_v), 1e-6) > 0.01:
                    _conflict_keys.append((k, legacy_v, safety_v))
        if _conflict_keys:
            import logging
            _log = logging.getLogger(__name__)
            _log.debug(
                "Metric conflict: safety evaluator overrode legacy metrics for %s. "
                "Conflicts (legacy→safety): %s",
                episode.scenario_id,
                [(k, f"{lv:.4f}→{sv:.4f}") for k, lv, sv in _conflict_keys[:5]],
            )
        metrics.update(safety_dict)
        # Replace NaN/Inf with None (JSON null), preserving data quality signal.
        # NEVER silently convert to 0.0 — that was a statistical error that
        # distorted safety metrics and hypothesis tests.
        invalid_fields = []
        for k, v in list(metrics.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                metrics[k] = None
                invalid_fields.append(k)
        if invalid_fields:
            result.metadata.setdefault("invalid_metric_fields", []).extend(invalid_fields)

        failures = detect_failures(result, episode, metrics)
        result.metrics = metrics
        result.failure_flags = failures["flags"]
        result.failure_type = failures["primary_type"]

        # ── Rule-object metadata ─────────────────────────────────────────
        rule_meta = {
            "enc_used": episode.enc_layer is not None,
            "vts_used": episode.vts_zone is not None,
            "coast_pilot_rules_used": bool(episode.coast_pilot_rules),
            "local_rules_used": (
                episode.vts_zone is not None or bool(episode.coast_pilot_rules)
            ),
            "rule_priority_levels_used": [],
            "missing_rule_fields": episode.perturbation_config.get("rule_data_missing", []),
        }
        result.metadata["rule_context"] = rule_meta

        # ── Failure status struct (mandatory for every episode) ──────────
        result.metadata["failure_status"] = {
            "success": result.failure_type is None,
            "failure_reason": result.failure_type,
            "invalid_metrics": result.metadata.get("invalid_metric_fields", []),
            "collision": result.failure_flags.get("F1", False),
            "grounding": result.failure_flags.get("F5", False),
            "infeasible": result.failure_flags.get("F7", False),
            "solver_failure": result.failure_flags.get("F7", False) or result.failure_flags.get("F8", False),
            "timeout": result.failure_flags.get("F10", False),
            "fallback_triggered": result.failure_flags.get("F9", False),
        }

        return result

    def run_batch(
        self,
        episodes: list[EpisodeConfig],
        methods: list[str],
        n_seeds: int = 1,
        seed_offset: int = 0,
        controller_configs: dict = None,
        verbose: bool = True,
        n_workers: int = 4,
        checkpoint_path: str = None,
        output_csv: str = None,
        csv_save_interval: int = 50,
    ) -> list[EpisodeResult]:
        """Run a batch of experiments with optional parallelization.

        Args:
            episodes: List of episode configurations.
            methods: List of method identifiers.
            n_seeds: Number of random seeds per (episode, method).
            seed_offset: Starting seed offset.
            controller_configs: Per-method controller config overrides.
            verbose: Show progress bar.
            n_workers: Number of parallel workers (1 = serial).
            checkpoint_path: Path to checkpoint file for resume support.
            output_csv: Path to save incremental CSV during execution.
            csv_save_interval: Save CSV every N completed tasks.

        Returns:
            List of EpisodeResult objects.
        """
        # Use Ray for distributed parallel execution when available
        if n_workers > 1:
            try:
                from .ray_runner import RayRunner
                import ray as _ray
                if _ray.is_initialized():
                    if verbose:
                        print(f"  Using Ray distributed runner ({n_workers} workers)")
                    ray_runner = RayRunner(self.config)
                    return ray_runner.run_batch(
                        episodes=episodes, methods=methods,
                        n_seeds=n_seeds, seed_offset=seed_offset,
                        controller_configs=controller_configs,
                        verbose=verbose, n_workers=n_workers,
                        checkpoint_path=checkpoint_path,
                        output_csv=output_csv,
                        csv_save_interval=csv_save_interval,
                    )
            except ImportError:
                pass  # Ray not installed, fall through to multiprocessing
            except Exception as e:
                logger.warning("Ray runner failed (%s), falling back to multiprocessing", e)

        # Build task list — seeds use a deterministic hash of (ep_idx, method, s)
        # so they are independent of task enumeration order and reproducible
        # across runs with different method lists or episode counts.
        tasks = []
        task_map = {}  # task_id -> (ep_idx, method, seed_idx)
        task_id = 0

        for ep_idx, ep in enumerate(episodes):
            for method in methods:
                for s in range(n_seeds):
                    method_hash = int(hashlib.md5(method.encode()).hexdigest()[:4], 16)
                    seed = seed_offset + ep.random_seed + s * 1000 + method_hash % 1000
                    cfg = controller_configs.get(method, {}) if controller_configs else {}
                    tasks.append((_episode_to_dict(ep), method, cfg, seed, self.config))
                    task_map[task_id] = (ep_idx, method, s)
                    task_id += 1

        # Check for existing checkpoint
        completed_tasks = set()
        existing_results = []
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = load_json(checkpoint_path)
            completed_tasks = set(ckpt.get("completed_task_ids", []))
            existing_results = ckpt.get("results", [])
            if verbose:
                print(f"Resuming from checkpoint: {len(completed_tasks)}/{len(tasks)} tasks completed")

        # Filter out completed tasks
        remaining_tasks = [(i, t) for i, t in enumerate(tasks) if i not in completed_tasks]

        if verbose and remaining_tasks:
            print(f"Remaining tasks: {len(remaining_tasks)}")

        results = []
        if n_workers > 1 and len(remaining_tasks) > 1:
            results = self._run_parallel(
                remaining_tasks, n_workers, verbose,
                checkpoint_path, completed_tasks, existing_results,
                output_csv=output_csv, csv_save_interval=csv_save_interval,
            )
        else:
            results = self._run_serial(
                remaining_tasks, verbose,
                checkpoint_path, completed_tasks, existing_results,
                output_csv=output_csv, csv_save_interval=csv_save_interval,
            )

        # Final incremental save (write mode = overwrite with all results)
        if output_csv and results:
            self._csv_saved_count = _incremental_csv_save(results, output_csv, mode="write")

        return results

    def _run_serial(
        self,
        tasks: list,
        verbose: bool,
        checkpoint_path: str = None,
        completed_tasks: set = None,
        existing_results: list = None,
        output_csv: str = None,
        csv_save_interval: int = 50,
    ) -> list[EpisodeResult]:
        """Run tasks serially with checkpoint support."""
        completed = completed_tasks or set()
        # existing_results may be pickled dicts (from checkpoint) or EpisodeResult objects
        results = list(existing_results or [])

        iterator = tqdm(tasks, desc="Running experiments") if verbose else tasks

        for _enum_idx, (task_id, (ep_dict, method, cfg, seed, _)) in enumerate(iterator):
            try:
                ep = self._dict_to_episode(ep_dict)
                result = self.run_single(ep, method, cfg, seed)
                # Serial / small-batch: keep full trajectories in memory
                results.append(_pickle_result(result, trajectory_dir=None))
                completed.add(task_id)

                # Checkpoint every 10 runs
                if checkpoint_path and len(results) % 10 == 0:
                    self._save_checkpoint(
                        checkpoint_path, list(completed), results,
                    )
                # Incremental CSV save — only unsaved rows
                if output_csv and (len(results) - self._csv_saved_count) >= csv_save_interval:
                    self._csv_saved_count = _incremental_csv_save(
                        results, output_csv, mode="append",
                        saved_count=self._csv_saved_count)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                # BUG #3 fix (2026-06-10): 'method' may be undefined if tuple
                # unpacking itself raises. Safeguard with getattr/default.
                safe_method = ep_dict.get("method", "unknown") if isinstance(ep_dict, dict) else "unknown"
                # Record failed task so we don't lose it on resume
                results.append({
                    "episode_id": ep_dict.get("scenario_id", "unknown") if isinstance(ep_dict, dict) else "unknown",
                    "scenario_id": ep_dict.get("scenario_id", "unknown") if isinstance(ep_dict, dict) else "unknown",
                    "method": safe_method,
                    "seed": seed if 'seed' in dir() else 0,
                    "failure_type": "serial_worker_crash",
                    "failure_flags": {},
                    "metrics": {},
                    "runtime_stats": {"error": str(e)},
                    "metadata": {},
                })
                completed.add(task_id)

        # Final checkpoint
        if checkpoint_path:
            self._save_checkpoint(
                checkpoint_path, list(completed), results,
            )

        return self._reconstruct_results(results)

    def _run_parallel(
        self,
        tasks: list,
        n_workers: int,
        verbose: bool,
        checkpoint_path: str = None,
        completed_tasks: set = None,
        existing_results: list = None,
        output_csv: str = None,
        csv_save_interval: int = 50,
    ) -> list[EpisodeResult]:
        """Run tasks in parallel using multiprocessing.

        IMPORTANT: Workers return *pickled dicts* (via _pickle_result), not raw
        EpisodeResult objects. This keeps the pipe payload small (~2 KB per
        result instead of ~1+ MB) and prevents BrokenPipeError from buffer
        overflow. Results are reconstructed at the end via _reconstruct_results.
        """
        completed = set(completed_tasks or set())
        # existing_results from checkpoint are already pickled dicts
        results: list = list(existing_results or [])
        actual_workers = min(n_workers, len(tasks), multiprocessing.cpu_count())

        # Trajectory directory for parallel workers — always created so
        # heavy state/command/target histories are preserved to disk.
        traj_dir = os.path.join(
            os.path.dirname(output_csv) if output_csv else "results",
            "trajectories",
        )
        os.makedirs(traj_dir, exist_ok=True)

        # Include task_id in worker input so we can track completions.
        # trajectory_dir is packed so workers can save heavy history to .npz.
        task_data = [(task_id, ep_dict, method, cfg, seed, config_dict, traj_dir)
                     for (task_id, (ep_dict, method, cfg, seed, config_dict)) in tasks]

        pool = None
        try:
            # Use 'spawn' start context to avoid fork-after-pthreads deadlock
            # when JAX has been imported in the parent process.
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(
                processes=actual_workers,
                maxtasksperchild=50,  # prevent memory leak from long-lived workers
            )
            if verbose:
                results_iter = tqdm(
                    pool.imap_unordered(_run_single_worker, task_data),
                    total=len(task_data),
                    desc="Running experiments (parallel)",
                )
            else:
                results_iter = pool.imap_unordered(_run_single_worker, task_data)

            completed_count = len(results)
            for task_id, ep_result in results_iter:
                # Worker returns either EpisodeResult or fallback dict.
                # Always convert to pickled dict for consistent storage.
                if isinstance(ep_result, dict):
                    results.append(ep_result)
                else:
                    results.append(_pickle_result(ep_result))
                completed.add(task_id)
                completed_count += 1

                if checkpoint_path and completed_count % 20 == 0:
                    self._save_checkpoint(
                        checkpoint_path, sorted(completed), results,
                    )
                # Incremental CSV save — only unsaved rows
                if output_csv and completed_count % csv_save_interval == 0:
                    self._csv_saved_count = _incremental_csv_save(
                        results, output_csv, mode="append",
                        saved_count=self._csv_saved_count)
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt received — gracefully shutting down pool "
                "and saving partial results."
            )
            # Close the pool gracefully first (wait for in-flight tasks),
            # then save. This avoids the race where pool.__exit__->terminate()
            # kills workers mid-transmission, causing BrokenPipeError.
            if pool is not None:
                pool.close()
                pool.join()
            if checkpoint_path:
                self._save_checkpoint(
                    checkpoint_path, sorted(completed), results,
                )
            if output_csv and results:
                self._csv_saved_count = _incremental_csv_save(
                    results, output_csv, mode="write")
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Final checkpoint — only save ACTUALLY completed task IDs
        if checkpoint_path:
            self._save_checkpoint(
                checkpoint_path, sorted(completed), results,
            )

        return self._reconstruct_results(results)

    def _save_checkpoint(self, path: str, completed_ids: list, results: list):
        """Save experiment checkpoint to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_json({
            "completed_task_ids": completed_ids,
            "results": results,
            "timestamp": time.time(),
        }, path)

    def _dict_to_episode(self, ep_dict: dict) -> EpisodeConfig:
        """Reconstruct EpisodeConfig from dict, including rule objects."""
        from ..types import VesselState, VesselParams

        # ── Rebuild rule objects from serialised dicts ──────────────────
        enc_layer = None
        vts_zone = None
        coast_pilot_rules = None

        enc_data = ep_dict.get("enc_layer")
        if enc_data:
            try:
                from ..data.enc_layers import EncLayer
                enc_layer = EncLayer.from_dict(enc_data)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Failed to deserialise enc_layer for episode "
                    f"'{ep_dict.get('scenario_id', '?')}': {exc}. "
                    f"ENC navigability data will be missing.",
                    RuntimeWarning,
                )

        vts_data = ep_dict.get("vts_zone")
        if vts_data:
            try:
                from ..data.vts_rules import VTSZone
                vts_zone = VTSZone.from_dict(vts_data)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Failed to deserialise vts_zone for episode "
                    f"'{ep_dict.get('scenario_id', '?')}': {exc}. "
                    f"VTS zone rules will be missing.",
                    RuntimeWarning,
                )

        cp_data = ep_dict.get("coast_pilot_rules")
        if cp_data:
            try:
                from ..data.vts_rules import CoastPilotRule
                coast_pilot_rules = [CoastPilotRule.from_dict(r) for r in cp_data]
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Failed to deserialise coast_pilot_rules for episode "
                    f"'{ep_dict.get('scenario_id', '?')}': {exc}. "
                    f"Coast pilot rules will be missing.",
                    RuntimeWarning,
                )

        return EpisodeConfig(
            scenario_id=ep_dict["scenario_id"],
            waterway=ep_dict.get("waterway", ""),
            duration=ep_dict["duration"],
            dt=ep_dict["dt"],
            ownship_initial_state=VesselState(**ep_dict["ownship_initial_state"]),
            ownship_params=VesselParams(**ep_dict.get("ownship_params", {})),
            targets=_reconstruct_targets(ep_dict.get("targets", [])),
            environment_sequence=_reconstruct_env_seq(ep_dict.get("environment_sequence", [])),
            random_seed=ep_dict["random_seed"],
            perturbation_config=ep_dict.get("perturbation_config", {}),
            enc_layer=enc_layer,
            vts_zone=vts_zone,
            coast_pilot_rules=coast_pilot_rules,
        )

    def _reconstruct_results(self, pickled_results: list) -> list[EpisodeResult]:
        """Reconstruct EpisodeResult list from pickled dicts or raw objects.

        When a ``trajectory_file`` key is present the histories are loaded
        from the .npz file.  When ``state_history`` / ``command_history``
        are embedded directly (serial / small-batch path) they are
        reconstructed from the dataclass dicts.
        """
        from ..types import ControlCommand, VesselState
        import dataclasses as _dc

        results = []
        for pr in pickled_results:
            if isinstance(pr, EpisodeResult):
                results.append(pr)
                continue

            # Load trajectories
            state_history = []
            command_history = []
            target_histories = []

            traj_file = pr.get("trajectory_file")
            if traj_file and os.path.exists(traj_file):
                data = _load_trajectory_npz(traj_file)
                if "own_state" in data:
                    arr = data["own_state"]
                    state_history = [
                        VesselState(x=float(r[0]), y=float(r[1]), psi=float(r[2]),
                                    u=float(r[3]), v=float(r[4]), r=float(r[5]),
                                    timestamp=float(r[6]))
                        for r in arr
                    ]
                if "own_command" in data:
                    arr = data["own_command"]
                    command_history = [
                        ControlCommand(rudder=float(r[0]), propeller=float(r[1]),
                                       cbf_modified=bool(r[2]), fallback_active=bool(r[3]),
                                       source="")
                        for r in arr
                    ]
                # Target histories: find keys like "target_0_state"
                tgt_keys = sorted([k for k in data if k.startswith("target_") and k.endswith("_state")])
                for tk in tgt_keys:
                    arr = data[tk]
                    th = [VesselState(x=float(r[0]), y=float(r[1]), psi=float(r[2]),
                                      u=float(r[3]), v=float(r[4]), r=float(r[5]),
                                      timestamp=float(r[6]))
                          for r in arr]
                    target_histories.append(th)
            else:
                # Embedded path (serial / small-batch)
                sh_list = pr.get("state_history", [])
                state_history = [VesselState(**s) for s in sh_list] if sh_list else []
                ch_list = pr.get("command_history", [])
                command_history = [ControlCommand(**c) for c in ch_list] if ch_list else []
                th_list = pr.get("target_histories", [])
                target_histories = [[VesselState(**ts) for ts in th] for th in th_list]

            r = EpisodeResult(
                episode_id=pr.get("episode_id", ""),
                scenario_id=pr.get("scenario_id", ""),
                method=pr.get("method", ""),
                seed=pr.get("seed", 0),
                state_history=state_history,
                command_history=command_history,
                target_histories=target_histories,
                failure_type=pr.get("failure_type") or None,
                failure_flags=pr.get("failure_flags", {}),
                metrics=pr.get("metrics", {}),
                runtime_stats=pr.get("runtime_stats", {}),
                metadata=pr.get("metadata", {}),
            )
            results.append(r)
        return results

    # ── Baseline registry (single source of truth) ───────────────────────
    # All baseline / ablation feature flags are defined in
    #   experiments/baseline_registry.py
    # This module imports from there — no local overrides permitted.

    def _build_controller(
        self, method: str, vessel_params: VesselParams, config: dict = None,
    ):
        """Build the appropriate controller for a given method.

        Sources feature flags from ``experiments.baseline_registry``
        (single source of truth).  B1 and B2 use standalone controllers
        (VelocityObstacleController, DynamicWindowController).
        """
        from ..experiments.baseline_registry import REGISTRY_VERSION, get_baseline

        cfg = config or {}
        baseline = get_baseline(method)
        ctrl_type = baseline["controller"]

        if ctrl_type == "velocity_obstacle":
            return VelocityObstacleController(vessel_params, cfg)
        elif ctrl_type == "dynamic_window":
            return DynamicWindowController(vessel_params, cfg)
        elif ctrl_type == "mpc":
            features = baseline["features"]

            # ── Merge order (last wins): ─────────────────────────────────
            #   1. default config     (self.config)
            #   2. registry features  (explicit per-method)
            #   3. CLI / caller cfg   (experiment script overrides)
            #   4. ablation override  (cfg["controller"] disables one key)
            # Registry features are the base for controller flags. Config
            # controller overrides apply ONLY for keys that are explicitly
            # disabled (False) — these come from ablation deep-merges.
            # Keys that are True in config are treated as defaults and do
            # NOT override the registry's per-method feature set.
            registry_features = dict(features)
            config_controller = self.config.get("controller", {})
            merged_controller = dict(registry_features)
            for k, v in config_controller.items():
                if k in merged_controller and v is False:
                    # Only apply config overrides that DISABLE features
                    # (ablation use-case).  True values in config are
                    # defaults and must not override registry's False.
                    merged_controller[k] = v
            merged_cfg = {**self.config, "controller": merged_controller}
            if cfg:
                # Allow cfg to override controller flags (ablation use-case)
                if "controller" in cfg:
                    merged_cfg["controller"].update(cfg["controller"])
                # Other top-level keys from cfg override as well
                for k, v in cfg.items():
                    if k != "controller":
                        merged_cfg[k] = v

            # ── Backend override: FORCE (not setdefault) ─────────────────
            # setdefault would be a no-op when self.config already has the
            # key, silently ignoring the registry's backend_override.
            backend = baseline.get("backend_override")
            if backend:
                merged_cfg["use_sampling"] = (backend == "sampling")
                merged_cfg.setdefault("mpc", {})
                merged_cfg["mpc"]["backend"] = backend
                merged_cfg["mpc"]["solver"] = backend

            # ── MPC overrides (G1 — 2026-06-10): per-baseline horizon, dt, etc. ─
            mpc_overrides = baseline.get("mpc_overrides", {})
            if mpc_overrides:
                merged_cfg.setdefault("mpc", {})
                for k, v in mpc_overrides.items():
                    merged_cfg["mpc"][k] = v
                    # Also apply at top level for TubeMPC.__init__ which reads mpc.horizon
                    if k == "horizon":
                        merged_cfg.setdefault("simulation", {})
                        merged_cfg["simulation"]["mpc_horizon_steps"] = v

            # ── Record baseline metadata for audit trail ─────────────────
            merged_cfg["_baseline_meta"] = {
                "method": method,
                "label": baseline["label"],
                "registry_version": REGISTRY_VERSION,
                "backend_applied": backend,
                "feature_count": sum(1 for v in features.values() if v),
            }
            return TAMRCPECCTubeMPCController(vessel_params, merged_cfg)
        else:
            raise ValueError(
                f"Unknown controller type '{ctrl_type}' for method '{method}'."
            )

    # Registry version for metadata tagging — use module-level constant
    # (removed class attribute that shadowed the imported REGISTRY_VERSION)


def _incremental_csv_save(results: list, output_path: str, mode: str = "append",
                          saved_count: int = 0) -> int:
    """Save results incrementally to CSV, preventing duplicate rows.

    Only unsaved rows (from index *saved_count*) are appended in 'append'
    mode.  'write' mode overwrites with all results.  Returns the new
    saved_count (len(results) after successful save).

    Args:
        results: Full list of accumulated result dicts or EpisodeResult objects.
        output_path: CSV file path.
        mode: 'append' for incremental (new rows only), 'write' for full rewrite.
        saved_count: Number of rows already saved to this file.

    Returns:
        New saved_count after this save.
    """
    import os as _os
    import pandas as pd

    _os.makedirs(_os.path.dirname(output_path) if _os.path.dirname(output_path) else ".", exist_ok=True)

    # Inner helper: flatten one result (dict or EpisodeResult) to a CSV row
    _SKIP_KEYS = {
        "failure_flags", "metrics", "metadata", "runtime_stats",
        # Exclude inline trajectory data — these are Python dicts/lists
        # with embedded commas that break CSV parsing.  Trajectories are
        # saved separately as .npz files; the CSV only needs scalars.
        "state_history", "command_history", "target_histories",
        "_trajectory_meta", "trajectory_file", "trajectory_format",
    }

    def _flat_row(r):
        if isinstance(r, dict):
            row = {k: v for k, v in r.items()
                   if k not in _SKIP_KEYS}
            ff = r.get("failure_flags", {})
            for i in range(1, 11):
                row[f"fail_F{i}"] = ff.get(f"F{i}", False)
            for k, v in r.get("metrics", {}).items():
                if isinstance(v, (int, float, bool, str, type(None))):
                    row[k] = v
            for k, v in r.get("runtime_stats", {}).items():
                if isinstance(v, (int, float, bool, str, type(None))):
                    row[k] = v
            return row
        else:
            # BUG #2 fix (2026-06-10): EpisodeResult objects may contain NaN/Inf
            # in metrics.  _safe_float must be applied before writing to CSV to
            # prevent "NaN" string literals that break pandas roundtrip.
            filtered_metrics = {}
            for k, v in r.metrics.items():
                filtered_metrics[k] = _safe_float(v) if isinstance(v, (int, float, bool)) or v is None else v
            filtered_runtime = {}
            for k, v in r.runtime_stats.items():
                filtered_runtime[k] = _safe_float(v) if isinstance(v, (int, float, bool)) or v is None else v
            return {
                "episode_id": r.episode_id,
                "scenario_id": r.scenario_id,
                "method": r.method,
                "seed": r.seed,
                "failure_type": r.failure_type or "",
                **{f"fail_F{i}": r.failure_flags.get(f"F{i}", False) for i in range(1, 11)},
                **filtered_metrics,
                **filtered_runtime,
            }

    # Determine which rows are new
    new_results = results[saved_count:] if saved_count > 0 else results

    if not new_results and mode != "write":
        return saved_count  # nothing new to save

    # Build rows from new (unsaved) results only
    rows = [_flat_row(r) for r in new_results]

    if mode == "write":
        # Write ALL results (final save, overwrite) — rebuild all rows
        df = pd.DataFrame([_flat_row(r) for r in results])
        df.to_csv(output_path, mode="w", header=True, index=False)
    else:
        df = pd.DataFrame(rows)
        write_header = not _os.path.exists(output_path)
        df.to_csv(output_path, mode="a", header=write_header, index=False)

    return len(results)
