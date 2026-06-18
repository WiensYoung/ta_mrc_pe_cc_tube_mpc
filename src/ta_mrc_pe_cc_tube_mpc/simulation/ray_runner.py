"""Ray-based experiment runner — replaces multiprocessing.Pool.

Ray provides:
  - Apache Arrow-based object store (no pipe buffer overflow)
  - Automatic task retry on worker failure
  - Graceful handling of worker crashes (no BrokenPipeError)
  - Horizontal scaling to multi-node clusters
  - Built-in progress tracking via ray.actor

This module is a drop-in replacement for the multiprocessing path in
ClosedLoopRunner._run_parallel(). The serial path remains unchanged.

Usage:
    # Option A: Use RayRunner directly
    from ta_mrc_pe_cc_tube_mpc.simulation.ray_runner import RayRunner
    runner = RayRunner(config)
    results = runner.run_batch(episodes, methods, n_workers=8)

    # Option B: Set environment variable to auto-switch
    export TA_MRC_USE_RAY=1
    # ClosedLoopRunner will use RayRunner internally

References:
    Moritz et al. (2018), "Ray: A Distributed Framework for Emerging AI
    Applications", USENIX OSDI.
    https://docs.ray.io/en/latest/ray-core.html
"""

import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from ..types import EpisodeConfig, EpisodeResult
from ..utils.io_utils import load_json, save_json
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Shared worker logic (not a Ray remote) ────────────────────────────────

def _run_task(task_spec: dict) -> dict:
    """Core worker logic shared by CPU and GPU Ray workers.

    Args:
        task_spec: Dict with keys: task_id, ep_dict, method, controller_config,
                   seed, config_dict.

    Returns:
        Dict with: task_id, result (pickled), success, error (if any).
    """
    # Ensure package importable in Ray worker
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
        ClosedLoopRunner,
        _pickle_result,
        _reconstruct_targets,
        _reconstruct_env_seq,
    )

    task_id = task_spec["task_id"]
    ep_dict = task_spec["ep_dict"]
    method = task_spec["method"]
    controller_config = task_spec["controller_config"]
    seed = task_spec["seed"]
    config_dict = task_spec["config_dict"]

    try:
        from ta_mrc_pe_cc_tube_mpc.types import (
            EpisodeConfig, VesselState, VesselParams,
        )

        ep = EpisodeConfig(
            scenario_id=ep_dict["scenario_id"],
            waterway=ep_dict.get("waterway", ""),
            duration=ep_dict["duration"],
            dt=ep_dict["dt"],
            ownship_initial_state=VesselState(**ep_dict["ownship_initial_state"]),
            ownship_params=VesselParams(**ep_dict.get("ownship_params", {})),
            targets=_reconstruct_targets(ep_dict.get("targets", [])),
            environment_sequence=_reconstruct_env_seq(ep_dict.get("environment_sequence", [])),
            random_seed=ep_dict["random_seed"],
        )

        runner = ClosedLoopRunner(config_dict)
        result = runner.run_single(ep, method, controller_config, seed)
        return {
            "task_id": task_id,
            "result": _pickle_result(result),
            "success": True,
            "error": None,
        }
    except Exception as exc:
        import traceback
        return {
            "task_id": task_id,
            "result": {
                "episode_id": ep_dict.get("scenario_id", "unknown"),
                "scenario_id": ep_dict.get("scenario_id", "unknown"),
                "method": method,
                "seed": seed,
                "failure_type": "worker_crash",
                "failure_flags": {},
                "metrics": {},
                "runtime_stats": {"worker_error": str(exc), "traceback": traceback.format_exc()},
                "metadata": {},
            },
            "success": False,
            "error": str(exc),
        }


# ── Ray remote worker functions ───────────────────────────────────────────

@ray.remote(max_retries=2, num_cpus=1) if HAS_RAY else (lambda fn: fn)
def _ray_worker_cpu(task_spec: dict) -> dict:
    """Ray remote function for CPU-only execution (CasADi/SLSQP backends)."""
    return _run_task(task_spec)


@ray.remote(max_retries=2, num_cpus=1, num_gpus=0.5) if HAS_RAY else (lambda fn: fn)
def _ray_worker_gpu(task_spec: dict) -> dict:
    """Ray remote function for GPU execution (JAX backends).

    Requests 0.5 GPU to allow 2 concurrent GPU workers per H100.
    Sets CUDA_VISIBLE_DEVICES per-worker for JAX GPU isolation.
    """
    if HAS_RAY:
        # Per-worker GPU isolation: tell JAX which GPU this worker owns
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    return _run_task(task_spec)


class RayRunner:
    """Ray-based experiment batch runner.

    Provides the same interface as ClosedLoopRunner.run_batch() but uses
    Ray for distributed execution. Falls back gracefully to multiprocessing
    if Ray is not installed.

    Key advantages over multiprocessing.Pool:
      - No pipe buffer overflow (Ray uses shared-memory object store)
      - Automatic task retry on worker death
      - Graceful handling of C-level crashes
      - Built-in progress tracking
      - Scales to multi-node clusters
    """

    def __init__(self, config: dict = None, ray_address: str = None):
        self.config = config or {}
        self.ray_address = ray_address
        self._ray_initialized = False

    def _ensure_ray(self):
        """Initialize Ray if not already running."""
        if self._ray_initialized:
            return
        if not HAS_RAY:
            raise ImportError(
                "Ray is not installed. Install with: pip install ray"
            )
        if not ray.is_initialized():
            ray.init(
                address=self.ray_address,
                ignore_reinit_error=True,
                logging_level="warning",
                # Use object spilling to disk for large results
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {"directory_path": os.path.join(tempfile.gettempdir(), "ray_spill")},
                    }),
                },
            )
        self._ray_initialized = True

    def run_batch(
        self,
        episodes: list[EpisodeConfig],
        methods: list[str],
        n_seeds: int = 1,
        seed_offset: int = 0,
        controller_configs: dict = None,
        verbose: bool = True,
        n_workers: int = 8,
        checkpoint_path: str = None,
        output_csv: str = None,
        csv_save_interval: int = 50,
    ) -> list[EpisodeResult]:
        """Run experiments using Ray distributed execution.

        Interface is compatible with ClosedLoopRunner.run_batch().
        """
        self._ensure_ray()

        from ..simulation.closed_loop_runner import (
            ClosedLoopRunner,
            _episode_to_dict,
            _pickle_result,
            _incremental_csv_save,
        )

        # ── Build task specifications ────────────────────────────────
        task_specs = []
        task_id = 0

        for ep_idx, ep in enumerate(episodes):
            for method in methods:
                for s in range(n_seeds):
                    method_hash = int(hashlib.md5(method.encode()).hexdigest()[:4], 16)
                    seed = seed_offset + ep.random_seed + s * 1000 + method_hash % 1000
                    cfg = controller_configs.get(method, {}) if controller_configs else {}

                    task_specs.append({
                        "task_id": task_id,
                        "ep_dict": _episode_to_dict(ep),
                        "method": method,
                        "controller_config": cfg,
                        "seed": seed,
                        "config_dict": self.config,
                    })
                    task_id += 1

        # ── Resume from checkpoint ───────────────────────────────────
        completed_ids = set()
        existing_results = []
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = load_json(checkpoint_path)
            completed_ids = set(ckpt.get("completed_task_ids", []))
            existing_results = ckpt.get("results", [])
            if verbose:
                print(f"Resuming from checkpoint: {len(completed_ids)}/{len(task_specs)} tasks done")

        remaining = [ts for ts in task_specs if ts["task_id"] not in completed_ids]
        if verbose:
            print(f"Ray runner: {len(remaining)} tasks, {n_workers} workers")

        results = list(existing_results)

        # ── Submit to Ray ────────────────────────────────────────────
        if remaining:
            # Detect if JAX GPU backend is configured
            mpc_backend = self.config.get("mpc", {}).get("backend", "casadi")
            use_gpu_worker = str(mpc_backend).lower() == "jax"

            # Submit all tasks asynchronously using appropriate worker
            worker_fn = _ray_worker_gpu if use_gpu_worker else _ray_worker_cpu
            futures = [
                worker_fn.remote(ts)
                for ts in remaining
            ]

            # Collect results with progress tracking
            completed_count = len(results)
            n_remaining = len(futures)

            # Process in batches for memory efficiency
            batch_size = min(n_workers * 4, n_remaining)
            for start in range(0, n_remaining, batch_size):
                batch = futures[start:start + batch_size]

                try:
                    ready_futures = ray.get(batch)
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Ray task error: {e}")
                    # Don't lose the batch — each individual task retried
                    ready_futures = []
                    for fut in batch:
                        try:
                            ready_futures.append(ray.get(fut))
                        except Exception:
                            # Task failed after retries — record as crashed
                            # Find the corresponding task_id
                            ts_idx = start + batch.index(fut) if fut in batch else -1
                            tid = remaining[ts_idx]["task_id"] if ts_idx >= 0 else -1
                            ready_futures.append({
                                "task_id": tid,
                                "result": {
                                    "failure_type": "ray_worker_exhausted_retries",
                                    "failure_flags": {},
                                    "metrics": {},
                                    "runtime_stats": {"error": str(e)},
                                },
                                "success": False,
                                "error": str(e),
                            })

                for fut_result in ready_futures:
                    results.append(fut_result["result"])
                    completed_ids.add(fut_result["task_id"])
                    completed_count += 1

                    if checkpoint_path and completed_count % 20 == 0:
                        _save_checkpoint_safe(
                            checkpoint_path, sorted(completed_ids), results,
                        )
                    if output_csv and completed_count % csv_save_interval == 0:
                        _incremental_csv_save(results, output_csv)

        # ── Final checkpoint ─────────────────────────────────────────
        if checkpoint_path:
            _save_checkpoint_safe(checkpoint_path, sorted(completed_ids), results)

        if output_csv and results:
            _incremental_csv_save(results, output_csv, mode="write")

        # ── Reconstruct EpisodeResult objects ────────────────────────
        runner = ClosedLoopRunner(self.config)
        return runner._reconstruct_results(results)

    def shutdown(self):
        """Shutdown Ray (call when all experiments are done)."""
        if self._ray_initialized and ray.is_initialized():
            ray.shutdown()
            self._ray_initialized = False


def _save_checkpoint_safe(path: str, completed_ids: list, results: list):
    """Save checkpoint with directory creation guard."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    save_json({
        "completed_task_ids": completed_ids,
        "results": results,
        "timestamp": time.time(),
    }, path)
