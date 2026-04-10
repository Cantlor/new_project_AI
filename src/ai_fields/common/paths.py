# Run directory helpers.
# Pure path functions — no filesystem side-effects, no I/O.
# (REPO_CONVENTIONS.md §14.1, REPO_SKELETON.md §runs)

from pathlib import Path


def get_run_dir(module_name: str, run_id: str) -> Path:
    """Return the output directory for a given module run.

    Convention: runs/<module_name>/<run_id>/
    (REPO_CONVENTIONS.md §14.1)
    """
    return Path("runs") / module_name / run_id


def get_artifacts_dir(run_dir: Path) -> Path:
    """Return the artifacts subdirectory for a run."""
    return run_dir / "artifacts"


def get_logs_dir(run_dir: Path) -> Path:
    """Return the logs subdirectory for a run."""
    return run_dir / "logs"
