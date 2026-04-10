"""Smoke tests for canonical shell runner tools/run_module_net_train.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "run_module_net_train.sh"
BASELINE_CFG = REPO_ROOT / "configs" / "module_net_train" / "baseline.raw8.yaml"


def _run_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestRunModuleNetTrainScript:
    def test_prep_run_dir_mode_resolves_dataset_paths_in_dry_run(self) -> None:
        prep_run = "runs/module_prep_data/prep-raw8-384-ps384"
        result = _run_runner(
            "--config",
            str(BASELINE_CFG),
            "--prep-run-dir",
            prep_run,
            "--patch-size",
            "384",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Dataset selection mode: prep-run-dir" in result.stdout
        assert f"[INFO] Prep run dir: {prep_run}" in result.stdout
        assert f"[INFO] Dataset root: {prep_run}/06_split_dataset/dataset" in result.stdout
        assert f"[INFO] Dataset source manifest: {prep_run}/06_split_dataset/split_manifest.json" in result.stdout
        assert "[INFO] Patch size consistency check: 384" in result.stdout

    def test_dataset_root_mode_requires_explicit_manifest_in_dry_run(self) -> None:
        result = _run_runner(
            "--config",
            str(BASELINE_CFG),
            "--dataset-root",
            "prep_data_for_train/raw8/512",
            "--dataset-source-manifest",
            "runs/module_prep_data/prep-raw8-512-ps512/06_split_dataset/split_manifest.json",
            "--dry-run",
        )

        assert result.returncode == 0, result.stderr
        assert "[INFO] Dataset selection mode: dataset-root" in result.stdout
        assert "[INFO] Dataset root: prep_data_for_train/raw8/512" in result.stdout

    def test_missing_dataset_selection_fails_clearly(self) -> None:
        result = _run_runner(
            "--config",
            str(BASELINE_CFG),
            "--dry-run",
        )

        assert result.returncode != 0
        assert "Missing dataset selection" in result.stderr

    def test_ambiguous_selection_fails_clearly(self) -> None:
        result = _run_runner(
            "--config",
            str(BASELINE_CFG),
            "--prep-run-dir",
            "runs/module_prep_data/prep-raw8-256-ps256",
            "--dataset-root",
            "prep_data_for_train/raw8/256",
            "--dry-run",
        )

        assert result.returncode != 0
        assert "Ambiguous dataset selection" in result.stderr
