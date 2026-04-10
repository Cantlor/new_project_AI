"""Unit tests for module_net_train.run_train minimal orchestration layer."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pytest

from ai_fields.common.errors import ContractError, ValidPolicyError
from ai_fields.common.manifests import read_manifest
from ai_fields.common.constants import CHANNEL_COUNTS
from ai_fields.module_net_train import run_train as run_train_module
from ai_fields.module_net_train.run_train import run_train_baseline
from ai_fields.module_net_train.schemas import (
    MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1,
    MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
    ModelConfig,
    MonitoringConfig,
    NetTrainConfig,
    SchedulerConfig,
    TrainingConfig,
)

torch = pytest.importorskip("torch")


def _normalization_contract() -> dict:
    stats_source = (
        Path(__file__).resolve().parents[2] / "fixtures" / "net_train_norm_stats.raw8.json"
    )
    return {
        "normalization_name": "per_band_robust_percentile",
        "stats_source": str(stats_source),
        "clip_percentiles": [0.5, 99.5],
        "scaling_range": [0.0, 1.0],
    }


def _make_net_train_cfg() -> NetTrainConfig:
    cfg = NetTrainConfig(
        feature_mode="raw8",
        model=ModelConfig(encoder_depth=3, base_channels=8),
        training=TrainingConfig(
            batch_size=2,
            num_epochs=1,
            gradient_clip=1.0,
            amp=False,
            seed=7,
            num_workers=1,
            gradient_accumulation_steps=1,
            device="cpu",
        ),
    )
    cfg.validate()
    return cfg


def _make_split_dir(
    root: Path,
    split_name: str,
    *,
    feature_mode: str,
    n_samples: int,
    hw: int = 32,
) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    split_dir = root / split_name
    for layer in ("img", "extent", "boundary", "distance", "valid", "meta"):
        (split_dir / layer).mkdir(parents=True, exist_ok=True)

    count = CHANNEL_COUNTS[feature_mode]
    transform = from_bounds(0, 0, float(hw), float(hw), width=hw, height=hw)
    crs = CRS.from_epsg(32637)

    for i in range(n_samples):
        sid = f"{split_name}_patch_{i:06d}"

        img = np.full((count, hw, hw), fill_value=1000.0, dtype=np.float32)
        with rasterio.open(
            split_dir / "img" / f"{sid}_img.tif",
            "w",
            driver="GTiff",
            height=hw,
            width=hw,
            count=count,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(img)

        extent = np.zeros((1, hw, hw), dtype=np.uint8)
        extent[:, 4:-4, 4:-4] = 1
        with rasterio.open(
            split_dir / "extent" / f"{sid}_extent.tif",
            "w",
            driver="GTiff",
            height=hw,
            width=hw,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(extent)

        boundary = np.zeros((1, hw, hw), dtype=np.uint8)
        boundary[:, 4, 4:-4] = 1
        boundary[:, 5, 4:-4] = 2
        with rasterio.open(
            split_dir / "boundary" / f"{sid}_boundary.tif",
            "w",
            driver="GTiff",
            height=hw,
            width=hw,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(boundary)

        distance = np.random.default_rng(42 + i).random((1, hw, hw), dtype=np.float32)
        with rasterio.open(
            split_dir / "distance" / f"{sid}_distance.tif",
            "w",
            driver="GTiff",
            height=hw,
            width=hw,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(distance)

        valid = np.ones((1, hw, hw), dtype=np.uint8)
        with rasterio.open(
            split_dir / "valid" / f"{sid}_valid.tif",
            "w",
            driver="GTiff",
            height=hw,
            width=hw,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(valid)

        meta = {
            "patch_id": sid,
            "feature_mode": feature_mode,
            "feature_channel_count": count,
            "channel_names": [f"band_{j + 1}" for j in range(count)],
        }
        (split_dir / "meta" / f"{sid}_meta.json").write_text(
            json.dumps(meta),
            encoding="utf-8",
        )

    return split_dir


class TestRunTrainBaseline:
    def test_happy_path_creates_artifacts_and_result(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_001",
            run_id="net_train_run_001",
            dataset_source_run_id="prep_run_001",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
        )

        assert result.run_dir.exists()
        assert result.checkpoint_path.exists()
        assert result.checkpoint_metadata_path.exists()
        assert result.train_manifest_path.exists()
        assert result.summary_path.exists()
        assert result.config_used_path.exists()
        assert result.dataset_root == tmp_path / "dataset"
        assert result.dataset_patch_size == 32
        assert "total" in result.train_summary
        assert "total" in result.val_summary
        assert result.history_path.exists()
        assert result.last_checkpoint_path.exists()
        assert "extent_f1" in result.train_summary
        assert "boundary_f1" in result.train_summary
        assert "extent_f1" in result.val_summary
        assert "boundary_f1" in result.val_summary
        expected_composite = 0.6 * float(result.val_summary["boundary_f1"]) + 0.4 * float(
            result.val_summary["extent_f1"]
        )
        assert result.best_metric_name == "composite_boundary_extent_f1"
        assert result.monitored_metric_mode == "max"
        assert result.monitored_metric_policy_note is None
        assert result.best_metric_value == pytest.approx(expected_composite)
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["dataset_root"] == str(tmp_path / "dataset")
        assert summary["patch_size"] == 32
        assert summary["dataset_feature_mode"] == "raw8"
        assert summary["scheduler_name"] == "cosine_with_warmup"
        assert summary["scheduler_step_policy"] == "epoch_end"
        assert isinstance(summary["scheduler_last_lr"], (int, float))

    def test_export_metadata_consistent_with_contract(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=1)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_002",
            run_id="net_train_run_002",
            dataset_source_run_id="prep_run_002",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
        )

        metadata = read_manifest(result.checkpoint_metadata_path)
        assert metadata["feature_mode"] == "raw8"
        assert metadata["assembled_model_input"] == "raw8_valid"
        assert metadata["in_channels"] == 9
        assert metadata["valid_as_input_channel"] is True
        assert metadata["channel_semantics"][-1] == "valid"
        assert metadata["target_heads"]["boundary"]["classes"]["2"] == "buffer"
        assert metadata["model_architecture"] == "edge_aware_multitask_v1"
        assert metadata["encoder_depth"] == 3
        assert metadata["base_channels"] == 8

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert "train" in summary
        assert "val" in summary
        assert "total" in summary["train"]
        assert "total" in summary["val"]

    def test_missing_source_manifest_raises_contract_error(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=1)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=1)
        missing_src_manifest = tmp_path / "dataset" / "missing_split_manifest.json"

        with pytest.raises(ContractError, match="dataset_source_manifest_path does not exist"):
            run_train_baseline(
                config=cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_003",
                run_id="net_train_run_003",
                dataset_source_run_id="prep_run_003",
                dataset_source_manifest_path=missing_src_manifest,
                normalization=_normalization_contract(),
            )

    def test_history_csv_and_policy_metadata_are_explicit(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_history",
            run_id="net_train_run_history",
            dataset_source_run_id="prep_run_history",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            epochs_override=2,
        )

        assert result.history_path.exists()
        with result.history_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames is not None
            assert reader.fieldnames[:11] == [
                "epoch",
                "train_extent",
                "train_boundary",
                "train_distance",
                "train_total",
                "val_extent",
                "val_boundary",
                "val_distance",
                "val_total",
                "monitored_metric",
                "is_best",
            ]
            rows = list(reader)

        assert len(rows) == 2
        assert any(r["is_best"] == "True" for r in rows)
        for row in rows:
            monitored = float(row["monitored_metric"])
            assert 0.0 <= monitored <= 1.0

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["best_metric_name"] == "composite_boundary_extent_f1"
        assert summary["monitored_metric_mode"] == "max"
        assert summary["best_metric_value"] == pytest.approx(result.best_metric_value)
        assert summary["best_epoch"] == result.best_epoch
        assert summary["best_checkpoint_path"] == str(result.checkpoint_path)
        assert summary["last_checkpoint_path"] == str(result.last_checkpoint_path)
        assert summary["history_path"] == str(result.history_path)
        assert summary["monitored_metric_policy_note"] is None
        assert summary["scheduler_name"] == "cosine_with_warmup"
        assert summary["scheduler_step_policy"] == "epoch_end"
        assert isinstance(summary["scheduler_last_lr"], (int, float))

        train_manifest = read_manifest(result.train_manifest_path)
        assert train_manifest["dataset_root"] == str(tmp_path / "dataset")
        assert train_manifest["patch_size"] == 32
        assert train_manifest["dataset_feature_mode"] == "raw8"
        assert train_manifest["best_checkpoint_path"] == str(result.checkpoint_path)
        assert train_manifest["last_checkpoint_path"] == str(result.last_checkpoint_path)
        assert train_manifest["history_path"] == str(result.history_path)
        assert train_manifest["monitored_metric_name"] == "composite_boundary_extent_f1"
        assert train_manifest["monitored_metric_mode"] == "max"
        assert train_manifest["scheduler"]["name"] == "cosine_with_warmup"
        assert train_manifest["scheduler"]["step_policy"] == "epoch_end"
        assert isinstance(train_manifest["scheduler"]["last_lr"], (int, float))

    def test_best_last_checkpoint_selection_flow(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_ckpt_policy",
            run_id="net_train_run_ckpt_policy",
            dataset_source_run_id="prep_run_ckpt_policy",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            epochs_override=3,
        )

        assert result.checkpoint_path.name == "best.ckpt"
        assert result.last_checkpoint_path.name == "last.ckpt"
        assert result.checkpoint_path.exists()
        assert result.last_checkpoint_path.exists()
        assert result.best_epoch in {1, 2, 3}
        assert result.epochs_completed == 3

        with result.history_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        monitored = [float(r["monitored_metric"]) for r in rows]
        max_metric = max(monitored)
        max_epoch = monitored.index(max_metric) + 1  # csv rows are 1-based epochs
        assert result.best_metric_value == pytest.approx(max_metric)
        assert result.best_epoch == max_epoch

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["best_checkpoint_path"] == str(result.checkpoint_path)
        assert summary["last_checkpoint_path"] == str(result.last_checkpoint_path)
        assert summary["best_epoch"] == result.best_epoch

    def test_composite_metric_requires_explicit_f1_signals(self) -> None:
        with pytest.raises(ContractError, match="boundary_f1"):
            run_train_module._compute_monitored_metric(
                {"extent_f1": 0.5},
                "composite_boundary_extent_f1",
            )

    def test_interim_policy_is_available_only_by_explicit_override(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_interim_override",
            run_id="net_train_run_interim_override",
            dataset_source_run_id="prep_run_interim_override",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            monitored_metric_name="interim_val_total_loss",
        )

        assert result.best_metric_name == "interim_val_total_loss"
        assert result.monitored_metric_mode == "min"
        assert isinstance(result.monitored_metric_policy_note, str)
        assert result.monitored_metric_policy_note.strip() != ""

    def test_run_layer_uses_config_monitoring_policy_as_source_of_truth(self, tmp_path: Path) -> None:
        cfg = NetTrainConfig(
            feature_mode="raw8",
            monitoring=MonitoringConfig(
                monitored_metric_name=MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
                monitored_metric_mode="min",
            ),
            model=ModelConfig(encoder_depth=3, base_channels=8),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=1,
                gradient_clip=1.0,
                amp=False,
                seed=13,
                num_workers=1,
                gradient_accumulation_steps=1,
                device="cpu",
            ),
        )
        cfg.validate()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_cfg_policy",
            run_id="net_train_run_cfg_policy",
            dataset_source_run_id="prep_run_cfg_policy",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
        )

        assert result.best_metric_name == MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS
        assert result.monitored_metric_mode == "min"

    def test_run_layer_override_still_overrides_config_policy(self, tmp_path: Path) -> None:
        cfg = NetTrainConfig(
            feature_mode="raw8",
            monitoring=MonitoringConfig(
                monitored_metric_name=MONITORED_METRIC_INTERIM_VAL_TOTAL_LOSS,
                monitored_metric_mode="min",
            ),
            model=ModelConfig(encoder_depth=3, base_channels=8),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=1,
                gradient_clip=1.0,
                amp=False,
                seed=17,
                num_workers=1,
                gradient_accumulation_steps=1,
                device="cpu",
            ),
        )
        cfg.validate()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_cfg_override",
            run_id="net_train_run_cfg_override",
            dataset_source_run_id="prep_run_cfg_override",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            monitored_metric_name=MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1,
        )

        assert result.best_metric_name == MONITORED_METRIC_COMPOSITE_BOUNDARY_EXTENT_F1
        assert result.monitored_metric_mode == "max"

    def test_run_layer_unsupported_scheduler_raises_explicit_error(self, tmp_path: Path) -> None:
        cfg = NetTrainConfig(
            feature_mode="raw8",
            scheduler=SchedulerConfig(name="one_cycle", warmup_epochs=0, min_lr=1e-6),
            model=ModelConfig(encoder_depth=3, base_channels=8),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=1,
                gradient_clip=1.0,
                amp=False,
                seed=19,
                num_workers=1,
                gradient_accumulation_steps=1,
                device="cpu",
            ),
        )
        cfg.validate()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        with pytest.raises(ContractError, match="one_cycle"):
            run_train_baseline(
                config=cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_bad_scheduler",
                run_id="net_train_run_bad_scheduler",
                dataset_source_run_id="prep_run_bad_scheduler",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
            )

    def test_run_layer_plateau_scheduler_policy_is_reflected(self, tmp_path: Path) -> None:
        cfg = NetTrainConfig(
            feature_mode="raw8",
            scheduler=SchedulerConfig(name="plateau", warmup_epochs=0, min_lr=1e-6),
            model=ModelConfig(encoder_depth=3, base_channels=8),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=2,
                gradient_clip=1.0,
                amp=False,
                seed=23,
                num_workers=1,
                gradient_accumulation_steps=1,
                device="cpu",
            ),
        )
        cfg.validate()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=2)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=2)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        result = run_train_baseline(
            config=cfg,
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            run_dir=tmp_path / "runs" / "net_train_run_plateau",
            run_id="net_train_run_plateau",
            dataset_source_run_id="prep_run_plateau",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
        )

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["scheduler_name"] == "plateau"
        assert summary["scheduler_step_policy"] == "epoch_end_val_total"
        assert isinstance(summary["scheduler_last_lr"], (int, float))

    def test_invalid_config_raises_explicit_contract_error(self, tmp_path: Path) -> None:
        bad_cfg = NetTrainConfig(
            feature_mode="raw8",
            valid_as_input_channel=False,  # violates contract by design
            model=ModelConfig(encoder_depth=3, base_channels=8),
            training=TrainingConfig(
                batch_size=2,
                num_epochs=1,
                gradient_clip=1.0,
                amp=False,
                seed=11,
                num_workers=1,
                gradient_accumulation_steps=1,
                device="cpu",
            ),
        )
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=1)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=1)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        with pytest.raises(ValidPolicyError):
            run_train_baseline(
                config=bad_cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_004",
                run_id="net_train_run_004",
                dataset_source_run_id="prep_run_004",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
            )

    def test_mixed_patch_sizes_are_rejected_before_training(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(
            tmp_path / "dataset",
            "train",
            feature_mode="raw8",
            n_samples=2,
            hw=32,
        )
        val_dir = _make_split_dir(
            tmp_path / "dataset",
            "val",
            feature_mode="raw8",
            n_samples=2,
            hw=24,
        )
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        with pytest.raises(ContractError, match="patch size"):
            run_train_baseline(
                config=cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_mixed_patch",
                run_id="net_train_run_mixed_patch",
                dataset_source_run_id="prep_run_mixed_patch",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
            )

    def test_dataset_feature_mode_mismatch_is_rejected(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=1)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=1)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        with pytest.raises(ContractError, match="dataset_feature_mode is inconsistent"):
            run_train_baseline(
                config=cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_feature_mode_mismatch",
                run_id="net_train_run_feature_mode_mismatch",
                dataset_source_run_id="prep_run_feature_mode_mismatch",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
                dataset_feature_mode="raw8_idx3",
            )

    def test_missing_normalization_stats_source_fails_explicitly(self, tmp_path: Path) -> None:
        cfg = _make_net_train_cfg()
        train_dir = _make_split_dir(tmp_path / "dataset", "train", feature_mode="raw8", n_samples=1)
        val_dir = _make_split_dir(tmp_path / "dataset", "val", feature_mode="raw8", n_samples=1)
        src_manifest = tmp_path / "dataset" / "split_manifest.json"
        src_manifest.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")

        bad_norm = {
            "normalization_name": "per_band_robust_percentile",
            "stats_source": str(tmp_path / "dataset" / "missing_norm_stats.json"),
            "clip_percentiles": [0.5, 99.5],
            "scaling_range": [0.0, 1.0],
        }
        with pytest.raises(ContractError, match="normalization.stats_source does not exist"):
            run_train_baseline(
                config=cfg,
                train_split_dir=train_dir,
                val_split_dir=val_dir,
                run_dir=tmp_path / "runs" / "net_train_run_missing_norm_stats",
                run_id="net_train_run_missing_norm_stats",
                dataset_source_run_id="prep_run_missing_norm_stats",
                dataset_source_manifest_path=src_manifest,
                normalization=bad_norm,
            )
