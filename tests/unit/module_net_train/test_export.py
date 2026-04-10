"""Unit tests for module_net_train.export (Stage E checkpoint/metadata layer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_fields.common.errors import ContractError
from ai_fields.common.manifests import read_manifest
from ai_fields.module_net_train.export import export_training_artifacts
from ai_fields.module_net_train.model import build_model
from ai_fields.module_net_train.schemas import ModelConfig, NetTrainConfig, TrainingConfig

torch = pytest.importorskip("torch")


def _make_config() -> NetTrainConfig:
    cfg = NetTrainConfig(
        feature_mode="raw8",
        model=ModelConfig(encoder_depth=4, base_channels=8),
        training=TrainingConfig(batch_size=2, num_epochs=2, num_workers=1, device="cpu"),
    )
    cfg.validate()
    return cfg


def _normalization_contract() -> dict:
    return {
        "normalization_name": "per_band_robust_percentile",
        "stats_source": "train_norm_stats.json",
        "clip_percentiles": [0.5, 99.5],
        "scaling_range": [0.0, 1.0],
    }


def _write_dummy_source_manifest(path: Path) -> None:
    path.write_text(json.dumps({"schema_name": "prep_data.split_manifest"}), encoding="utf-8")


class TestExportTrainingArtifacts:
    def test_checkpoint_save_and_load_smoke(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)

        result = export_training_artifacts(
            run_dir=tmp_path / "run_001",
            run_id="run_001",
            config=cfg,
            model=model,
            dataset_source_run_id="prep_run_001",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            epochs_completed=2,
            best_metric_name="composite_metric",
            best_metric_value=0.73,
        )

        assert result.checkpoint_path.exists()
        checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert checkpoint["feature_mode"] == "raw8"
        assert checkpoint["assembled_model_input"] == "raw8_valid"
        assert checkpoint["in_channels"] == 9

    def test_checkpoint_metadata_contains_downstream_critical_fields(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)

        result = export_training_artifacts(
            run_dir=tmp_path / "run_meta",
            run_id="run_meta",
            config=cfg,
            model=model,
            dataset_source_run_id="prep_run_meta",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            epochs_completed=1,
            best_metric_name="composite_metric",
            best_metric_value=0.61,
        )

        metadata = read_manifest(result.checkpoint_metadata_path)
        assert metadata["schema_name"] == "net_train.checkpoint_metadata"
        assert metadata["feature_mode"] == "raw8"
        assert metadata["assembled_model_input"] == "raw8_valid"
        assert metadata["in_channels"] == 9
        assert metadata["valid_as_input_channel"] is True
        assert metadata["channel_semantics"][-1] == "valid"
        assert metadata["target_heads"]["boundary"]["classes"]["1"] == "skeleton"
        assert metadata["target_heads"]["boundary"]["classes"]["2"] == "buffer"
        assert metadata["target_heads"]["extent"]["ignore_label"] == 255
        assert metadata["normalization"]["stats_source"] == "train_norm_stats.json"
        assert metadata["model_architecture"] == "edge_aware_multitask_v1"
        assert metadata["encoder_depth"] == 4
        assert metadata["base_channels"] == 8

    def test_train_manifest_summary_and_config_used_exist(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)

        result = export_training_artifacts(
            run_dir=tmp_path / "run_artifacts",
            run_id="run_artifacts",
            config=cfg,
            model=model,
            dataset_source_run_id="prep_run_artifacts",
            dataset_source_manifest_path=src_manifest,
            normalization=_normalization_contract(),
            epochs_completed=3,
            best_metric_name="composite_metric",
            best_metric_value=0.66,
            dataset_root=tmp_path / "dataset_root",
            dataset_patch_size=384,
            dataset_feature_mode="raw8",
            summary_warnings=["amp_not_used_on_cpu"],
            scheduler_step_policy="epoch_end",
            scheduler_last_lr=1e-4,
        )

        assert result.config_used_path.exists()
        assert result.train_manifest_path.exists()
        assert result.summary_path.exists()

        train_manifest = read_manifest(result.train_manifest_path)
        assert train_manifest["schema_name"] == "net_train.train_manifest"
        assert train_manifest["dataset_source_manifest_path"] == str(src_manifest)
        assert train_manifest["dataset_root"] == str(tmp_path / "dataset_root")
        assert train_manifest["patch_size"] == 384
        assert train_manifest["dataset_feature_mode"] == "raw8"
        assert train_manifest["feature_mode"] == "raw8"
        assert train_manifest["assembled_model_input"] == "raw8_valid"
        assert train_manifest["final_input_channel_count"] == 9
        assert train_manifest["scheduler"]["step_policy"] == "epoch_end"
        assert train_manifest["scheduler"]["last_lr"] == pytest.approx(1e-4)

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["schema_name"] == "net_train.summary"
        assert summary["best_checkpoint_path"] == str(result.checkpoint_path)
        assert summary["dataset_root"] == str(tmp_path / "dataset_root")
        assert summary["patch_size"] == 384
        assert summary["dataset_feature_mode"] == "raw8"
        assert summary["warnings"] == ["amp_not_used_on_cpu"]
        assert summary["scheduler_name"] == "cosine_with_warmup"
        assert summary["scheduler_step_policy"] == "epoch_end"
        assert summary["scheduler_last_lr"] == pytest.approx(1e-4)

    def test_missing_critical_normalization_metadata_raises(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)
        bad_normalization = {
            "normalization_name": "per_band_robust_percentile",
            "clip_percentiles": [0.5, 99.5],
            "scaling_range": [0.0, 1.0],
        }

        with pytest.raises(ContractError, match="normalization.stats_source"):
            export_training_artifacts(
                run_dir=tmp_path / "run_bad_norm",
                run_id="run_bad_norm",
                config=cfg,
                model=model,
                dataset_source_run_id="prep_run_bad_norm",
                dataset_source_manifest_path=src_manifest,
                normalization=bad_normalization,
                epochs_completed=1,
                best_metric_name="composite_metric",
                best_metric_value=0.5,
            )

    def test_in_channels_mismatch_raises_explicit_error(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)

        # Simulate contract drift between model and config.
        model.in_channels = 12  # type: ignore[assignment]

        with pytest.raises(ContractError, match="does not match assembled contract"):
            export_training_artifacts(
                run_dir=tmp_path / "run_bad_channels",
                run_id="run_bad_channels",
                config=cfg,
                model=model,
                dataset_source_run_id="prep_run_bad_channels",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
                epochs_completed=1,
                best_metric_name="composite_metric",
                best_metric_value=0.5,
            )

    def test_dataset_feature_mode_mismatch_raises(self, tmp_path: Path) -> None:
        cfg = _make_config()
        model = build_model(cfg)
        src_manifest = tmp_path / "split_manifest.json"
        _write_dummy_source_manifest(src_manifest)

        with pytest.raises(ContractError, match="dataset_feature_mode is inconsistent"):
            export_training_artifacts(
                run_dir=tmp_path / "run_feature_mode_mismatch",
                run_id="run_feature_mode_mismatch",
                config=cfg,
                model=model,
                dataset_source_run_id="prep_run_feature_mode_mismatch",
                dataset_source_manifest_path=src_manifest,
                normalization=_normalization_contract(),
                epochs_completed=1,
                best_metric_name="composite_metric",
                best_metric_value=0.5,
                dataset_feature_mode="raw8_idx3",
            )
