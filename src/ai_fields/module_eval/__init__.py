"""module_eval package."""

from ai_fields.module_eval.input_contract import (
    EvaluationInputContractResult,
    EvaluationOutputContractSkeleton,
    EvaluationRasterMetadata,
    EvaluationTrackReadiness,
    EvaluationVectorMetadata,
    resolve_evaluation_input_contract,
)
from ai_fields.module_eval.pixel_metrics import (
    GlobalPixelMetricsResult,
    PixelBinarizationPolicy,
    build_pixel_metrics_summary,
    compute_global_pixel_metrics,
)
from ai_fields.module_eval.boundary_metrics import (
    BoundaryEvaluationPolicy,
    BoundaryMetricsResult,
    build_boundary_metrics_summary,
    compute_boundary_metrics,
)
from ai_fields.module_eval.object_metrics import (
    ObjectMatchingPolicy,
    ObjectStructureMetricsResult,
    build_object_structure_metrics_summary,
    compute_object_structure_metrics,
)
from ai_fields.module_eval.export import (
    EvalExportArtifacts,
    export_eval_artifacts,
)
from ai_fields.module_eval.run_eval import (
    EvalRunInputs,
    EvalRunPolicies,
    EvalRunResult,
    run_eval,
)
from ai_fields.module_eval.comparison_contract import (
    EvalComparisonInputContractResult,
    build_comparison_readiness_summary,
    resolve_eval_comparison_contract,
)
from ai_fields.module_eval.pairwise_comparison import (
    PairwiseEvalComparisonResult,
    build_pairwise_comparison_summary,
    compute_pairwise_eval_comparison,
)
from ai_fields.module_eval.comparison_export import (
    PairwiseComparisonExportArtifacts,
    build_comparison_delta_table,
    export_pairwise_comparison_artifacts,
)
from ai_fields.module_eval.run_compare import (
    PairwiseComparisonRunInputs,
    PairwiseComparisonRunResult,
    run_pairwise_comparison,
)

__all__ = [
    "EvaluationInputContractResult",
    "EvaluationOutputContractSkeleton",
    "EvaluationRasterMetadata",
    "EvaluationTrackReadiness",
    "EvaluationVectorMetadata",
    "GlobalPixelMetricsResult",
    "PixelBinarizationPolicy",
    "BoundaryEvaluationPolicy",
    "BoundaryMetricsResult",
    "ObjectMatchingPolicy",
    "ObjectStructureMetricsResult",
    "EvalExportArtifacts",
    "EvalRunInputs",
    "EvalRunPolicies",
    "EvalRunResult",
    "EvalComparisonInputContractResult",
    "PairwiseEvalComparisonResult",
    "PairwiseComparisonExportArtifacts",
    "PairwiseComparisonRunInputs",
    "PairwiseComparisonRunResult",
    "build_boundary_metrics_summary",
    "build_comparison_readiness_summary",
    "build_comparison_delta_table",
    "build_pairwise_comparison_summary",
    "compute_boundary_metrics",
    "compute_pairwise_eval_comparison",
    "build_object_structure_metrics_summary",
    "compute_object_structure_metrics",
    "export_eval_artifacts",
    "export_pairwise_comparison_artifacts",
    "run_pairwise_comparison",
    "resolve_eval_comparison_contract",
    "run_eval",
    "build_pixel_metrics_summary",
    "compute_global_pixel_metrics",
    "resolve_evaluation_input_contract",
]
