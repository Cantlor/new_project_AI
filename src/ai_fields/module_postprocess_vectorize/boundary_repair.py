"""Stage B.5: Hybrid boundary repair for module_postprocess_vectorize.

Applies morphological closing to the binary boundary map before marker generation
to fill small gaps in boundary continuity.  This is the baseline "hybrid" repair:
morphology-first with a conservative structuring element.

Scope:
- Input: binary boundary support mask (2-D bool/uint8 array).
- Output: repaired binary mask of the same shape.
- No polygon topology operations at this stage (those live in Stage D).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_fields.common.errors import ContractError


@dataclass(frozen=True)
class BoundaryRepairPolicy:
    """Explicit policy for Stage B.5 boundary repair."""

    enabled: bool = True
    closing_radius: int = 2
    """Radius (in pixels) of the disk-shaped structuring element for morphological closing.
    closing_radius=0 disables the closing step even when enabled=True."""

    def validate(self) -> "BoundaryRepairPolicy":
        if not isinstance(self.enabled, bool):
            raise ContractError("BoundaryRepairPolicy.enabled must be a bool.")
        if not isinstance(self.closing_radius, int) or self.closing_radius < 0:
            raise ContractError(
                "BoundaryRepairPolicy.closing_radius must be a non-negative integer."
            )
        return self


@dataclass(frozen=True)
class BoundaryRepairResult:
    """Result of Stage B.5 boundary repair."""

    repaired_boundary_mask: np.ndarray
    shape: tuple[int, int]
    repair_applied: bool
    closing_radius: int
    policy: dict[str, Any]


def _require_scipy() -> Any:
    try:
        import scipy.ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ContractError(
            "scipy is required for boundary repair (Stage B.5)."
        ) from exc
    return ndi


def _disk_structuring_element(radius: int) -> np.ndarray:
    """Return a binary disk-shaped structuring element of given radius."""
    if radius == 0:
        return np.array([[1]], dtype=bool)
    size = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(bool)


def apply_boundary_repair(
    boundary_mask: np.ndarray,
    *,
    policy: BoundaryRepairPolicy,
) -> BoundaryRepairResult:
    """Apply hybrid boundary repair to a binary boundary mask.

    Args:
        boundary_mask: 2-D array with dtype bool or uint8 (1 = boundary, 0 = background).
        policy: BoundaryRepairPolicy controlling closing radius and enabled flag.

    Returns:
        BoundaryRepairResult with repaired_boundary_mask of the same shape.
    """
    if not isinstance(policy, BoundaryRepairPolicy):
        raise ContractError("policy must be BoundaryRepairPolicy.")
    policy.validate()

    if not isinstance(boundary_mask, np.ndarray) or boundary_mask.ndim != 2:
        raise ContractError(
            "boundary_mask must be a 2-D numpy array."
        )

    h, w = boundary_mask.shape
    binary = (boundary_mask > 0).astype(np.uint8)

    if not policy.enabled or policy.closing_radius == 0:
        return BoundaryRepairResult(
            repaired_boundary_mask=binary,
            shape=(h, w),
            repair_applied=False,
            closing_radius=policy.closing_radius,
            policy={
                "enabled": policy.enabled,
                "closing_radius": policy.closing_radius,
                "operation": "none",
            },
        )

    ndi = _require_scipy()
    struct = _disk_structuring_element(policy.closing_radius)
    closed = ndi.binary_closing(binary.astype(bool), structure=struct).astype(np.uint8)

    return BoundaryRepairResult(
        repaired_boundary_mask=closed,
        shape=(h, w),
        repair_applied=True,
        closing_radius=policy.closing_radius,
        policy={
            "enabled": policy.enabled,
            "closing_radius": policy.closing_radius,
            "operation": "morphological_binary_closing",
            "structuring_element": "disk",
        },
    )


__all__ = [
    "BoundaryRepairPolicy",
    "BoundaryRepairResult",
    "apply_boundary_repair",
]
