#!/usr/bin/env python3
"""Shared semantic exclusions for NumSharp's operation-matrix rollups.

The timing floor catches most view/metadata calls, but it is not a proof: a noisy O(1) call can
cross 1 us and accidentally enter a geomean. These sets describe benchmark SCENARIOS whose work is
independent of the input element count N. O(ndim) counts as O(1) here because benchmark rank is
fixed. Raw timings remain published; only performance rollups/rankings exclude these cells.
"""

from __future__ import annotations

import re


# Browser-facing API/method names and display-only scenarios. The DocFX dashboard mirrors these
# two sets so it can apply the policy to already-published snapshots that predate semantic status
# tagging; test_o1_exclusions.py locks the Python and JavaScript copies together.
O1_FUNCTION_NAMES = frozenset({
    "np.real", "np.asarray", "np.asanyarray", "np.asmatrix", "np.frombuffer",
    "np.atleast_1d", "np.atleast_2d", "np.atleast_3d", "np.array_split", "np.split",
    "np.dsplit", "np.hsplit", "np.vsplit", "np.unstack", "np.broadcast_arrays",
    "np.broadcast_to", "np.diagonal", "np.expand_dims", "np.flip", "np.fliplr", "np.flipud",
    "np.matrix_transpose", "np.moveaxis", "np.permute_dims", "np.ravel", "np.reshape",
    "np.rollaxis", "np.rot90", "np.squeeze", "np.swapaxes", "np.transpose", "np.ix_", "np.size",
    "np.iscomplexobj", "np.isrealobj", "np.isscalar", "np.isfortran", "np.can_cast",
    "np.common_type", "np.isdtype", "np.issubdtype", "np.min_scalar_type", "np.mintypecode",
    "np.promote_types", "np.result_type", "np.iterable", "np.nested_iters",
    "np.format_float_positional", "np.format_float_scientific", "np.get_printoptions",
    "np.printoptions", "np.set_printoptions", "np.einsum_path", "np.linalg.diagonal",
    "np.linalg.matrix_transpose", "ndarray.T", "ndarray.conj", "ndarray.conjugate",
    "ndarray.diagonal", "ndarray.getfield", "ndarray.item", "ndarray.ravel", "ndarray.reshape",
    "ndarray.setflags", "ndarray.squeeze", "ndarray.swapaxes", "ndarray.to_device",
    "ndarray.transpose", "ndarray.view",
})

O1_EXACT_DISPLAY_OPERATIONS = frozenset({
    "np.diag", "np.diag(a2d)", "reshape 1d->2d", "reshape 1d->3d", "reshape 2d->1d",
    "a[100:1000] (contiguous slice)", "a[10:100, :] (row slice 2d)",
    "a[:, 10:100] (col slice 2d)", "a[::-1] (reversed)", "a[::-1]",
    "a[::2] (strided slice)",
})


# One/few wrapper views, shape/stride metadata, scalar metadata, or a scalar element access.
O1_VIEW_OR_METADATA_KEYS = frozenset({
    "np.real",
    "np.asarray",
    "np.asanyarray",
    "np.asmatrix",
    "np.frombuffer",
    "np.atleast_1d",
    "np.atleast_2d",
    "np.atleast_3d",
    "np.broadcast_arrays",
    "np.diag view",
    "np.diagonal",
    "np.flip",
    "np.fliplr",
    "np.flipud",
    "np.matrix_transpose",
    "np.permute_dims",
    "np.ravel",
    "np.reshape",
    "np.rot90",
    "np.squeeze",
    "np.transpose",
    "np.transpose axes",
    "np.ix_",
    "np.size",
    "np.iscomplexobj",
    "np.isrealobj",
    "np.isscalar",
    "np.isfortran",
    "np.can_cast",
    "np.common_type",
    "np.isdtype",
    "np.issubdtype",
    "np.min_scalar_type",
    "np.mintypecode",
    "np.promote_types",
    "np.result_type",
    "np.iterable",
    "np.nested_iters",
    "np.format_float_positional",
    "np.format_float_scientific",
    "np.get_printoptions",
    "np.printoptions",
    "np.set_printoptions",
    "np.linalg.diagonal",
    "np.linalg.matrix_transpose",
    "a.t (transpose 2d)",
    "a.conj",
    "a.conjugate",
    "a.diagonal",
    "a.getfield",
    "a.item",
    "a.ravel",
    "a.reshape",
    "a.setflags(write=true)",
    "a.squeeze",
    "a.swapaxes(0, 1)",
    "a.to_device",
    "a.transpose",
    "a.view",
    "a[100:1000] (contiguous slice)",
    "a[10:100, :] (row slice 2d)",
    "a[:, 10:100] (col slice 2d)",
    "a[::-1]",
    "a[::2] (strided slice)",
    "reshape 1d->2d",
    "reshape 1d->3d",
    "reshape 2d->1d",
})


# Each benchmark fixes the number of sections/views (7 or 10), so work is O(k) with constant k,
# independent of the number of elements in the source array.
O1_FIXED_VIEW_COLLECTION_KEYS = frozenset({
    "np.array_split",
    "np.split",
    "np.dsplit",
    "np.hsplit",
    "np.vsplit",
    "np.unstack",
})


# With the benchmark's fixed two-operand expression, einsum_path reads shapes/dtypes and plans a
# contraction without traversing array elements.
O1_SHAPE_PLANNER_KEYS = frozenset({"np.einsum_path"})


# Axis/target arguments intentionally remain in normalized keys, so match the API stem. Each is a
# shape/stride-only operation for the benchmark inputs.
O1_VIEW_PREFIXES = (
    "np.broadcast_to",
    "np.expand_dims",
    "np.moveaxis",
    "np.rollaxis",
    "np.swapaxes",
)


O1_OPERATION_KEYS = (
    O1_VIEW_OR_METADATA_KEYS | O1_FIXED_VIEW_COLLECTION_KEYS | O1_SHAPE_PLANNER_KEYS)


def o1_exclusion_reason(normalized_operation: str) -> str | None:
    """Return why this normalized benchmark scenario is excluded from performance rollups."""
    key = normalized_operation.strip().lower()
    if key in O1_FIXED_VIEW_COLLECTION_KEYS:
        return "O(1) in N: fixed-count view collection"
    if key in O1_SHAPE_PLANNER_KEYS:
        return "O(1) in N: shape-only planning with fixed operand count"
    if key in O1_VIEW_OR_METADATA_KEYS or key.startswith(O1_VIEW_PREFIXES):
        return "O(1) in N: view, shape/stride, dtype, or scalar metadata only"
    return None


def o1_display_exclusion_reason(operation: str) -> str | None:
    """Classify a report/display operation without requiring the merge normalizer."""
    value = re.sub(
        r"\s*\((?:bool|u?int\d+|float\d+|complex\d+|decimal)\)\s*$", "", str(operation),
        flags=re.I).strip()
    match = re.search(r"\bnp(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value)
    function_name = match.group(0) if match else None
    if function_name is None:
        match = re.match(r"^\s*a\.([A-Za-z_][A-Za-z0-9_]*)", value)
        function_name = f"ndarray.{match.group(1)}" if match else None
    if function_name in O1_FUNCTION_NAMES or value.lower() in O1_EXACT_DISPLAY_OPERATIONS:
        return "O(1) in N: semantic report exclusion"
    return None
