#!/bin/bash
# =============================================================================
# ILKernelGenerator Split Script - Phase 2
# =============================================================================
# Splits:
#   - ILKernelGenerator.Reduction.Axis.cs (2221 lines) -> 4 files
#   - ILKernelGenerator.Masking.cs (1343 lines) -> 4 files
#
# Usage:
#   ./split-ilkernel-phase2.sh --dry-run    # Preview what will be created
#   ./split-ilkernel-phase2.sh              # Execute the split
# =============================================================================

set -e

# Configuration
KERNELS_DIR="src/NumSharp.Core/Backends/Kernels"
AXIS_SRC="$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.cs"
MASKING_SRC="$KERNELS_DIR/ILKernelGenerator.Masking.cs"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be created ==="
    echo ""
fi

# Verify source files exist
if [[ ! -f "$AXIS_SRC" ]]; then
    echo "ERROR: $AXIS_SRC not found"
    exit 1
fi
if [[ ! -f "$MASKING_SRC" ]]; then
    echo "ERROR: $MASKING_SRC not found"
    exit 1
fi

# Verify line counts
AXIS_LINES=$(wc -l < "$AXIS_SRC")
MASKING_LINES=$(wc -l < "$MASKING_SRC")

echo "Source file verification:"
echo "  $AXIS_SRC: $AXIS_LINES lines (expected: 2221)"
echo "  $MASKING_SRC: $MASKING_LINES lines (expected: 1343)"
echo ""

if [[ "$AXIS_LINES" -ne 2221 ]]; then
    echo "WARNING: Reduction.Axis.cs line count mismatch! Expected 2221, got $AXIS_LINES"
    echo "Line ranges may be incorrect. Aborting."
    exit 1
fi
if [[ "$MASKING_LINES" -ne 1343 ]]; then
    echo "WARNING: Masking.cs line count mismatch! Expected 1343, got $MASKING_LINES"
    echo "Line ranges may be incorrect. Aborting."
    exit 1
fi

echo "Line counts verified. Proceeding..."
echo ""

# =============================================================================
# BOUNDARY VERIFICATION
# =============================================================================

echo "=== Boundary Verification ==="

verify_line() {
    local file="$1"
    local line="$2"
    local expected="$3"
    local actual=$(sed -n "${line}p" "$file")
    if [[ "$actual" == *"$expected"* ]]; then
        echo "  ✓ Line $line: found '$expected'"
    else
        echo "  ✗ Line $line: expected '$expected', got '$actual'"
        echo "ERROR: Boundary mismatch. Aborting."
        exit 1
    fi
}

echo "Reduction.Axis.cs boundaries:"
verify_line "$AXIS_SRC" 25 "#region Axis Reduction"
verify_line "$AXIS_SRC" 116 "CreateAxisArgReductionKernel"
verify_line "$AXIS_SRC" 428 "}"
verify_line "$AXIS_SRC" 434 "CreateAxisReductionKernelGeneral"
verify_line "$AXIS_SRC" 804 "}"
verify_line "$AXIS_SRC" 809 "CreateAxisReductionKernelTyped"
verify_line "$AXIS_SRC" 1478 "#endregion"
verify_line "$AXIS_SRC" 1480 "#region IKernelProvider"
verify_line "$AXIS_SRC" 1584 "#endregion"
verify_line "$AXIS_SRC" 1586 "#region Var/Std"
verify_line "$AXIS_SRC" 2219 "#endregion"

echo ""
echo "Masking.cs boundaries:"
verify_line "$MASKING_SRC" 27 "#region NonZero"
verify_line "$MASKING_SRC" 248 "#endregion"
verify_line "$MASKING_SRC" 250 "#region Boolean Masking"
verify_line "$MASKING_SRC" 255 "CountTrueSimdHelper"
verify_line "$MASKING_SRC" 328 "CopyMaskedElementsHelper"
verify_line "$MASKING_SRC" 404 "}"
verify_line "$MASKING_SRC" 415 "VarSimdHelper"
verify_line "$MASKING_SRC" 676 "}"
verify_line "$MASKING_SRC" 685 "NanSumSimdHelperFloat"
verify_line "$MASKING_SRC" 1341 "#endregion"

echo ""
echo "All boundaries verified!"
echo ""

# =============================================================================
# COMMON HEADER COMPONENTS
# =============================================================================

USINGS='using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;'

NS_OPEN='namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {'

NS_CLOSE='    }
}'

# =============================================================================
# HELPER FUNCTION
# =============================================================================

create_file() {
    local dest="$1"
    local header="$2"
    local content="$3"

    if $DRY_RUN; then
        echo "Would create: $dest"
        local header_lines=$(echo "$header" | wc -l)
        local content_lines=$(echo "$content" | wc -l)
        local total=$((header_lines + content_lines))
        echo "  Header: $header_lines lines, Content: $content_lines lines, Total: ~$total lines"
        echo ""
    else
        {
            echo "$header"
            echo "$content"
        } > "$dest"
        local lines=$(wc -l < "$dest")
        echo "Created: $dest ($lines lines)"
    fi
}

# =============================================================================
# REDUCTION.AXIS.CS SPLIT (2221 lines -> 4 files)
# =============================================================================

# FILE 1: ILKernelGenerator.Reduction.Axis.cs (REFACTORED)
# Lines: 25-115 (cache+API+dispatcher) + 434-804 (general kernels)

HEADER_AXIS="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Axis.cs - Axis Reduction Core
// =============================================================================
//
// RESPONSIBILITY:
//   - Axis reduction cache and API (TryGetAxisReductionKernel, ClearAxisReduction)
//   - Main dispatcher (CreateAxisReductionKernel)
//   - General axis reduction kernels (scalar loop with type conversion)
//
// RELATED FILES:
//   - ILKernelGenerator.Reduction.Axis.Arg.cs - ArgMax/ArgMin axis
//   - ILKernelGenerator.Reduction.Axis.Simd.cs - Typed SIMD kernels
//   - ILKernelGenerator.Reduction.Axis.VarStd.cs - Var/Std axis
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 1: ILKernelGenerator.Reduction.Axis.NEW.cs ==="
BODY_AXIS=$(sed -n '25,115p' "$AXIS_SRC")
BODY_AXIS+=$'\n'
BODY_AXIS+=$(sed -n '434,804p' "$AXIS_SRC")
BODY_AXIS+=$'\n\n        #endregion'
BODY_AXIS+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.NEW.cs" "$HEADER_AXIS" "$BODY_AXIS"

# FILE 2: ILKernelGenerator.Reduction.Axis.Arg.cs (NEW)
# Lines: 116-428 (ArgMax/ArgMin axis)

HEADER_AXIS_ARG="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Axis.Arg.cs - ArgMax/ArgMin Axis Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisArgReductionKernel - ArgMax/ArgMin dispatcher
//   - CreateAxisArgReductionKernelTyped<T> - typed kernel
//   - AxisArgReductionHelper<T> - SIMD helper
//   - ArgReduceAxis variants (float NaN, double NaN, bool, numeric)
//
// =============================================================================

$NS_OPEN
        #region ArgMax/ArgMin Axis Reduction"

echo "=== FILE 2: ILKernelGenerator.Reduction.Axis.Arg.cs ==="
BODY_AXIS_ARG=$(sed -n '116,428p' "$AXIS_SRC")
BODY_AXIS_ARG+=$'\n\n        #endregion'
BODY_AXIS_ARG+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.Arg.cs" "$HEADER_AXIS_ARG" "$BODY_AXIS_ARG"

# FILE 3: ILKernelGenerator.Reduction.Axis.Simd.cs (NEW)
# Lines: 809-1478 (typed SIMD) + 1480-1584 (IKernelProvider)

HEADER_AXIS_SIMD="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Axis.Simd.cs - SIMD Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisReductionKernelTyped<T> - typed SIMD kernel
//   - AxisReductionSimdHelper<T> - main SIMD helper
//   - ReduceContiguousAxis variants (SIMD256, SIMD128, scalar)
//   - Vector identity/combine/horizontal helpers
//   - IKernelProvider interface implementation
//
// =============================================================================

$NS_OPEN
        #region Typed SIMD Axis Reduction"

echo "=== FILE 3: ILKernelGenerator.Reduction.Axis.Simd.cs ==="
BODY_AXIS_SIMD=$(sed -n '809,1478p' "$AXIS_SRC")
BODY_AXIS_SIMD+=$'\n'
BODY_AXIS_SIMD+=$(sed -n '1480,1584p' "$AXIS_SRC")
BODY_AXIS_SIMD+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.Simd.cs" "$HEADER_AXIS_SIMD" "$BODY_AXIS_SIMD"

# FILE 4: ILKernelGenerator.Reduction.Axis.VarStd.cs (NEW)
# Lines: 1586-2219 (Var/Std axis)

HEADER_AXIS_VARSTD="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Axis.VarStd.cs - Variance/StdDev Axis Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - CreateAxisVarStdReductionKernel - Var/Std dispatcher
//   - AxisVarStdSimdHelper<T> - two-pass SIMD algorithm
//   - SumContiguousAxisDouble<T> - first pass (compute mean)
//   - SumSquaredDiffContiguous<T> - second pass (squared differences)
//   - Decimal and general helpers
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 4: ILKernelGenerator.Reduction.Axis.VarStd.cs ==="
BODY_AXIS_VARSTD=$(sed -n '1586,2219p' "$AXIS_SRC")
BODY_AXIS_VARSTD+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.VarStd.cs" "$HEADER_AXIS_VARSTD" "$BODY_AXIS_VARSTD"

# =============================================================================
# MASKING.CS SPLIT (1343 lines -> 4 files)
# =============================================================================

# FILE 5: ILKernelGenerator.Masking.cs (REFACTORED)
# Lines: 27-248 (NonZero)

HEADER_MASKING="$USINGS

// =============================================================================
// ILKernelGenerator.Masking.cs - NonZero SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - NonZeroSimdHelper<T> - finds indices of non-zero elements
//   - ConvertFlatIndicesToCoordinates - flat indices to per-dimension arrays
//   - FindNonZeroStridedHelper<T> - strided array support
//
// RELATED FILES:
//   - ILKernelGenerator.Masking.Boolean.cs - CountTrue, CopyMasked
//   - ILKernelGenerator.Masking.VarStd.cs - Var/Std SIMD helpers
//   - ILKernelGenerator.Masking.NaN.cs - NaN-aware helpers
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 5: ILKernelGenerator.Masking.NEW.cs ==="
BODY_MASKING=$(sed -n '27,248p' "$MASKING_SRC")
BODY_MASKING+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Masking.NEW.cs" "$HEADER_MASKING" "$BODY_MASKING"

# FILE 6: ILKernelGenerator.Masking.Boolean.cs (NEW)
# Lines: 250-404 (Boolean Masking region start + CountTrue + CopyMasked)

HEADER_MASKING_BOOL="$USINGS

// =============================================================================
// ILKernelGenerator.Masking.Boolean.cs - Boolean Masking SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - CountTrueSimdHelper - count true values in bool array
//   - CopyMaskedElementsHelper<T> - copy elements where mask is true
//
// =============================================================================

$NS_OPEN
        #region Boolean Masking SIMD Helpers"

echo "=== FILE 6: ILKernelGenerator.Masking.Boolean.cs ==="
BODY_MASKING_BOOL=$(sed -n '250,404p' "$MASKING_SRC")
BODY_MASKING_BOOL+=$'\n\n        #endregion'
BODY_MASKING_BOOL+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Masking.Boolean.cs" "$HEADER_MASKING_BOOL" "$BODY_MASKING_BOOL"

# FILE 7: ILKernelGenerator.Masking.VarStd.cs (NEW)
# Lines: 406-676 (VarSimdHelper + StdSimdHelper)

HEADER_MASKING_VARSTD="$USINGS

// =============================================================================
// ILKernelGenerator.Masking.VarStd.cs - Variance/StdDev SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - VarSimdHelper<T> - variance of contiguous array (two-pass algorithm)
//   - StdSimdHelper<T> - standard deviation (sqrt of variance)
//
// NOTE: These are element-wise helpers for full-array Var/Std.
//       For axis reductions, see ILKernelGenerator.Reduction.Axis.VarStd.cs
//
// =============================================================================

$NS_OPEN
        #region Var/Std SIMD Helpers"

echo "=== FILE 7: ILKernelGenerator.Masking.VarStd.cs ==="
BODY_MASKING_VARSTD=$(sed -n '406,676p' "$MASKING_SRC")
BODY_MASKING_VARSTD+=$'\n\n        #endregion'
BODY_MASKING_VARSTD+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Masking.VarStd.cs" "$HEADER_MASKING_VARSTD" "$BODY_MASKING_VARSTD"

# FILE 8: ILKernelGenerator.Masking.NaN.cs (NEW)
# Lines: 678-1341 (NaN helpers)

HEADER_MASKING_NAN="$USINGS

// =============================================================================
// ILKernelGenerator.Masking.NaN.cs - NaN-aware SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - NanSumSimdHelperFloat/Double - sum ignoring NaN values
//   - NanProdSimdHelperFloat/Double - product ignoring NaN values
//   - NanMinSimdHelperFloat/Double - min ignoring NaN values
//   - NanMaxSimdHelperFloat/Double - max ignoring NaN values
//
// =============================================================================

$NS_OPEN
        #region NaN-aware SIMD Helpers"

echo "=== FILE 8: ILKernelGenerator.Masking.NaN.cs ==="
BODY_MASKING_NAN=$(sed -n '678,1341p' "$MASKING_SRC")
BODY_MASKING_NAN+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Masking.NaN.cs" "$HEADER_MASKING_NAN" "$BODY_MASKING_NAN"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================================="
if $DRY_RUN; then
    echo "DRY RUN COMPLETE - No files were created"
    echo ""
    echo "Line coverage verification:"
    echo "  Reduction.Axis.cs (2221 lines):"
    echo "    25-115   (91 lines)  -> Axis.NEW.cs (cache+API+dispatcher)"
    echo "    116-428  (313 lines) -> Axis.Arg.cs (ArgMax/ArgMin)"
    echo "    434-804  (371 lines) -> Axis.NEW.cs (general kernels)"
    echo "    809-1478 (670 lines) -> Axis.Simd.cs (typed SIMD)"
    echo "    1480-1584 (105 lines) -> Axis.Simd.cs (IKernelProvider)"
    echo "    1586-2219 (634 lines) -> Axis.VarStd.cs"
    echo ""
    echo "  Masking.cs (1343 lines):"
    echo "    27-248   (222 lines) -> Masking.NEW.cs (NonZero)"
    echo "    250-404  (155 lines) -> Masking.Boolean.cs (CountTrue+CopyMasked)"
    echo "    406-676  (271 lines) -> Masking.VarStd.cs"
    echo "    678-1341 (664 lines) -> Masking.NaN.cs"
    echo ""
    echo "To execute the split, run:"
    echo "  bash scripts/split-ilkernel-phase2.sh"
else
    echo "SPLIT COMPLETE"
    echo ""
    echo "New files created with .NEW.cs suffix to preserve originals."
    echo ""
    echo "Next steps:"
    echo "  1. Build: dotnet build src/NumSharp.Core/NumSharp.Core.csproj -v q --nologo"
    echo "  2. If successful, swap files:"
    echo "     rm $AXIS_SRC $MASKING_SRC"
    echo "     mv $KERNELS_DIR/ILKernelGenerator.Reduction.Axis.NEW.cs $AXIS_SRC"
    echo "     mv $KERNELS_DIR/ILKernelGenerator.Masking.NEW.cs $MASKING_SRC"
fi

echo ""
echo "Files in $KERNELS_DIR:"
ls -la "$KERNELS_DIR"/ILKernelGenerator.*.cs 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}' | sed "s|$KERNELS_DIR/||"
