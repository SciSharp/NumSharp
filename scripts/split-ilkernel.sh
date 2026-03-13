#!/bin/bash
# =============================================================================
# ILKernelGenerator Split Script
# =============================================================================
# Splits ILKernelGenerator.Reduction.cs (5338 lines) and ILKernelGenerator.Unary.cs (1915 lines)
# into 11 smaller, focused partial class files.
#
# Usage:
#   ./split-ilkernel.sh --dry-run    # Preview what will be created
#   ./split-ilkernel.sh              # Execute the split
#
# VERIFIED LINE BOUNDARIES (closing braces from grep -n "^        }$"):
#   Reduction.cs:
#     740  - end EmitReductionStridedLoop
#     919  - end AnySimdHelper
#     1206 - end ArgMinSimdHelper
#     2523 - end Boolean Masking section (blank line)
#     3140 - #endregion IL Helpers
#     5336 - end Var/Std section
#   Unary.cs:
#     622  - end EmitUnaryStridedLoop
#     1172 - end EmitRad2DegCall
#     1256 - end EmitIsInfCall
#     1504 - end EmitUnaryDecimalOperation
#     1766 - #endregion Unary Kernel Generation
#     1913 - end Scalar section
# =============================================================================

set -e

# Configuration
KERNELS_DIR="src/NumSharp.Core/Backends/Kernels"
REDUCTION_SRC="$KERNELS_DIR/ILKernelGenerator.Reduction.cs"
UNARY_SRC="$KERNELS_DIR/ILKernelGenerator.Unary.cs"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be created ==="
    echo ""
fi

# Verify source files exist
if [[ ! -f "$REDUCTION_SRC" ]]; then
    echo "ERROR: $REDUCTION_SRC not found"
    exit 1
fi
if [[ ! -f "$UNARY_SRC" ]]; then
    echo "ERROR: $UNARY_SRC not found"
    exit 1
fi

# Verify line counts match expectations
REDUCTION_LINES=$(wc -l < "$REDUCTION_SRC")
UNARY_LINES=$(wc -l < "$UNARY_SRC")

echo "Source file verification:"
echo "  $REDUCTION_SRC: $REDUCTION_LINES lines (expected: 5338)"
echo "  $UNARY_SRC: $UNARY_LINES lines (expected: 1915)"
echo ""

if [[ "$REDUCTION_LINES" -ne 5338 ]]; then
    echo "WARNING: Reduction.cs line count mismatch! Expected 5338, got $REDUCTION_LINES"
    echo "Line ranges may be incorrect. Aborting."
    exit 1
fi
if [[ "$UNARY_LINES" -ne 1915 ]]; then
    echo "WARNING: Unary.cs line count mismatch! Expected 1915, got $UNARY_LINES"
    echo "Line ranges may be incorrect. Aborting."
    exit 1
fi

echo "Line counts verified. Proceeding..."
echo ""

# =============================================================================
# BOUNDARY VERIFICATION
# =============================================================================

echo "=== Boundary Verification ==="

# Verify key lines contain expected content
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

echo "Reduction.cs boundaries:"
verify_line "$REDUCTION_SRC" 107 "/// <summary>"
verify_line "$REDUCTION_SRC" 740 "}"
verify_line "$REDUCTION_SRC" 742 "/// <summary>"
verify_line "$REDUCTION_SRC" 919 "}"
verify_line "$REDUCTION_SRC" 921 "/// <summary>"
verify_line "$REDUCTION_SRC" 1206 "}"
verify_line "$REDUCTION_SRC" 1208 "#region NonZero"
verify_line "$REDUCTION_SRC" 2522 "#endregion"
verify_line "$REDUCTION_SRC" 2524 "#region Reduction IL Helpers"
verify_line "$REDUCTION_SRC" 3140 "#endregion"
verify_line "$REDUCTION_SRC" 3142 "#region Axis Reduction"
verify_line "$REDUCTION_SRC" 5336 "#endregion"

echo ""
echo "Unary.cs boundaries:"
verify_line "$UNARY_SRC" 96 "#region Unary Kernel"
verify_line "$UNARY_SRC" 622 "}"
verify_line "$UNARY_SRC" 624 "/// <summary>"
verify_line "$UNARY_SRC" 1172 "}"
verify_line "$UNARY_SRC" 1174 "/// <summary>"
verify_line "$UNARY_SRC" 1256 "}"
verify_line "$UNARY_SRC" 1258 "/// <summary>"
verify_line "$UNARY_SRC" 1504 "}"
verify_line "$UNARY_SRC" 1506 "/// <summary>"
verify_line "$UNARY_SRC" 1766 "#endregion"
verify_line "$UNARY_SRC" 1768 "#region Scalar"
verify_line "$UNARY_SRC" 1913 "#endregion"

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
# FILE 1: ILKernelGenerator.Reduction.cs (REFACTORED)
# Lines: 107-740 (element reduction) + 2524-3140 (IL helpers)
# =============================================================================

HEADER_REDUCTION="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.cs - Element-wise Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - Element-wise reduction kernel generation (Sum, Prod, Min, Max, Mean)
//   - SIMD loop emission with 4x unrolling
//   - Scalar and strided fallback loops
//   - Shared IL emission helpers for reduction operations
//
// RELATED FILES:
//   - ILKernelGenerator.Reduction.Boolean.cs - All/Any with early-exit
//   - ILKernelGenerator.Reduction.Arg.cs - ArgMax/ArgMin
//   - ILKernelGenerator.Reduction.Axis.cs - Axis reductions
//   - ILKernelGenerator.Masking.cs - NonZero and boolean masking
//
// =============================================================================

$NS_OPEN
        #region Element Reduction Kernel Generation"

echo "=== FILE 1: ILKernelGenerator.Reduction.NEW.cs ==="
BODY_REDUCTION=$(sed -n '107,740p' "$REDUCTION_SRC")
BODY_REDUCTION+=$'\n'
BODY_REDUCTION+=$(sed -n '2524,3140p' "$REDUCTION_SRC")
BODY_REDUCTION+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.NEW.cs" "$HEADER_REDUCTION" "$BODY_REDUCTION"

# =============================================================================
# FILE 2: ILKernelGenerator.Reduction.Boolean.cs (NEW)
# Lines: 742-919 (All/Any with early-exit)
# =============================================================================

HEADER_REDUCTION_BOOL="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Boolean.cs - All/Any Boolean Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - All/Any reduction with early-exit SIMD optimization
//   - EmitAllAnySimdLoop() - IL emission for All/Any
//   - AllSimdHelper<T>() - returns false immediately on first zero
//   - AnySimdHelper<T>() - returns true immediately on first non-zero
//
// =============================================================================

$NS_OPEN
        #region Boolean Reduction Helpers (All/Any)"

echo "=== FILE 2: ILKernelGenerator.Reduction.Boolean.cs ==="
BODY_REDUCTION_BOOL=$(sed -n '742,919p' "$REDUCTION_SRC")
BODY_REDUCTION_BOOL+=$'\n\n        #endregion'
BODY_REDUCTION_BOOL+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Boolean.cs" "$HEADER_REDUCTION_BOOL" "$BODY_REDUCTION_BOOL"

# =============================================================================
# FILE 3: ILKernelGenerator.Reduction.Arg.cs (NEW)
# Lines: 921-1206 (ArgMax/ArgMin)
# =============================================================================

HEADER_REDUCTION_ARG="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Arg.cs - ArgMax/ArgMin Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - ArgMax/ArgMin reduction with SIMD index tracking
//   - Two-pass algorithm: find extreme value with SIMD, then find index
//   - EmitArgMaxMinSimdLoop() - IL emission
//   - ArgMaxSimdHelper<T>(), ArgMinSimdHelper<T>() - SIMD helpers
//
// =============================================================================

$NS_OPEN
        #region ArgMax/ArgMin Reduction Helpers"

echo "=== FILE 3: ILKernelGenerator.Reduction.Arg.cs ==="
BODY_REDUCTION_ARG=$(sed -n '921,1206p' "$REDUCTION_SRC")
BODY_REDUCTION_ARG+=$'\n\n        #endregion'
BODY_REDUCTION_ARG+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Arg.cs" "$HEADER_REDUCTION_ARG" "$BODY_REDUCTION_ARG"

# =============================================================================
# FILE 4: ILKernelGenerator.Reduction.Axis.cs (NEW)
# Lines: 3142-5336 (Axis reductions + IKernelProvider + Var/Std)
# =============================================================================

HEADER_REDUCTION_AXIS="$USINGS

// =============================================================================
// ILKernelGenerator.Reduction.Axis.cs - Axis Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - Axis reduction kernel generation (reduce along specific axis)
//   - Variance/StdDev axis reductions (two-pass algorithm)
//   - IKernelProvider interface implementation
//   - SIMD helpers for contiguous and strided axis operations
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 4: ILKernelGenerator.Reduction.Axis.cs ==="
BODY_REDUCTION_AXIS=$(sed -n '3142,5336p' "$REDUCTION_SRC")
BODY_REDUCTION_AXIS+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Reduction.Axis.cs" "$HEADER_REDUCTION_AXIS" "$BODY_REDUCTION_AXIS"

# =============================================================================
# FILE 5: ILKernelGenerator.Masking.cs (NEW)
# Lines: 1208-2523 (NonZero + Boolean masking)
# =============================================================================

HEADER_MASKING="$USINGS

// =============================================================================
// ILKernelGenerator.Masking.cs - NonZero and Boolean Masking Operations
// =============================================================================
//
// RESPONSIBILITY:
//   - NonZero SIMD helpers for np.nonzero
//   - Boolean masking SIMD helpers for fancy indexing
//   - CountTrueSimdHelper, CopyMaskedElementsHelper
//   - NaN-aware helpers (NanSum, NanProd, NanMin, NanMax)
//
// NOTE: These are selection/masking operations, NOT reductions.
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 5: ILKernelGenerator.Masking.cs ==="
BODY_MASKING=$(sed -n '1208,2522p' "$REDUCTION_SRC")
BODY_MASKING+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Masking.cs" "$HEADER_MASKING" "$BODY_MASKING"

# =============================================================================
# FILE 6: ILKernelGenerator.Unary.cs (REFACTORED)
# Lines: 96-622 (core unary infrastructure + loop emission)
# =============================================================================

HEADER_UNARY="$USINGS

// =============================================================================
// ILKernelGenerator.Unary.cs - Unary Kernel Infrastructure
// =============================================================================
//
// RESPONSIBILITY:
//   - Unary kernel cache and API (GetUnaryKernel, TryGetUnaryKernel, ClearUnary)
//   - SIMD loop emission with 4x unrolling
//   - Scalar and strided fallback loops
//   - Capability detection (CanUseUnarySimd, IsPredicateOp)
//
// RELATED FILES:
//   - ILKernelGenerator.Unary.Math.cs - Math function emission
//   - ILKernelGenerator.Unary.Predicate.cs - IsNaN/IsFinite/IsInf
//   - ILKernelGenerator.Unary.Decimal.cs - Decimal operations
//   - ILKernelGenerator.Unary.Vector.cs - SIMD vector operations
//   - ILKernelGenerator.Scalar.cs - Scalar kernel delegates
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 6: ILKernelGenerator.Unary.NEW.cs ==="
BODY_UNARY=$(sed -n '96,622p' "$UNARY_SRC")
BODY_UNARY+=$'\n\n        #endregion'
BODY_UNARY+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Unary.NEW.cs" "$HEADER_UNARY" "$BODY_UNARY"

# =============================================================================
# FILE 7: ILKernelGenerator.Unary.Math.cs (NEW)
# Lines: 624-1172 (EmitUnaryScalarOperation + math helpers)
# =============================================================================

HEADER_UNARY_MATH="$USINGS

// =============================================================================
// ILKernelGenerator.Unary.Math.cs - Math Function IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryScalarOperation - main dispatch for all unary ops
//   - EmitMathCall - Math.X/MathF.X function emission
//   - Trig functions: Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh
//   - Exp/Log: Exp, Exp2, Log, Log2, Log10, Log1p, Expm1
//   - Rounding: Floor, Ceil, Round, Truncate
//   - Sign, Reciprocal, Deg2Rad, Rad2Deg
//
// =============================================================================

$NS_OPEN
        #region Unary Math IL Emission"

echo "=== FILE 7: ILKernelGenerator.Unary.Math.cs ==="
BODY_UNARY_MATH=$(sed -n '624,1172p' "$UNARY_SRC")
BODY_UNARY_MATH+=$'\n\n        #endregion'
BODY_UNARY_MATH+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Unary.Math.cs" "$HEADER_UNARY_MATH" "$BODY_UNARY_MATH"

# =============================================================================
# FILE 8: ILKernelGenerator.Unary.Predicate.cs (NEW)
# Lines: 1174-1256 (IsFinite, IsNaN, IsInf)
# =============================================================================

HEADER_UNARY_PRED="$USINGS

// =============================================================================
// ILKernelGenerator.Unary.Predicate.cs - Predicate IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitIsFiniteCall - float.IsFinite/double.IsFinite
//   - EmitIsNanCall - float.IsNaN/double.IsNaN
//   - EmitIsInfCall - float.IsInfinity/double.IsInfinity
//   - Integer types always return true/false appropriately
//
// =============================================================================

$NS_OPEN
        #region Unary Predicate IL Emission"

echo "=== FILE 8: ILKernelGenerator.Unary.Predicate.cs ==="
BODY_UNARY_PRED=$(sed -n '1174,1256p' "$UNARY_SRC")
BODY_UNARY_PRED+=$'\n\n        #endregion'
BODY_UNARY_PRED+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Unary.Predicate.cs" "$HEADER_UNARY_PRED" "$BODY_UNARY_PRED"

# =============================================================================
# FILE 9: ILKernelGenerator.Unary.Decimal.cs (NEW)
# Lines: 1258-1504 (Decimal operations)
# =============================================================================

HEADER_UNARY_DEC="$USINGS

// =============================================================================
// ILKernelGenerator.Unary.Decimal.cs - Decimal IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryDecimalOperation - all decimal unary operations
//   - Negate, Abs, Sign, Ceiling, Floor, Round, Truncate
//   - Sqrt, trig functions via double conversion
//
// =============================================================================

$NS_OPEN
        #region Unary Decimal IL Emission"

echo "=== FILE 9: ILKernelGenerator.Unary.Decimal.cs ==="
BODY_UNARY_DEC=$(sed -n '1258,1504p' "$UNARY_SRC")
BODY_UNARY_DEC+=$'\n\n        #endregion'
BODY_UNARY_DEC+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Unary.Decimal.cs" "$HEADER_UNARY_DEC" "$BODY_UNARY_DEC"

# =============================================================================
# FILE 10: ILKernelGenerator.Unary.Vector.cs (NEW)
# Lines: 1506-1766 (SIMD vector operations)
# =============================================================================

HEADER_UNARY_VEC="$USINGS

// =============================================================================
// ILKernelGenerator.Unary.Vector.cs - SIMD Vector IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryVectorOperation - main vector op dispatch
//   - EmitVectorSquare - x * x
//   - EmitVectorReciprocal - 1 / x
//   - EmitVectorDeg2Rad, EmitVectorRad2Deg - angle conversion
//   - EmitVectorBitwiseNot - ~x
//
// =============================================================================

$NS_OPEN
        #region Unary Vector IL Emission"

echo "=== FILE 10: ILKernelGenerator.Unary.Vector.cs ==="
BODY_UNARY_VEC=$(sed -n '1506,1766p' "$UNARY_SRC")
BODY_UNARY_VEC+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Unary.Vector.cs" "$HEADER_UNARY_VEC" "$BODY_UNARY_VEC"

# =============================================================================
# FILE 11: ILKernelGenerator.Scalar.cs (NEW - from Unary.cs)
# Lines: 1768-1913 (Scalar kernel delegates)
# =============================================================================

HEADER_SCALAR="$USINGS

// =============================================================================
// ILKernelGenerator.Scalar.cs - Scalar Kernel Delegates
// =============================================================================
//
// RESPONSIBILITY:
//   - Unary scalar kernels: Func<TInput, TOutput>
//   - Binary scalar kernels: Func<TLhs, TRhs, TResult>
//   - Used for single-value operations in broadcasting
//
// =============================================================================

$NS_OPEN"

echo "=== FILE 11: ILKernelGenerator.Scalar.cs ==="
BODY_SCALAR=$(sed -n '1768,1913p' "$UNARY_SRC")
BODY_SCALAR+=$'\n'"$NS_CLOSE"

create_file "$KERNELS_DIR/ILKernelGenerator.Scalar.cs" "$HEADER_SCALAR" "$BODY_SCALAR"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================================="
if $DRY_RUN; then
    echo "DRY RUN COMPLETE - No files were created"
    echo ""
    echo "Line coverage verification:"
    echo "  Reduction.cs (5338 lines):"
    echo "    107-740   (634 lines) -> Reduction.NEW.cs"
    echo "    742-919   (178 lines) -> Reduction.Boolean.cs"
    echo "    921-1206  (286 lines) -> Reduction.Arg.cs"
    echo "    1208-2522 (1315 lines) -> Masking.cs"
    echo "    2524-3140 (617 lines) -> Reduction.NEW.cs (IL helpers)"
    echo "    3142-5336 (2195 lines) -> Reduction.Axis.cs"
    echo "    Total content: 5226 lines (+ headers/gaps = 5338)"
    echo ""
    echo "  Unary.cs (1915 lines):"
    echo "    96-622    (527 lines) -> Unary.NEW.cs"
    echo "    624-1172  (549 lines) -> Unary.Math.cs"
    echo "    1174-1256 (83 lines)  -> Unary.Predicate.cs"
    echo "    1258-1504 (247 lines) -> Unary.Decimal.cs"
    echo "    1506-1766 (261 lines) -> Unary.Vector.cs"
    echo "    1768-1913 (146 lines) -> Scalar.cs"
    echo "    Total content: 1813 lines (+ headers/gaps = 1915)"
    echo ""
    echo "To execute the split, run:"
    echo "  bash scripts/split-ilkernel.sh"
else
    echo "SPLIT COMPLETE"
    echo ""
    echo "New files created with .NEW.cs suffix for Reduction/Unary to preserve originals."
    echo ""
    echo "Next steps:"
    echo "  1. Build: dotnet build -v q --nologo"
    echo "  2. If successful:"
    echo "     - Delete originals: rm $REDUCTION_SRC $UNARY_SRC"
    echo "     - Rename: mv $KERNELS_DIR/ILKernelGenerator.Reduction.NEW.cs $REDUCTION_SRC"
    echo "     - Rename: mv $KERNELS_DIR/ILKernelGenerator.Unary.NEW.cs $UNARY_SRC"
    echo "  3. Build and test again"
fi

echo ""
echo "Files in $KERNELS_DIR:"
ls -la "$KERNELS_DIR"/ILKernelGenerator.*.cs 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}' | sed "s|$KERNELS_DIR/||"
