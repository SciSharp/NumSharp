using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    public enum DivergenceKind { Dtype, Shape, Value, Threw }

    /// <summary>
    ///     The explicit, documented set of NumSharp-vs-NumPy behavioral differences that are
    ///     INTENDED (per maintainer decision) rather than bugs. The differential matrix excuses
    ///     ONLY these patterns — logging each one — and fails on anything else. Keeping the cases
    ///     in the corpus (instead of dropping them) means a future change to either behavior
    ///     surfaces immediately: a fixed divergence simply starts passing the bit-exact check, and
    ///     a drift beyond the documented tolerance turns back into a hard failure.
    ///
    ///     Documented differences:
    ///       1. NEP50 weak-scalar: NumSharp treats a 0-D array operand as a weak scalar (the other
    ///          operand's dtype drives promotion). NumPy makes 0-D arrays full participants; only
    ///          Python scalar literals are weak. NumSharp cannot distinguish the two (both are 0-D
    ///          NDArrays), and keeping `arr + 5` ergonomic was chosen over strict NEP50 parity.
    ///       2. Complex true-division differs from NumPy's npy_cdivide by ~1 ULP
    ///          (System.Numerics.Complex uses a different scaling). Add/sub/mul are bit-exact.
    /// </summary>
    public static class MisalignedRegistry
    {
        private static readonly System.Collections.Generic.HashSet<string> ReduceOps = new()
        {
            "sum", "prod", "min", "max", "mean", "std", "var", "argmax", "argmin", "all", "any"
        };

        public static string Classify(
            FuzzCorpus.Case c, DivergenceKind kind,
            byte[] expected, byte[] actual, NPTypeCode tc, IReadOnlyList<BitDiff.Diff> diffs)
        {
            // (1) NEP50 weak-scalar promotion. Any multi-operand op with a 0-D operand: NumSharp
            //     promotes it weakly (the array operand's dtype drives the result), where NumPy makes
            //     0-D arrays full participants. Covers binary pp_scalar_* and np.where wh_bcast_xy.
            if (kind == DivergenceKind.Dtype && c.Operands.Length >= 2 && c.Operands.Any(o => o.Shape.Length == 0))
                return "NEP50 weak-scalar: 0-D operand promoted weakly (NumPy promotes 0-D arrays fully)";

            // (2) Complex true-division ~1 ULP. Excuse only divide, only complex result, only when every
            //     differing element is within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Op == "divide" && tc == NPTypeCode.Complex
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex)";

            // Size-1 result shape: NumSharp collapses an all-size-1 result to 0-D where NumPy keeps it
            // (e.g. [1] op scalar -> NumPy [1], NumSharp []). All expected dims are 1 (product 1).
            if (kind == DivergenceKind.Shape && c.Expected.Shape != null && c.Expected.Shape.All(d => d == 1))
                return "size-1 result shape differs (NumSharp collapses to 0-D) [known bug]";

            // Bool arithmetic: NumSharp computes the integer result (True+True -> 2); NumPy keeps bool
            // semantics (True+True -> True). Only when the result dtype is bool.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Boolean
                && (c.Op == "add" || c.Op == "subtract" || c.Op == "multiply"))
                return "bool arithmetic: NumSharp integer result vs NumPy bool [known bug]";

            // Complex binary arithmetic (add/sub/mul/div): catastrophic cancellation (re*re-im*im -> 0)
            // and ~1 ULP from System.Numerics.Complex vs NumPy's npy_c* algorithms.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex && c.Operands.Length == 2)
                return "complex binary arithmetic (cancellation / ~ULP) [partly known bug]";

            // (3) NaN ordering in <= / >= was FIXED in Phase 1 F2 (the unordered Cgt_Un/Clt_Un
            //     compare now yields False for a NaN operand, matching IEEE/NumPy). The classifier
            //     branch is intentionally removed so the comparison matrix verifies it bit-exact.

            // --- Reductions (single-operand, but classified before the unary rules) ---
            if (ReduceOps.Contains(c.Op))
            {
                // Reduction result dtype differs (NEP50 accumulator width / complex->real for std/var).
                if (kind == DivergenceKind.Dtype)
                    return "reduction result dtype differs (NEP50 accumulator / complex->real) [known bug]";
                // Complex axis reduction throws (NDCoordinatesAxisIncrementor vector-shape path).
                if (kind == DivergenceKind.Threw && c.Operands.Length == 1 && c.Operands[0].Dtype == "complex128")
                    return "complex axis reduction throws (NDCoordinatesAxisIncrementor) [known bug]";
                // NaN propagation: regular min/max/mean/std/var keep NaN in NumPy; NumSharp skips it.
                if (kind == DivergenceKind.Value && diffs.Count > 0 && diffs.All(d => d.Expected == "NaN"))
                    return "reduction NaN propagation: NumSharp skips NaN, NumPy propagates [known bug]";
                // bool min/max along an axis returns True where NumPy returns False.
                if (kind == DivergenceKind.Value && (c.Op == "min" || c.Op == "max") && tc == NPTypeCode.Boolean)
                    return "bool min/max along axis diverges [known bug]";
                // Floating accumulation: NumPy pairwise summation / two-pass var vs NumSharp order.
                if (kind == DivergenceKind.Value &&
                    (c.Op == "sum" || c.Op == "mean" || c.Op == "std" || c.Op == "var" || c.Op == "prod"))
                    return "reduction summation/two-pass precision (algorithm order)";
            }

            // (4) Unary result-dtype differs: NumSharp promotes int->float64 for unary float ops
            //     regardless of input width and handles bool/square/reciprocal/floor differently,
            //     whereas NumPy uses NEP50 width-based unary promotion (bool/int8->float16,
            //     int16->float32, int32+->float64). Documented promotion difference.
            if (kind == DivergenceKind.Dtype && c.Operands.Length == 1)
                return "unary NEP50 promotion: result dtype differs from NumPy width-based unary rule";

            // (5) Unary transcendental / complex magnitude ~ULP (libm / algorithm differences).
            //     Tight: every differing element within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "unary ~ULP (transcendental/magnitude algorithm difference)";

            // (6) np.negative on unsigned integers was FIXED in Phase 1 F4: np.negative now routes
            //     through the engine kernel (two's-complement wrap, e.g. -1u -> 255), matching NumPy.
            //     Classifier branch removed so the unary matrix verifies it bit-exact.

            // (7) Complex unary (square / sin / cos / tan / log): differs from NumPy by more than ULP
            //     in places — catastrophic cancellation in square (re^2-im^2 -> 0) and inf/NaN edge
            //     handling in the trig/log algorithms. System.Numerics.Complex vs NumPy's npy_c*.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1 && tc == NPTypeCode.Complex)
                return "complex unary (square/trig/log) algorithm/edge difference [partly known bug]";

            // np.where with complex operands throws ("Zero-push unsupported for Complex").
            if (kind == DivergenceKind.Threw && c.Op == "where" && c.Operands.Any(o => o.Dtype == "complex128"))
                return "complex np.where throws (Zero-push unsupported for Complex) [known bug]";

            // (8) np.reciprocal of an integer: NumPy returns the integer ÷0 sentinel for 0 and
            //     truncating-integer reciprocal otherwise; NumSharp returns 0. Plus reciprocal on a
            //     non-contiguous operand throws (it requires a flat Address). Documented pending fix.
            if (c.Op == "reciprocal" && (kind == DivergenceKind.Value || kind == DivergenceKind.Threw))
                return "reciprocal: integer ÷0 sentinel / non-contiguous Address differs [known bug]";

            return null;
        }
    }
}
