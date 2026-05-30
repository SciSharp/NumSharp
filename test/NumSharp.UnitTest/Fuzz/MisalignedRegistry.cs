using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    public enum DivergenceKind { Dtype, Shape, Value }

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
        public static string Classify(
            FuzzCorpus.Case c, DivergenceKind kind,
            byte[] expected, byte[] actual, NPTypeCode tc, IReadOnlyList<BitDiff.Diff> diffs)
        {
            // (1) NEP50 weak-scalar promotion. The pp_scalar_* layouts are constructed with exactly
            //     one 0-D operand, so a result-dtype divergence there IS the documented weak-scalar rule.
            if (kind == DivergenceKind.Dtype && (c.Layout == "pp_scalar_left" || c.Layout == "pp_scalar_right"))
                return "NEP50 weak-scalar: 0-D operand promoted weakly (NumPy promotes 0-D arrays fully)";

            // (2) Complex true-division ~1 ULP. Excuse only divide, only complex result, only when every
            //     differing element is within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Op == "divide" && tc == NPTypeCode.Complex
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex)";

            return null;
        }
    }
}
