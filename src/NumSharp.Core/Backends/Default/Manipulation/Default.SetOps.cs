// NumPy's set routines np.union1d, np.setxor1d and np.setdiff1d (numpy/lib/_arraysetops_impl.py) as DefaultEngine's
// members, with the NaN pass that makes their float32 / float64 results NumPy's bits. NumPy's statements call np.unique,
// np.concatenate, np.ravel, np.sort, np.isin, np.isnan, np.any and np.where; here they are this engine's or the
// operand's engine members (np.unique's Unique with its defaults, values only) and the array object's (the ravel, the
// concatenation, the one-element flag, NDArray.where).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped]
        public override NDArray Union1d(NDArray ar1, NDArray ar2)
        {
            // Text: NumPy's statement below runs unchanged — the concatenation formats numbers into text and decodes bytes
            // beside str, and np.unique sorts the text (CanonicalizeSetOpNaN leaves a text result as it is).
            // Boundary scope: the ravel views, the concatenate buffer and unique's internals are
            // reclaimed at exit; only the final (possibly NaN-canonicalised) array is yielded.
            // NumPy: unique(concatenate((ar1, ar2), axis=None)) — axis=None flattens both operands.
            // NumPy's unique-sort canonicalises a surviving float32/float64 NaN; match it (see CanonicalizeSetOpNaN).
            return CanonicalizeSetOpNaN(Unique(
                NDArray.concatenate(new[] { NDArray.ravel(ar1, 'C'), NDArray.ravel(ar2, 'C') }, 0, null, null, "same_kind"),
                false, false, false, null, true, true)[0]);
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Setxor1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            // Text runs the statements below unchanged: np.unique and np.sort order the text, the concatenation formats
            // numbers and decodes bytes as NumPy's does, and the text != builds the flags.
            // Boundary scope: the unique/ravel/concatenate/sort temps and the flag masks are
            // reclaimed at exit; only the surviving-values array is yielded.

            if (!assume_unique)
            {
                ar1 = Unique(ar1, false, false, false, null, true, true)[0];
                ar2 = Unique(ar2, false, false, false, null, true, true)[0];
            }

            // concatenate((ar1, ar2), axis=None) — flattens both.
            NDArray aux = NDArray.concatenate(new[] { NDArray.ravel(ar1, 'C'), NDArray.ravel(ar2, 'C') }, 0, null, null, "same_kind");
            if (aux.size == 0)
                return aux;

            aux = Sort(aux, -1, null, null, null);

            // flag = concatenate(([True], aux[1:] != aux[:-1], [True]))
            NDArray trueOne = new NDArray(Unmanaged.ArraySlice.FromArray(new[] { true }, true), Shape.Vector(1));   // np.array([True])
            NDArray neq = aux["1:"] != aux[":-1"];
            NDArray flag = NDArray.concatenate(new[] { trueOne, neq, trueOne }, 0, null, null, "same_kind");

            // return aux[flag[1:] & flag[:-1]]  — values appearing exactly once survive
            NDArray sel = flag["1:"] & flag[":-1"];
            // aux came from np.sort; NumPy's sort canonicalises a surviving float32/float64 NaN (see CanonicalizeSetOpNaN).
            return CanonicalizeSetOpNaN(aux[sel]);
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Setdiff1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            // Text runs the statements below unchanged: np.unique sorts the text and np.isin runs NumPy's _isin over it.
            // Boundary scope: the unique/ravel temps and the isin membership mask are reclaimed
            // at exit; only the selected-values array is yielded.

            if (assume_unique)
            {
                ar1 = NDArray.ravel(ar1, 'C');
            }
            else
            {
                ar1 = Unique(ar1, false, false, false, null, true, true)[0];
                ar2 = Unique(ar2, false, false, false, null, true, true)[0];
            }

            // ar1[isin(ar1, ar2, assume_unique=True, invert=True)] — keep only the values absent from ar2.
            NDArray mask = Isin(ar1, ar2, true, true, null);   // np.isin(ar1, ar2, assume_unique=True, invert=True)
            // ar1 came from np.unique; NumPy canonicalises a surviving float32/float64 NaN (see CanonicalizeSetOpNaN).
            return CanonicalizeSetOpNaN(ar1[mask]);
        }

        /// <summary>
        ///     Rewrite the NaN lanes of a set-operation result to the exact bits NumPy produces, so
        ///     <c>np.union1d</c> / <c>np.setxor1d</c> / <c>np.setdiff1d</c> are bit-identical
        ///     to NumPy 2.4.2 even for NaN-bearing float input.
        /// </summary>
        /// <remarks>
        ///     NumPy's set operations route their surviving values through the SIMD quicksort/heapsort
        ///     that <c>np.unique</c>/<c>np.sort</c> use, and that sort CANONICALISES every NaN — signalling,
        ///     quiet, negative-signed, or payload-bearing — to the single positive quiet NaN, but ONLY for
        ///     float32/float64. float16 and complex128 NaNs are left exactly as they are (probed against
        ///     NumPy 2.4.2: f16 <c>0x7E00/0xFE00/0x7C01/…</c> and a complex negative-NaN real part
        ///     <c>0xFFF8…</c> all pass through unchanged). <c>np.sort</c> writes NumPy's canonical NaN too (since
        ///     2026-10-05; it wrote .NET's negative <c>0xFFF8…</c> before), but <c>np.unique</c>, which these
        ///     set operations go through, preserves the input payload — so a NaN that survives a set operation
        ///     would diverge. This restores parity by rewriting only the NaN lanes of a float32/float64 result to
        ///     NumPy's canonical value.
        ///
        ///     <para><c>isin</c> returns bool (no NaN) and <c>intersect1d</c> never emits a NaN (NaN != NaN,
        ///     so a shared-value adjacency check never selects one), so neither needs this pass.</para>
        /// </remarks>
        /// <param name="result">A set operation's result.</param>
        /// <returns><paramref name="result"/> itself when it is not float32 / float64, empty or free of NaN; otherwise a
        ///     new array with every NaN lane NumPy's canonical positive quiet NaN.</returns>
        [NDScopedCovered] // only reached from the [NDScoped] Union1d / Setxor1d / Setdiff1d members; the canon scalar + mask ride their scope
        private static NDArray CanonicalizeSetOpNaN(NDArray result)
        {
            // Only float32/float64 are canonicalised by NumPy's sort; f16/complex preserve, ints have no NaN.
            if (result.typecode != NPTypeCode.Single && result.typecode != NPTypeCode.Double)
                return result;
            if (result.size == 0)
                return result;

            NDArray mask = result.TensorEngine.IsNan(result, null, null, null);
            if (!mask.TensorEngine.Any(mask))               // common case: no NaN survived -> untouched
                return result;

            // The exact bits NumPy's float sort produces: 0x7FF8000000000000 (f64) / 0x7FC00000 (f32).
            NDArray canon = result.typecode == NPTypeCode.Double
                ? NDArray.Scalar(BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL))
                : NDArray.Scalar(BitConverter.UInt32BitsToSingle(0x7FC00000U));
            return NDArray.where(mask, canon, result);
        }
    }
}
