// The functions of numpy/_core/einsumfunc.py that compute: np.einsum and np.einsum_path. NumPy writes einsum in Python
// (its optimize= path, bmm_einsum) over the C core's c_einsum; here it is an engine member that DefaultEngine implements
// with that path's statements over the matrix products. np.einsum_path computes the contraction plan and its cost report
// from the operands' shapes (the plan's D8 counts it among the 12 that compute), so it is an engine member too.

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe',
        ///     optimize=False)</c>: the Einstein summation convention over <paramref name="operands"/>, the whole call
        ///     from the argument checks to the copy into <paramref name="out"/>.
        /// </summary>
        /// <param name="subscripts">Comma-separated subscript labels, optionally followed by <c>-&gt;</c> and the
        ///     output labels (<c>"ij,jk-&gt;ik"</c>); spaces are ignored and <c>...</c> broadcasts. Without
        ///     <c>-&gt;</c> the output is every label used once, in ASCII order, after the broadcast dimensions.
        /// </param>
        /// <param name="operands">The arrays the subscripts label, in order (the facade turns NumPy's sublist spelling
        ///     into a subscripts string first).</param>
        /// <param name="out">An array to receive the result, or null for a new array. Its rank is checked with the
        ///     subscripts, its shape and dtype after the contraction.</param>
        /// <param name="dtype">The loop dtype, or null for the operands' promoted dtype (NEP 50 <c>result_type</c>, not
        ///     <see cref="Sum(NDArray, int?, DType, bool)"/>'s widened accumulator: <c>'ij-&gt;i'</c> over int32 is
        ///     int32).</param>
        /// <param name="order">The result's memory layout: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>, in either
        ///     case. <c>'A'</c> and <c>'K'</c> give F order when every operand is F-contiguous; <c>'A'</c> gives C
        ///     order otherwise and <c>'K'</c> keeps the computed layout.</param>
        /// <param name="casting">The rule every operand must meet to reach the loop dtype, and the result to reach
        ///     <paramref name="out"/>'s dtype: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c> or
        ///     <c>"unsafe"</c>, spelled exactly.</param>
        /// <param name="optimize">NumPy's <c>optimize=</c>: null or <c>false</c>, <c>true</c>, any string
        ///     (<c>"greedy"</c>, <c>"optimal"</c>; NumPy reads another string as greedy), a path from
        ///     <c>np.einsum_path</c> (an <see cref="NumSharp.EinsumPath"/> or NumPy's raw <c>['einsum_path', ...]</c> list), or
        ///     a <c>(string, number)</c> tuple. It is checked like NumPy checks it and does not change the result: the
        ///     contraction always runs left to right.</param>
        /// <returns>
        ///     <para>
        ///     One operand, no <paramref name="out"/> and no label summed: a VIEW of the operand (diagonals and a
        ///     transpose), writeable when the operand is, never the operand itself; <paramref name="order"/> and
        ///     <paramref name="dtype"/> are ignored there, as NumPy ignores them.
        ///     </para>
        ///     <para>
        ///     Otherwise a new array in the loop dtype, or <paramref name="out"/> itself holding the result cast to its
        ///     dtype. A full contraction gives a 0-d array.
        ///     </para>
        /// </returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="subscripts"/> is null.</exception>
        /// <exception cref="ValueError">No operand (<c>must specify the einstein sum subscripts string and at least one
        ///     operand</c>); an <paramref name="order"/> letter or a <paramref name="casting"/> name NumPy refuses,
        ///     with its texts; subscripts NumPy refuses (malformed labels, a wrong operand count, a bad ellipsis, an
        ///     output label absent from the inputs, an impossible diagonal, label extents that disagree, an
        ///     <paramref name="out"/> of the wrong rank), with its parser's texts; an <paramref name="out"/> whose
        ///     extents differ from the result's; a read-only <paramref name="out"/>.</exception>
        /// <exception cref="TypeError"><paramref name="casting"/> is null; an <paramref name="optimize"/> NumPy does
        ///     not understand (<c>Did not understand the path: ...</c>); an operand that cannot reach the loop dtype
        ///     under <paramref name="casting"/>, or a result that cannot reach <paramref name="out"/>'s dtype, with
        ///     NumPy's iterator texts; a text operand or a text <paramref name="dtype"/> outside the view path, which
        ///     have no sum of products (NumPy's cast, zero-fill and loop refusals, in its order).</exception>
        /// <remarks>
        ///     <para> The contraction is NumPy's <c>optimize=</c> path (<c>bmm_einsum</c>): a left-to-right pairwise
        ///     fold, each step one <see cref="Matmul(NDArray, NDArray)"/> (batched when labels are shared and kept)
        ///     after diagonals, sums and a transpose of each term, or one broadcast <see cref="Multiply"/> when nothing
        ///     is contracted. The products dispatch on <see cref="Blas"/>, so float32 / float64 / complex128
        ///     contractions match NumPy's <c>optimize=True</c> byte for byte when the OpenBLAS backend is installed.
        ///     Integer and boolean contractions are exact; a float sum outside a product (<c>'ij-&gt;i'</c>, three or
        ///     more operands) follows NumSharp's sum order and can differ from NumPy's default (C) einsum in the last
        ///     ULP.
        ///     </para>
        ///     <para>
        ///     An override replaces the whole call, the checks included; the facade only applies the numpy.matrix and
        ///     numpy.char.chararray boundaries and converts the sublist spelling before it calls this member.
        ///     </para>
        /// </remarks>
        public abstract NDArray Einsum(string subscripts, NDArray[] operands, NDArray @out = null,
            DType dtype = null, char order = 'K', string casting = "safe", object optimize = null);

        /// <summary>
        ///     NumPy's <c>np.einsum_path(subscripts, *operands, optimize='greedy')</c>: the order in which to contract the
        ///     operands pairwise and NumPy's printed cost report, computed from the subscripts and the operands' SHAPES
        ///     (no element is read): <c>_parse_einsum_input</c>, then the <c>_greedy_path</c> / <c>_optimal_path</c>
        ///     search with the <c>_flop_count</c> cost model, then the info string.
        /// </summary>
        /// <param name="subscripts">The einsum subscripts, as <see cref="Einsum"/> reads them (the facade turns NumPy's
        ///     sublist spelling into this string first).</param>
        /// <param name="operands">The operands the subscripts label; null reads as none. Only their shapes are read.</param>
        /// <param name="optimize">NumPy's <c>optimize=</c>, required here (NumPy's default, <c>'greedy'</c>, is the facades':
        ///     a C# object parameter cannot default to a string), in its decision order: <c>false</c> / null (no
        ///     optimization: the operands in order), <c>true</c>, <c>"greedy"</c> or <c>"optimal"</c>, a precomputed
        ///     <see cref="NumSharp.EinsumPath"/> (an explicit path, checked for completeness), or a
        ///     <c>("greedy" | "optimal", memory)</c> tuple capping the largest intermediate.</param>
        /// <returns>The path (NumPy's <c>['einsum_path', (i, j), ...]</c> list as an <see cref="NumSharp.EinsumPath"/>) and the
        ///     report; both byte-identical to NumPy's except the placeholder letters an ellipsis expands to in the report,
        ///     which NumPy draws from a hash-randomized set and NumSharp deterministically.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="subscripts"/> is null.</exception>
        /// <exception cref="ValueError">A null operand (<c>einsum_path operand {i} is null</c>); subscripts NumPy's Python
        ///     parser refuses, with its texts (an operand count that differs from the terms', a second <c>-&gt;</c>, an
        ///     invalid symbol, an output label no input has, a term of the wrong rank, extents that disagree).</exception>
        /// <exception cref="KeyError">An unknown path name (NumPy's <c>('Path name %s not found', ...)</c>).</exception>
        /// <exception cref="RuntimeError">An explicit path that does not contract every operand.</exception>
        /// <exception cref="TypeError">An <paramref name="optimize"/> NumPy does not understand (<c>Did not understand the
        ///     path: ...</c>, <c>object of type 'int' has no len()</c>).</exception>
        /// <exception cref="System.IndexOutOfRangeException">An ellipsis with no operand to read its rank from (where NumPy fails
        ///     indexing its operand list).</exception>
        /// <remarks>
        ///     The cost model is <see cref="System.Numerics.BigInteger"/> arithmetic, as NumPy's Python ints are, so products
        ///     past 64 bits still compare correctly. An override replaces the plan of <c>np.einsum_path</c> (whose facade
        ///     dispatches on the first operand's engine, the default one with none); <see cref="Einsum"/> resolves its own
        ///     <c>optimize=</c> separately and contracts left to right whatever the plan.
        /// </remarks>
        public abstract (EinsumPath path, string repr) EinsumPath(string subscripts, NDArray[] operands, object optimize);
    }
}
