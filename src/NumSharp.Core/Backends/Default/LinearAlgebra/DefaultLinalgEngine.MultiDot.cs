using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements: two arrays are <c>dot(a, b, out=out)</c>; three or more are promoted at their vector ends
        ///     (<c>atleast_2d</c>, and its transpose for the last), checked by <c>_assert_2d</c>, multiplied in the cheapest
        ///     association order (<c>_multi_dot_three</c> for three, the matrix-chain dynamic program above) with the out
        ///     array handed to the FINAL dot, and the result reshaped back: <c>result[0, 0]</c> when both ends were vectors,
        ///     <c>ravel()</c> when one was.
        ///     </para>
        ///     <para>
        ///     <b>out= is np.dot's.</b> NumPy passes <c>out</c> to the final <c>dot</c>, whose check is
        ///     <c>new_array_for_sum</c>: no cast, no broadcast. So <c>multi_dot([v, B, C], out=o)</c> wants <c>o</c> shaped
        ///     <c>(1, k)</c> and of the product's dtype, and a float64 <c>o</c> for an integer chain is refused where a copy
        ///     into it would have cast. The inner products are computed first, so their errors come before the out check.
        ///     </para>
        ///     <para>
        ///     <b>The vector ends.</b> NumPy's <c>result[0, 0]</c> of the (1, 1) product is a scalar: here a fresh 0-d copy
        ///     of that element, marked like NumPy's scalars (<see cref="NDArray.MarkReductionScalar"/>), never a view of the
        ///     product or of the out array. <c>result.ravel()</c> of the C-ordered (1, k) or (k, 1) product is a view of it.
        ///     </para>
        ///     <para>
        ///     <b>Members, not facades.</b> Every dot is np.dot's member (<see cref="CoreDot"/>), the promotions and the
        ///     ravel the array object's (<see cref="NDArray.atleast_2d"/>, <see cref="NDArray.ravel"/>). The
        ///     np.linalg.multi_dot facade has a matrix rule only, so a numpy.char.chararray vector end used to stay a
        ///     chararray through np.atleast_2d; it is a plain view now, which changes no result: every chain of three or
        ///     more ends in a dot, whose result is plain either way, and a text chain is refused by the dot.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public override NDArray MultiDot(NDArray[] arrays, NDArray @out = null)
        {
            if (arrays is null || arrays.Length < 2)
                throw new ValueError("Expecting at least two arrays.");

            int n = arrays.Length;
            if (n == 2)
                return CoreDot(arrays[0], arrays[1], @out);

            var work = (NDArray[])arrays.Clone();

            // NumPy saves the ends' ranks, then makes them matrices so the cost model sees matrices throughout:
            // `arrays[0] = atleast_2d(arrays[0])` is a (1, k) row and `arrays[-1] = atleast_2d(arrays[-1]).T` a (k, 1)
            // column (np.dot never reads an extent-1 axis's stride, so the views' strides there are immaterial).
            int ndimFirst = work[0].ndim, ndimLast = work[n - 1].ndim;
            if (ndimFirst == 1)
                work[0] = NDArray.atleast_2d(work[0]);
            if (ndimLast == 1)
                work[n - 1] = NDArray.atleast_2d(work[n - 1]).T;

            // NumPy runs `_assert_2d(*arrays)` over the PROMOTED operands, so every array other
            // than a vector endpoint (already lifted to 2-D just above) must be exactly
            // two-dimensional: a 0-D, a 3-D-or-higher, or a 1-D operand in a MIDDLE position is
            // rejected here with the same LinAlgError. Without this guard the cost model reads a
            // phantom `dimensions[1]` off a non-matrix and the chain either leaks a raw bounds
            // error or — worse — computes a silently wrong-shaped product.
            Assert2d(work);

            NDArray result = n == 3
                ? ThreeInBestOrder(work[0], work[1], work[2], @out)
                : Chain(work, Order(work), 0, n - 1, @out);

            if (ndimFirst == 1 && ndimLast == 1)
                return result[0, 0].copy().MarkReductionScalar();   // NumPy's result[0, 0]: a scalar, not a view
            if (ndimFirst == 1 || ndimLast == 1)
                return NDArray.ravel(result, 'C');                   // result.ravel(): np.ravel's member, order 'C'
            return result;
        }

        /// <summary>
        ///     NumPy's <c>_multi_dot_three</c>: the two-way choice for a three-matrix chain, <c>(AB)C</c> costing
        ///     <c>a0·b1c0·(a1b0 + c1)</c> against <c>A(BC)</c> costing <c>a1b0·c1·(a0 + b1c0)</c>, with the dimensions
        ///     read from <c>A.shape</c> and <c>C.shape</c> only (so a misaligned B reports the misalignment of the order
        ///     NumPy picks).
        /// </summary>
        /// <param name="a">The first matrix.</param>
        /// <param name="b">The second matrix.</param>
        /// <param name="c">The third matrix.</param>
        /// <param name="out">The final dot's out array, or null.</param>
        /// <returns>The product: <c>(AB)C</c> when it is strictly cheaper, else <c>A(BC)</c> (NumPy's
        ///     <c>cost1 &lt; cost2</c>, so a tie takes <c>A(BC)</c>); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">An out array np.dot refuses.</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands (np.dot's text).</exception>
        /// <remarks>NumPy's costs are Python integers, exact at any size; <see cref="Int128"/> keeps them exact for every
        ///     chain whose products could exist (three dimensions of at most 2^42 each).</remarks>
        [NDScopedCovered /* only caller: [NDScoped] multi_dot (the inner dot of the chosen parenthesisation is scope-reclaimed) */]
        private static NDArray ThreeInBestOrder(NDArray a, NDArray b, NDArray c, NDArray @out)
        {
            Int128 a0 = a.Shape.dimensions[0];
            Int128 a1b0 = a.Shape.dimensions[1];
            Int128 b1c0 = c.Shape.dimensions[0];
            Int128 c1 = c.Shape.dimensions[1];

            Int128 cost1 = a0 * b1c0 * (a1b0 + c1);
            Int128 cost2 = a1b0 * c1 * (a0 + b1c0);

            // np.dot's member at each step (CoreDot), the inner product computed first as the nested call did.
            return cost1 < cost2
                ? CoreDot(CoreDot(a, b, null), c, @out)
                : CoreDot(a, CoreDot(b, c, null), @out);
        }

        /// <summary>
        ///     NumPy's <c>_multi_dot_matrix_chain_order</c>, the matrix-chain-order dynamic program (Cormen et al. 15.2):
        ///     <c>s[i,j]</c> is the split point of the cheapest parenthesisation of <c>A_i … A_j</c>.
        /// </summary>
        /// <param name="arrays">The chain of two-dimensional matrices.</param>
        /// <returns>The split table <c>s</c>.</returns>
        /// <remarks>NumPy's cost table is float64 (<c>m = zeros((n, n), dtype=double)</c>, <c>m[i, j] = inf</c>) and each
        ///     candidate is <c>m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]</c> with the product an exact Python
        ///     integer added as a float, so the comparisons are float64 ones: past 2^53 two candidates can tie where
        ///     exact integers would not, and the first split of a tie wins (<c>q &lt; m[i, j]</c>). The same arithmetic
        ///     here keeps NumPy's order, and so its rounding, for every chain.</remarks>
        private static int[,] Order(NDArray[] arrays)
        {
            int n = arrays.Length;

            // p[0..n] holds the chain's dimensions: A_i is p[i] x p[i+1].
            var p = new long[n + 1];
            for (int i = 0; i < n; i++)
                p[i] = arrays[i].Shape.dimensions[0];
            p[n] = arrays[n - 1].Shape.dimensions[1];

            var cost = new double[n, n];
            var split = new int[n, n];

            for (int length = 1; length < n; length++)
            {
                for (int i = 0; i < n - length; i++)
                {
                    int j = i + length;
                    cost[i, j] = double.PositiveInfinity;
                    for (int k = i; k < j; k++)
                    {
                        // Python: (m[i, k] + m[k + 1, j]) + int, the int converted with one rounding (exact product first).
                        double q = cost[i, k] + cost[k + 1, j] + (double)((Int128)p[i] * p[k + 1] * p[j + 1]);
                        if (q < cost[i, j])
                        {
                            cost[i, j] = q;
                            split[i, j] = k;
                        }
                    }
                }
            }

            return split;
        }

        /// <summary>NumPy's <c>_multi_dot</c>: the product of <c>A_i … A_j</c> in the order <paramref name="split"/>
        ///     chose, <paramref name="out"/> handed to the dot of this level only.</summary>
        /// <param name="arrays">The chain of two-dimensional matrices.</param>
        /// <param name="split">The split table of <see cref="Order"/>.</param>
        /// <param name="i">The first matrix of the sub-chain.</param>
        /// <param name="j">The last matrix of the sub-chain.</param>
        /// <param name="out">The out array of this level's dot (the top level's only: the sub-products get none), or
        ///     null.</param>
        /// <returns>The sub-chain's product (the matrix itself when <paramref name="i"/> equals <paramref name="j"/>, which
        ///     the top level never is: a chain of three or more has a split).</returns>
        /// <exception cref="ValueError">An out array np.dot refuses.</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands (np.dot's text).</exception>
        [NDScopedCovered /* only caller: [NDScoped] multi_dot (every recursive partial product is scope-reclaimed) */]
        private static NDArray Chain(NDArray[] arrays, int[,] split, int i, int j, NDArray @out = null)
            => i == j
                ? arrays[i]
                // np.dot's member (CoreDot); the left sub-chain is computed before the right, as the call's arguments were.
                : CoreDot(Chain(arrays, split, i, split[i, j]), Chain(arrays, split, split[i, j] + 1, j), @out);
    }
}
