using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.multi_dot</c>: two arrays are a plain <c>np.dot</c>; three or more are promoted at
        ///     their vector ends, checked by <c>_assert_2d</c>, multiplied in the cheapest association order, and the
        ///     added axes dropped again.
        /// </summary>
        /// <param name="arrays">The chain (plain: the facade applied the class boundary).</param>
        /// <param name="out">Where to deposit the two-dimensional product, or null.</param>
        /// <returns>The product: <paramref name="out"/> itself for a two-array chain given one, else a fresh array or a
        ///     reshaped view of <paramref name="out"/>.</returns>
        /// <exception cref="ValueError">Fewer than two arrays (<c>Expecting at least two arrays.</c>), or a product the
        ///     out array cannot receive.</exception>
        /// <exception cref="LinAlgError">With three or more arrays, an operand other than a vector end that is not
        ///     two-dimensional.</exception>
        [NDScoped]
        internal static NDArray MultiDot(NDArray[] arrays, NDArray @out)
        {
            if (arrays is null || arrays.Length < 2)
                throw new ValueError("Expecting at least two arrays.");

            if (arrays.Length == 2)
                return Deliver(np.dot(arrays[0], arrays[1]), @out);

            var work = (NDArray[])arrays.Clone();

            // Only the ENDS may be vectors; NumPy promotes them so the cost model sees matrices
            // throughout, then drops the axis it added.
            bool prependedRow = work[0].ndim == 1;
            bool appendedColumn = work[work.Length - 1].ndim == 1;
            if (prependedRow)
                work[0] = np.expand_dims(work[0], 0);
            if (appendedColumn)
                work[work.Length - 1] = np.expand_dims(work[work.Length - 1], 1);

            // NumPy runs `_assert_2d(*arrays)` over the PROMOTED operands, so every array other
            // than a vector endpoint (already lifted to 2-D just above) must be exactly
            // two-dimensional: a 0-D, a 3-D-or-higher, or a 1-D operand in a MIDDLE position is
            // rejected here with the same LinAlgError. Without this guard the cost model reads a
            // phantom `dimensions[1]` off a non-matrix and the chain either leaks a raw bounds
            // error or — worse — computes a silently wrong-shaped product.
            Assert2d(work);

            NDArray product = work.Length == 3
                ? ThreeInBestOrder(work[0], work[1], work[2])
                : Chain(work, Order(work), 0, work.Length - 1);

            // NumPy threads `out` into the FINAL dot, so `out` receives the two-dimensional
            // product even when an endpoint was a vector — which is why
            // `multi_dot([v, B, C], out: o)` wants `o` shaped (1, k), not (k,) — and the array
            // returned to the caller is a reshaped VIEW of it.
            if (@out is not null)
            {
                np.copyto(@out, product);
                product = @out;
            }

            if (prependedRow && appendedColumn)
                return np.reshape(product, Array.Empty<long>());
            if (prependedRow)
                return np.reshape(product, new[] {product.Shape.dimensions[1]});
            if (appendedColumn)
                return np.reshape(product, new[] {product.Shape.dimensions[0]});

            return product;
        }

        /// <summary>A two-array chain's delivery: the product itself, or copied into <paramref name="out"/>.</summary>
        /// <param name="result">The product.</param>
        /// <param name="out">The out array, or null.</param>
        /// <returns><paramref name="result"/>, or <paramref name="out"/> after the copy.</returns>
        /// <exception cref="ValueError">The product cannot be copied into <paramref name="out"/> (a broadcast or
        ///     writeability refusal of <c>np.copyto</c>).</exception>
        /// <exception cref="TypeError">The product's dtype cannot be cast into <paramref name="out"/>'s same_kind.</exception>
        private static NDArray Deliver(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;
            np.copyto(@out, result);
            return @out;
        }

        /// <summary>
        ///     The two-way choice for a three-matrix chain: <c>(AB)C</c> costs
        ///     <c>a0·a1·b1 + a0·b1·c1</c> and <c>A(BC)</c> costs <c>a1·b1·c1 + a0·a1·c1</c>.
        /// </summary>
        /// <param name="a">The first matrix.</param>
        /// <param name="b">The second matrix.</param>
        /// <param name="c">The third matrix.</param>
        /// <returns>The product: <c>(AB)C</c> when it is strictly cheaper, else <c>A(BC)</c> (NumPy's
        ///     <c>cost1 &lt; cost2</c>, so a tie takes <c>A(BC)</c>).</returns>
        [NDScopedCovered] // only caller: [NDScoped] multi_dot (the inner dot of the chosen parenthesisation is scope-reclaimed)
        private static NDArray ThreeInBestOrder(NDArray a, NDArray b, NDArray c)
        {
            long a0 = a.Shape.dimensions[0];
            long a1b0 = a.Shape.dimensions[1];
            long b1c0 = b.Shape.dimensions[1];
            long c1 = c.Shape.dimensions[1];

            long left = a0 * b1c0 * (a1b0 + c1);
            long right = a1b0 * c1 * (a0 + b1c0);

            return left < right
                ? np.dot(np.dot(a, b), c)
                : np.dot(a, np.dot(b, c));
        }

        /// <summary>
        ///     Matrix-chain-order dynamic program. <c>s[i,j]</c> is the split point of the
        ///     cheapest parenthesisation of <c>A_i … A_j</c>.
        /// </summary>
        /// <param name="arrays">The chain of two-dimensional matrices.</param>
        /// <returns>The split table <c>s</c>.</returns>
        private static int[,] Order(NDArray[] arrays)
        {
            int n = arrays.Length;

            // p[0..n] holds the chain's dimensions: A_i is p[i] x p[i+1].
            var p = new long[n + 1];
            for (int i = 0; i < n; i++)
                p[i] = arrays[i].Shape.dimensions[0];
            p[n] = arrays[n - 1].Shape.dimensions[1];

            var cost = new long[n, n];
            var split = new int[n, n];

            for (int length = 1; length < n; length++)
            {
                for (int i = 0; i < n - length; i++)
                {
                    int j = i + length;
                    cost[i, j] = long.MaxValue;
                    for (int k = i; k < j; k++)
                    {
                        long q = cost[i, k] + cost[k + 1, j] + p[i] * p[k + 1] * p[j + 1];
                        if (q >= cost[i, j])
                            continue;
                        cost[i, j] = q;
                        split[i, j] = k;
                    }
                }
            }

            return split;
        }

        /// <summary>The product of <c>A_i … A_j</c> in the order <paramref name="split"/> chose.</summary>
        /// <param name="arrays">The chain of two-dimensional matrices.</param>
        /// <param name="split">The split table of <see cref="Order"/>.</param>
        /// <param name="i">The first matrix of the sub-chain.</param>
        /// <param name="j">The last matrix of the sub-chain.</param>
        /// <returns>The sub-chain's product (the matrix itself when <paramref name="i"/> equals <paramref name="j"/>).</returns>
        [NDScopedCovered] // only caller: [NDScoped] multi_dot (every recursive partial product is scope-reclaimed)
        private static NDArray Chain(NDArray[] arrays, int[,] split, int i, int j)
            => i == j
                ? arrays[i]
                : np.dot(Chain(arrays, split, i, split[i, j]), Chain(arrays, split, split[i, j] + 1, j));
    }
}
