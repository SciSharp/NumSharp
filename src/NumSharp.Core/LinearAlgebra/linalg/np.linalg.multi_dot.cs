using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Chained matrix product, evaluated in the cheapest association order.
            /// </summary>
            /// <param name="arrays">
            ///     Two or more arrays. With three or more, every one must be 2-D EXCEPT the endpoints,
            ///     which may be 1-D (a 1-D first operand is a row, a 1-D last a column); a 0-D, a
            ///     3-D-or-higher, or a 1-D operand anywhere in the MIDDLE raises
            ///     <see cref="LinAlgError"/> "{ndim}-dimensional array given. Array must be
            ///     two-dimensional". With exactly two arrays this is a plain <see cref="np.dot"/> and
            ///     imposes no such restriction.
            /// </param>
            /// <param name="out">
            ///     Where to deposit the answer. NumPy hands <c>out</c> to the final <c>dot</c>, so it
            ///     receives the TWO-DIMENSIONAL product and the returned array is a reshaped view of
            ///     it — <c>out</c> for <c>multi_dot([v, B, C])</c> is shaped <c>(1, k)</c>, not
            ///     <c>(k,)</c>.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html
            ///     <para>
            ///     Matrix multiplication is associative but its COST is not: chaining
            ///     <c>(10000,100) @ (100,1000) @ (1000,5)</c> left to right costs about 10⁹
            ///     multiplications and right to left about 10⁷. This picks the order with the
            ///     classic O(n³) dynamic program (Cormen et al. 15.2) — except for exactly three
            ///     matrices, where the single comparison is done directly.
            ///     </para>
            ///     <para>
            ///     A 1-D first operand is treated as a ROW vector and a 1-D last operand as a COLUMN
            ///     vector, with the added axis removed again afterwards — so the chain's endpoints
            ///     behave like <c>np.dot</c>'s.
            ///     </para>
            /// </remarks>
            public static NDArray multi_dot(NDArray[] arrays, NDArray @out = null)
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

            /// <inheritdoc cref="multi_dot(NDArray[], NDArray)"/>
            public static NDArray multi_dot(params NDArray[] arrays) => multi_dot(arrays, null);

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

            private static NDArray Chain(NDArray[] arrays, int[,] split, int i, int j)
                => i == j
                    ? arrays[i]
                    : np.dot(Chain(arrays, split, i, split[i, j]), Chain(arrays, split, split[i, j] + 1, j));
        }
    }
}
