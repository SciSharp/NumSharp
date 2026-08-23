using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================================
        //  Array API unique_* family (NumPy 2.x — numpy/lib/_arraysetops_impl.py)
        //
        //  All four are thin, Array-API-compatible wrappers over np.unique with equal_nan=False
        //  (so each NaN is a distinct value) and axis=None (the input is flattened). They differ
        //  only in which of the four outputs they surface and in the container they return:
        //
        //      unique_values(x)   ≙ unique(x, equal_nan=False, sorted=False)            -> values
        //      unique_counts(x)   ≙ unique(x, return_counts=True,  equal_nan=False)     -> (values, counts)
        //      unique_inverse(x)  ≙ unique(x, return_inverse=True, equal_nan=False)     -> (values, inverse_indices)
        //      unique_all(x)      ≙ unique(x, return_index=True, return_inverse=True,
        //                                   return_counts=True, equal_nan=False)
        //                                        -> (values, indices, inverse_indices, counts)
        //
        //  NumPy returns namedtuples; C# stands in with the named-field result structs below
        //  (implicit NDArray[] conversion + Deconstruct + indexer, the np.meshgrid house shape).
        //
        //  Two behaviours pinned to NumPy 2.4.2:
        //   * values dtype is preserved; indices/inverse_indices/counts are int64 (NumPy intp).
        //   * inverse_indices is reshaped to the ORIGINAL input shape (NumPy 2.0 change), so the
        //     input can be reconstructed with np.take(values, inverse_indices). A 0-d input's
        //     inverse is a 0-d () scalar — see ReshapeInverseToInput.
        //
        //  ONE deliberate divergence, unique_values only ([Misaligned]): for INTEGER and COMPLEX
        //  dtypes NumPy's sorted=False path dedups through a std::unordered_set (unique.cpp) and
        //  returns values in its iteration order — platform/compiler-specific hash order that is
        //  neither sorted nor first-occurrence and is not reproducible in C#. NumSharp returns the
        //  SORTED values (identical to np.unique(x, equal_nan=False), and identical to NumPy's result
        //  AS A SET) — deterministic, portable, and consistent with the values field of the other
        //  three functions. The Array API leaves this order unspecified. Plain FLOAT dtypes are NOT
        //  in NumPy's hash map, so NumPy sorts them too — NumSharp's float unique_values is bit-exact.
        // =====================================================================================

        /// <summary>
        ///     The result of <see cref="unique_all"/>: NumPy's <c>UniqueAllResult</c> namedtuple
        ///     (<c>values</c>, <c>indices</c>, <c>inverse_indices</c>, <c>counts</c>). Converts
        ///     implicitly to <see cref="NDArray"/><c>[]</c>, <c>Deconstruct</c>s
        ///     (<c>var (values, indices, inv, counts) = np.unique_all(x);</c>) and indexes (<c>[k]</c>).
        /// </summary>
        public readonly struct UniqueAllResult
        {
            /// <summary>The sorted unique values of the input.</summary>
            public NDArray values { get; }

            /// <summary>The index of the first occurrence (in the flattened input) of each unique value.</summary>
            public NDArray indices { get; }

            /// <summary>The indices into <see cref="values"/> that reconstruct the input, shaped like the input.</summary>
            public NDArray inverse_indices { get; }

            /// <summary>The number of times each unique value appears in the input.</summary>
            public NDArray counts { get; }

            internal UniqueAllResult(NDArray values, NDArray indices, NDArray inverseIndices, NDArray counts)
            {
                this.values = values;
                this.indices = indices;
                this.inverse_indices = inverseIndices;
                this.counts = counts;
            }

            /// <summary>Number of outputs (always 4).</summary>
            public int Length => 4;

            /// <summary>The k-th output in NumPy field order: values, indices, inverse_indices, counts.</summary>
            public NDArray this[int index] => index switch
            {
                0 => values,
                1 => indices,
                2 => inverse_indices,
                3 => counts,
                _ => throw new IndexOutOfRangeException($"UniqueAllResult has 4 outputs; index {index} is out of range.")
            };

            /// <summary>Returns the outputs as an <see cref="NDArray"/><c>[]</c> in NumPy field order.</summary>
            public NDArray[] ToArray() => new[] { values, indices, inverse_indices, counts };

            /// <summary>Exposes the outputs — the tuple NumPy's <c>unique_all</c> returns.</summary>
            public static implicit operator NDArray[](UniqueAllResult result) => result.ToArray();

            /// <summary>Deconstructs into <c>(values, indices, inverse_indices, counts)</c>.</summary>
            public void Deconstruct(out NDArray values, out NDArray indices, out NDArray inverseIndices, out NDArray counts)
            {
                values = this.values;
                indices = this.indices;
                inverseIndices = this.inverse_indices;
                counts = this.counts;
            }
        }

        /// <summary>
        ///     The result of <see cref="unique_counts"/>: NumPy's <c>UniqueCountsResult</c> namedtuple
        ///     (<c>values</c>, <c>counts</c>). Converts implicitly to <see cref="NDArray"/><c>[]</c>,
        ///     <c>Deconstruct</c>s (<c>var (values, counts) = np.unique_counts(x);</c>) and indexes.
        /// </summary>
        public readonly struct UniqueCountsResult
        {
            /// <summary>The sorted unique values of the input.</summary>
            public NDArray values { get; }

            /// <summary>The number of times each unique value appears in the input.</summary>
            public NDArray counts { get; }

            internal UniqueCountsResult(NDArray values, NDArray counts)
            {
                this.values = values;
                this.counts = counts;
            }

            /// <summary>Number of outputs (always 2).</summary>
            public int Length => 2;

            /// <summary>The k-th output in NumPy field order: values, counts.</summary>
            public NDArray this[int index] => index switch
            {
                0 => values,
                1 => counts,
                _ => throw new IndexOutOfRangeException($"UniqueCountsResult has 2 outputs; index {index} is out of range.")
            };

            /// <summary>Returns the outputs as an <see cref="NDArray"/><c>[]</c> in NumPy field order.</summary>
            public NDArray[] ToArray() => new[] { values, counts };

            /// <summary>Exposes the outputs — the tuple NumPy's <c>unique_counts</c> returns.</summary>
            public static implicit operator NDArray[](UniqueCountsResult result) => result.ToArray();

            /// <summary>Deconstructs into <c>(values, counts)</c>.</summary>
            public void Deconstruct(out NDArray values, out NDArray counts)
            {
                values = this.values;
                counts = this.counts;
            }
        }

        /// <summary>
        ///     The result of <see cref="unique_inverse"/>: NumPy's <c>UniqueInverseResult</c> namedtuple
        ///     (<c>values</c>, <c>inverse_indices</c>). Converts implicitly to <see cref="NDArray"/><c>[]</c>,
        ///     <c>Deconstruct</c>s (<c>var (values, inv) = np.unique_inverse(x);</c>) and indexes.
        /// </summary>
        public readonly struct UniqueInverseResult
        {
            /// <summary>The sorted unique values of the input.</summary>
            public NDArray values { get; }

            /// <summary>The indices into <see cref="values"/> that reconstruct the input, shaped like the input.</summary>
            public NDArray inverse_indices { get; }

            internal UniqueInverseResult(NDArray values, NDArray inverseIndices)
            {
                this.values = values;
                this.inverse_indices = inverseIndices;
            }

            /// <summary>Number of outputs (always 2).</summary>
            public int Length => 2;

            /// <summary>The k-th output in NumPy field order: values, inverse_indices.</summary>
            public NDArray this[int index] => index switch
            {
                0 => values,
                1 => inverse_indices,
                _ => throw new IndexOutOfRangeException($"UniqueInverseResult has 2 outputs; index {index} is out of range.")
            };

            /// <summary>Returns the outputs as an <see cref="NDArray"/><c>[]</c> in NumPy field order.</summary>
            public NDArray[] ToArray() => new[] { values, inverse_indices };

            /// <summary>Exposes the outputs — the tuple NumPy's <c>unique_inverse</c> returns.</summary>
            public static implicit operator NDArray[](UniqueInverseResult result) => result.ToArray();

            /// <summary>Deconstructs into <c>(values, inverse_indices)</c>.</summary>
            public void Deconstruct(out NDArray values, out NDArray inverseIndices)
            {
                values = this.values;
                inverseIndices = this.inverse_indices;
            }
        }

        /// <summary>
        ///     Returns the unique elements of an input array <paramref name="x"/>.<br></br>
        ///
        ///     Array API compatible alternative to <c>np.unique(x, equal_nan=False, sorted=False)</c>.
        ///     The input is flattened. Each NaN is treated as a distinct value.
        /// </summary>
        /// <param name="x">Input array. Flattened if it is not already 1-D.</param>
        /// <returns>The unique elements of <paramref name="x"/> (dtype preserved).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unique_values.html<br></br>
        ///     NumSharp returns the values SORTED (like <c>np.unique(x, equal_nan=False)</c>). For integer and
        ///     complex dtypes NumPy 2.4.2's <c>sorted=False</c> path returns them in a platform-specific hash
        ///     order that is not portable (both contain the same set); float dtypes are sorted on both sides.
        ///     The Array API leaves the order unspecified.
        /// </remarks>
        // NDScoped: the hash/sort internals' temps are reclaimed; the values are yielded.
        [NDScoped]
        public static NDArray unique_values(NDArray x)
        {
            return x.uniqueValuesFast();
        }

        /// <summary>
        ///     Find the unique elements and counts of an input array <paramref name="x"/>.<br></br>
        ///
        ///     Array API compatible alternative to <c>np.unique(x, return_counts=True, equal_nan=False)</c>.
        ///     The input is flattened. Each NaN is treated as a distinct value.
        /// </summary>
        /// <param name="x">Input array. Flattened if it is not already 1-D.</param>
        /// <returns>A <see cref="UniqueCountsResult"/>: (values, counts).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique_counts.html</remarks>
        public static UniqueCountsResult unique_counts(NDArray x)
        {
            // Boundary scope: internals reclaimed; both result-struct members yielded.
            using var scope = NDScope.Open();
            var r = scope.Returns(x.uniqueCountsFast());
            return new UniqueCountsResult(r[0], r[1]);
        }

        /// <summary>
        ///     Find the unique elements of <paramref name="x"/> and the indices that reconstruct it.<br></br>
        ///
        ///     Array API compatible alternative to <c>np.unique(x, return_inverse=True, equal_nan=False)</c>.
        ///     The input is flattened. Each NaN is treated as a distinct value.
        /// </summary>
        /// <param name="x">Input array. Flattened if it is not already 1-D.</param>
        /// <returns>A <see cref="UniqueInverseResult"/>: (values, inverse_indices). <c>inverse_indices</c>
        ///     has the same shape as <paramref name="x"/>, so <c>np.take(values, inverse_indices)</c>
        ///     reconstructs the input.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique_inverse.html</remarks>
        public static UniqueInverseResult unique_inverse(NDArray x)
        {
            // Boundary scope: internals reclaimed; the values and the (possibly reshaped)
            // inverse are yielded. ReshapeInverseToInput may return r[1] itself or a fresh
            // view of it — Returns handles both (a second yield of the same array is a no-op).
            using var scope = NDScope.Open();
            var r = x.unique(return_index: false, return_inverse: true, return_counts: false,
                             axis: null, equal_nan: false);
            return new UniqueInverseResult(scope.Returns(r[0]), scope.Returns(ReshapeInverseToInput(r[1], x)));
        }

        /// <summary>
        ///     Find the unique elements of <paramref name="x"/> together with the first-occurrence indices,
        ///     reconstruction indices, and counts.<br></br>
        ///
        ///     Array API compatible alternative to <c>np.unique(x, return_index=True, return_inverse=True,
        ///     return_counts=True, equal_nan=False)</c>. The input is flattened. Each NaN is treated as
        ///     a distinct value.
        /// </summary>
        /// <param name="x">Input array. Flattened if it is not already 1-D.</param>
        /// <returns>A <see cref="UniqueAllResult"/>: (values, indices, inverse_indices, counts).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique_all.html</remarks>
        public static UniqueAllResult unique_all(NDArray x)
        {
            // Boundary scope: internals reclaimed; all four result-struct members yielded.
            using var scope = NDScope.Open();
            var r = x.unique(return_index: true, return_inverse: true, return_counts: true,
                             axis: null, equal_nan: false);
            return new UniqueAllResult(scope.Returns(r[0]), scope.Returns(r[1]),
                                       scope.Returns(ReshapeInverseToInput(r[2], x)), scope.Returns(r[3]));
        }

        /// <summary>
        ///     Reshapes the flat inverse-index array to the ORIGINAL input shape, matching NumPy's
        ///     <c>inv_idx.reshape(ar.shape)</c> so the input can be reconstructed with
        ///     <c>np.take(values, inverse_indices)</c>. The underlying kwargs path already yields the
        ///     input shape for ndim ≥ 1; a 0-d input needs its <c>()</c> scalar shape restored (the flat
        ///     path returns it as <c>(1,)</c>). Reshape is a zero-copy view over the freshly-allocated
        ///     inverse array.
        /// </summary>
        private static NDArray ReshapeInverseToInput(NDArray inverse, NDArray x)
            => inverse.reshape(x.shape);
    }
}
