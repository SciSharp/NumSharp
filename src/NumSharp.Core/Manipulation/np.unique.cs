using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="unique(NDArray, bool, bool, bool, int?, bool, bool)"/> — NumPy's
        ///     bare-array-OR-tuple return expressed as one type. It carries the sorted unique
        ///     <see cref="values"/> plus whichever of <see cref="indices"/>/<see cref="inverse_indices"/>/
        ///     <see cref="counts"/> were requested (the others are <c>null</c>), so
        ///     <c>np.unique(ar, return_counts=True)</c> ports from Python verbatim.
        /// <para>
        ///     It stands in for both NumPy return shapes: it converts <b>implicitly to
        ///     <see cref="NDArray"/></b> (the bare form — yields <see cref="values"/>, so
        ///     <c>NDArray u = np.unique(ar);</c> works) and <b>implicitly to <see cref="NDArray"/><c>[]</c></b>
        ///     (the tuple form — the present outputs in NumPy field order). It also indexes
        ///     (<c>[k]</c> = the k-th present output, matching the old <c>NDArray[]</c> return) and
        ///     <c>Deconstruct</c>s (<c>var (values, counts) = np.unique(ar, return_counts: true);</c>).
        /// </para>
        /// </summary>
        /// <remarks>
        ///     BREAKING vs the pre-2.4.2-parity API: <c>np.unique(ar)</c> now returns this struct rather
        ///     than a bare <see cref="NDArray"/>. Assignment and argument passing are unaffected (implicit
        ///     conversion), but chaining an <see cref="NDArray"/> member directly on the result
        ///     (<c>np.unique(ar).array_equal(...)</c>) needs <c>.values</c>, and <c>np.unique(ar)[k]</c>
        ///     now selects the k-th OUTPUT (values, index, …), not the k-th unique value — use
        ///     <c>np.unique(ar).values[k]</c> for the latter. The instance
        ///     <see cref="NDArray.unique(int?, bool, bool)"/> / <see cref="NDArray.unique(bool, bool, bool, int?, bool, bool)"/>
        ///     still return <see cref="NDArray"/>/<see cref="NDArray"/><c>[]</c> unchanged.
        /// </remarks>
        public readonly struct UniqueResult
        {
            private readonly NDArray[] _outputs;   // present outputs, NumPy field order: [values, index?, inverse?, counts?]

            /// <summary>The sorted unique values (dtype preserved). Always present.</summary>
            public NDArray values { get; }

            /// <summary>First-occurrence indices, or <c>null</c> when <c>return_index</c> was False.</summary>
            public NDArray indices { get; }

            /// <summary>Reconstruction indices (shaped like the input), or <c>null</c> when <c>return_inverse</c> was False.</summary>
            public NDArray inverse_indices { get; }

            /// <summary>Per-value occurrence counts, or <c>null</c> when <c>return_counts</c> was False.</summary>
            public NDArray counts { get; }

            internal UniqueResult(NDArray[] outputs, bool hasIndex, bool hasInverse, bool hasCounts)
            {
                _outputs = outputs;
                values = outputs[0];
                int p = 1;
                indices = hasIndex ? outputs[p++] : null;
                inverse_indices = hasInverse ? outputs[p++] : null;
                counts = hasCounts ? outputs[p++] : null;
            }

            // ---- Bare-return NDArray surface (delegates to values) ----
            // In the no-flags case np.unique(ar) IS the values array, so the common NDArray members
            // forward to values. This keeps np.unique(ar).size / .dtype / .GetDouble(i) / .array_equal(...)
            // compiling now that the return type is UniqueResult — they mean values, the bare result.
            // For the full NDArray surface or unambiguous access, use .values explicitly.

            /// <summary>Element count of <see cref="values"/>.</summary>
            public long size => values.size;

            /// <summary>Shape of <see cref="values"/> as a <c>long[]</c>.</summary>
            public long[] shape => values.shape;

            /// <summary>The <see cref="NumSharp.Shape"/> of <see cref="values"/>.</summary>
            public Shape Shape => values.Shape;

            /// <summary>Rank of <see cref="values"/>.</summary>
            public int ndim => values.ndim;

            /// <summary>Element <see cref="Type"/> of <see cref="values"/>.</summary>
            public Type dtype => values.dtype;

            /// <summary>Element <see cref="NPTypeCode"/> of <see cref="values"/>.</summary>
            public NPTypeCode typecode => values.typecode;

            /// <summary>True if <see cref="values"/> equals <paramref name="rhs"/> element-wise (same shape).</summary>
            public bool array_equal(NDArray rhs) => values.array_equal(rhs);

            /// <summary>The element of <see cref="values"/> at a flat index.</summary>
            public object GetAtIndex(long index) => values.GetAtIndex(index);

            /// <summary>The <typeparamref name="T"/> element of <see cref="values"/> at a flat index.</summary>
            public T GetAtIndex<T>(long index) where T : unmanaged => values.GetAtIndex<T>(index);

            /// <summary>The <see cref="double"/> value of <see cref="values"/> at the given coordinates.</summary>
            public double GetDouble(params long[] indices) => values.GetDouble(indices);

            /// <summary>The <see cref="double"/> value of <see cref="values"/> at the given coordinates.</summary>
            public double GetDouble(int[] indices) => values.GetDouble(indices);

            /// <summary>The <see cref="float"/> value of <see cref="values"/> at the given coordinates.</summary>
            public float GetSingle(params long[] indices) => values.GetSingle(indices);

            /// <summary>The <see cref="float"/> value of <see cref="values"/> at the given coordinates.</summary>
            public float GetSingle(int[] indices) => values.GetSingle(indices);

            /// <summary>The <see cref="bool"/> value of <see cref="values"/> at the given coordinates.</summary>
            public bool GetBoolean(params long[] indices) => values.GetBoolean(indices);

            /// <summary>The <see cref="bool"/> value of <see cref="values"/> at the given coordinates.</summary>
            public bool GetBoolean(int[] indices) => values.GetBoolean(indices);

            /// <summary>The <see cref="int"/> value of <see cref="values"/> at the given coordinates.</summary>
            public int GetInt32(params long[] indices) => values.GetInt32(indices);

            /// <summary>Number of present outputs (1 + one per requested return flag).</summary>
            public int Length => _outputs.Length;

            /// <summary>The k-th present output in NumPy field order (values, then index/inverse/counts as requested).</summary>
            public NDArray this[int index] => _outputs[index];

            /// <summary>The present outputs as an <see cref="NDArray"/><c>[]</c> in NumPy field order.</summary>
            public NDArray[] ToArray() => _outputs;

            /// <summary>Bare-return conversion: yields <see cref="values"/> (NumPy's single-array return).</summary>
            public static implicit operator NDArray(UniqueResult result) => result.values;

            /// <summary>Tuple-return conversion: the present outputs, NumPy field order.</summary>
            public static implicit operator NDArray[](UniqueResult result) => result._outputs;

            /// <summary>Deconstructs into the first two present outputs (e.g. <c>(values, counts)</c>).</summary>
            public void Deconstruct(out NDArray values, out NDArray second)
            {
                values = _outputs[0];
                second = _outputs[1];
            }

            /// <summary>Deconstructs into the first three present outputs.</summary>
            public void Deconstruct(out NDArray values, out NDArray second, out NDArray third)
            {
                values = _outputs[0];
                second = _outputs[1];
                third = _outputs[2];
            }

            /// <summary>Deconstructs into all four outputs (requires every return flag set).</summary>
            public void Deconstruct(out NDArray values, out NDArray index, out NDArray inverse, out NDArray counts)
            {
                values = _outputs[0];
                index = _outputs[1];
                inverse = _outputs[2];
                counts = _outputs[3];
            }
        }

        /// <summary>
        ///     Find the unique elements of an array — the single NumPy-shaped entry point.<br></br>
        ///
        ///     Returns the sorted unique elements and, per the <c>return_*</c> flags, the
        ///     first-occurrence indices, the reconstruction (inverse) indices, and the per-value counts,
        ///     bundled in a <see cref="UniqueResult"/> that stands in for NumPy's bare-array-or-tuple
        ///     return. Because <see cref="UniqueResult"/> converts implicitly to both <see cref="NDArray"/>
        ///     and <see cref="NDArray"/><c>[]</c>, every NumPy call shape ports verbatim — including
        ///     <c>np.unique(ar, return_counts: true)</c> and <c>np.unique(ar, axis: 0)</c>.
        /// </summary>
        /// <param name="ar">Input array. Unless <paramref name="axis"/> is given, it is flattened first.</param>
        /// <param name="return_index">If True, also return the first-occurrence indices of <paramref name="ar"/>
        ///   (along <paramref name="axis"/> if given) that produce the unique values.</param>
        /// <param name="return_inverse">If True, also return the indices of the unique array that
        ///   reconstruct <paramref name="ar"/>.</param>
        /// <param name="return_counts">If True, also return the number of times each unique value appears.</param>
        /// <param name="axis">The axis to operate on. If <c>null</c> (default), the array is flattened first.</param>
        /// <param name="equal_nan">If True (default), all NaN values collapse to a single output value;
        ///   if False, each NaN is a distinct value.</param>
        /// <param name="sorted">Accepted for NumPy 2.3 parity; NumSharp always returns sorted output
        ///   (NumPy's <c>sorted=False</c> hash-iteration order for integer/complex values is
        ///   platform-specific and not reproducible in C# — spec-compliant, the Array API leaves it
        ///   unspecified). See <see cref="unique_values"/>.</param>
        /// <returns>A <see cref="UniqueResult"/> carrying <c>values</c> and the requested outputs.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public static UniqueResult unique(NDArray ar,
            bool return_index = false,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
        {
            NDArray[] outputs = ar.unique(return_index, return_inverse, return_counts, axis, equal_nan, sorted);
            return new UniqueResult(outputs, return_index, return_inverse, return_counts);
        }
    }
}
