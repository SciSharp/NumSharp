using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="intersect1d(NDArray, NDArray, bool, bool)"/> — NumPy's bare-array-OR-tuple return
        ///     expressed as one type (np.unique's <see cref="UniqueResult"/> pattern): the sorted common
        ///     <see cref="values"/>, plus <see cref="comm1"/> / <see cref="comm2"/> when <c>return_indices</c> was set (null
        ///     otherwise).
        /// <para>
        ///     It stands in for both NumPy return shapes: it converts <b>implicitly to <see cref="NDArray"/></b> (the bare
        ///     form, the values: <c>NDArray v = np.intersect1d(a, b);</c>) and <b>implicitly to <see cref="NDArray"/><c>[]</c></b>
        ///     (the tuple form, the present outputs in NumPy's order), indexes the outputs (<c>[1]</c> is comm1), and
        ///     deconstructs: <c>var (values, comm1, comm2) = np.intersect1d(a, b, return_indices: true);</c>.
        /// </para>
        /// </summary>
        /// <remarks>
        ///     <para>API-parity batch 11 (2026-11-13): it replaces the two NDArray-returning overloads and the
        ///     <see cref="NDArray"/><c>[]</c>-returning one. Assignment and argument passing are unaffected (the implicit
        ///     conversions); the common NDArray members forward to <see cref="values"/> (size, shape, dtype, the typed
        ///     getters, <c>ToArray&lt;T&gt;</c>, <c>array_equal</c>), and <see cref="Dispose"/> releases every output, so
        ///     <c>using var r = np.intersect1d(...)</c> keeps working.</para>
        ///     <para>Footgun: an extension method of <see cref="NDArray"/> (FluentAssertions' <c>Should()</c>, a LINQ-style
        ///     helper) does not see through the implicit conversion - call it on <see cref="values"/>, or assign the
        ///     result to an <see cref="NDArray"/> first.</para>
        /// </remarks>
        public readonly struct Intersect1dResult : INDArrayCarrier, IDisposable
        {
            // The present outputs, NumPy's order: [values] or [values, comm1, comm2].
            private readonly NDArray[] _outputs;

            /// <summary>The sorted 1-D array of common (unique) values. Always present.</summary>
            public NDArray values { get; }

            /// <summary>The indices of the first occurrences of the common values in <c>ar1</c> (int64), or <c>null</c>
            ///     when <c>return_indices</c> was false.</summary>
            public NDArray comm1 { get; }

            /// <summary>The indices of the first occurrences of the common values in <c>ar2</c> (int64), or <c>null</c>
            ///     when <c>return_indices</c> was false.</summary>
            public NDArray comm2 { get; }

            /// <summary>Wraps the outputs <see cref="intersect1d(NDArray, NDArray, bool, bool)"/> computed.</summary>
            /// <param name="outputs">One array (the values) or three (values, comm1, comm2), NumPy's order.</param>
            internal Intersect1dResult(NDArray[] outputs)
            {
                _outputs = outputs;
                values = outputs[0];
                comm1 = outputs.Length > 1 ? outputs[1] : null;
                comm2 = outputs.Length > 2 ? outputs[2] : null;
            }

            // ---- Bare-return NDArray surface (delegates to values) ----
            // Without return_indices np.intersect1d(a, b) IS the values array, so the NDArray members callers read on it
            // forward to values; for the full NDArray surface use .values (or assign to an NDArray).

            /// <summary>Element count of <see cref="values"/>.</summary>
            public long size => values.size;

            /// <summary>Shape of <see cref="values"/> as a <c>long[]</c> (a copy, as <see cref="NDArray.shape"/> is).</summary>
            public long[] shape => values.shape;

            /// <summary>The <see cref="NumSharp.Shape"/> of <see cref="values"/>.</summary>
            public Shape Shape => values.Shape;

            /// <summary>Rank of <see cref="values"/> (always 1).</summary>
            public int ndim => values.ndim;

            /// <summary>The descriptor of <see cref="values"/> (the two operands' common dtype).</summary>
            public DType dtype => values.dtype;

            /// <summary>The storage lane of <see cref="values"/>.</summary>
            public NPTypeCode typecode => values.typecode;

            /// <summary>True if <see cref="values"/> equals <paramref name="rhs"/> element-wise (same shape).</summary>
            /// <param name="rhs">The array to compare with.</param>
            /// <returns>Whether the shapes and elements are equal.</returns>
            public bool array_equal(NDArray rhs) => values.array_equal(rhs);

            /// <summary>The elements of <see cref="values"/> as a new <typeparamref name="T"/>[] in C order.</summary>
            /// <typeparam name="T">The element type to read (it must be the dtype's own type, as
            ///     <see cref="NDArray.ToArray{T}"/> requires).</typeparam>
            /// <returns>A new managed array.</returns>
            /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the values' element type.</exception>
            public T[] ToArray<T>() where T : unmanaged => values.ToArray<T>();

            /// <summary>The element of <see cref="values"/> at a flat index, boxed in its own type.</summary>
            /// <param name="index">The flat index.</param>
            /// <returns>The element.</returns>
            public object GetAtIndex(long index) => values.GetAtIndex(index);

            /// <summary>The <see cref="double"/> value of <see cref="values"/> at the given coordinates.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as a double.</returns>
            public double GetDouble(params long[] indices) => values.GetDouble(indices);

            /// <summary>The <see cref="double"/> value of <see cref="values"/> at the given coordinates.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as a double.</returns>
            public double GetDouble(int[] indices) => values.GetDouble(indices);

            /// <summary>The <see cref="int"/> value of <see cref="values"/> at the given coordinates.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as an int.</returns>
            public int GetInt32(params long[] indices) => values.GetInt32(indices);

            /// <summary>The <see cref="long"/> value of <see cref="values"/> at the given coordinates.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as a long.</returns>
            public long GetInt64(params long[] indices) => values.GetInt64(indices);

            // ---- Tuple-return surface ----

            /// <summary>Number of present outputs: 1, or 3 with <c>return_indices</c> (NumPy's tuple length).</summary>
            public int Length => _outputs.Length;

            /// <summary>The <paramref name="index"/>-th present output in NumPy's order (values, comm1, comm2).</summary>
            /// <param name="index">0 for the values, 1 and 2 for the index arrays.</param>
            /// <returns>The output.</returns>
            /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is not below <see cref="Length"/>.</exception>
            public NDArray this[int index] => _outputs[index];

            /// <summary>The present outputs as an <see cref="NDArray"/><c>[]</c> in NumPy's order.</summary>
            /// <returns>The outputs array (the struct's own; do not mutate it).</returns>
            public NDArray[] ToArray() => _outputs;

            /// <summary>Bare-return conversion: the values (NumPy's single-array return).</summary>
            /// <param name="result">The result.</param>
            /// <returns><see cref="values"/>.</returns>
            public static implicit operator NDArray(Intersect1dResult result) => result.values;

            /// <summary>Tuple-return conversion: the present outputs, NumPy's order.</summary>
            /// <param name="result">The result.</param>
            /// <returns>The outputs array.</returns>
            public static implicit operator NDArray[](Intersect1dResult result) => result._outputs;

            /// <summary>Deconstructs a <c>return_indices=True</c> result: <c>var (values, comm1, comm2) = ...</c>.</summary>
            /// <param name="values">The common values.</param>
            /// <param name="comm1">The first-occurrence indices in <c>ar1</c>.</param>
            /// <param name="comm2">The first-occurrence indices in <c>ar2</c>.</param>
            /// <exception cref="InvalidOperationException">The result holds only the values (Python's <c>not enough values
            ///     to unpack (expected 3, got 1)</c>).</exception>
            public void Deconstruct(out NDArray values, out NDArray comm1, out NDArray comm2)
            {
                if (_outputs.Length != 3)
                    throw new InvalidOperationException("not enough values to unpack (expected 3, got 1)");
                values = this.values;
                comm1 = this.comm1;
                comm2 = this.comm2;
            }

            /// <summary>Releases every present output (the values and the index arrays): the <c>using</c> form of a
            ///     result the caller is done with.</summary>
            /// <remarks>Disposing the struct disposes the arrays it shares with every copy of it and with any variable
            ///     the implicit conversions filled - dispose once, when none of them is read again.</remarks>
            public void Dispose()
            {
                if (_outputs is null)
                    return;
                foreach (var output in _outputs)
                    output?.Dispose();
            }

            // Yields every present output (absent ones are null - a safe no-op): a [NDScoped] caller returning this
            // result keeps its arrays alive past its scope.
            void INDArrayCarrier.YieldTo(NDScope scope)
            {
                scope.Returns(values);
                scope.Returns(comm1);
                scope.Returns(comm2);
            }
        }

        /// <summary>
        ///     Find the intersection of two arrays — NumPy's <c>np.intersect1d(ar1, ar2, assume_unique=False,
        ///     return_indices=False)</c>, one signature: the sorted, unique values that are in both input arrays, and with
        ///     <paramref name="return_indices"/> the indices of their first occurrences in each.
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up the
        ///     calculation (and, when they are not, the result holds duplicates and wrong indices, as in NumPy).</param>
        /// <param name="return_indices">If True, also return the indices of the first occurrences of the common values in
        ///     <paramref name="ar1"/> and <paramref name="ar2"/> (NumPy sorts with the stable 'mergesort' here).</param>
        /// <returns>An <see cref="Intersect1dResult"/>: the values (it converts implicitly to <see cref="NDArray"/>), and
        ///     with <paramref name="return_indices"/> comm1 / comm2 (it deconstructs into the three).</returns>
        /// <exception cref="UnicodeDecodeError">The concatenation decodes <c>S</c> text beside <c>U</c> text, and a byte is
        ///     not ASCII (in the order of the unique values).</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> operand beside a class the concatenation finds no
        ///     common DType with (a number, bool, <c>S</c> bytes): NumPy's verbatim promotion text.</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> array whose sort compares a None-like NA (an NA object
        ///     that is neither a string nor NaN-like) with another entry: NumPy's <c>Cannot compare null that is not a
        ///     nan-like value</c>.</exception>
        /// <remarks>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html</para>
        ///     <para>API-parity batch 11: one signature returning the result type (np.unique's pattern) - it replaces the
        ///     <c>(ar1, ar2)</c> and <c>(ar1, ar2, assume_unique)</c> overloads that returned the values and the
        ///     <see cref="NDArray"/><c>[]</c> one that always returned an array of outputs.</para>
        /// </remarks>
        public static Intersect1dResult intersect1d(NDArray ar1, NDArray ar2, bool assume_unique = false, bool return_indices = false)
            => new Intersect1dResult(IntersectCore(ar1, ar2, assume_unique, return_indices));

        /// <summary>
        ///     NumPy's <c>intersect1d</c> statements: unique (or ravel) each operand, concatenate, sort (argsort with the
        ///     stable 'mergesort' when the indices are wanted), the adjacent-equal mask, and the first-occurrence indices
        ///     mapped back through each operand's unique index.
        /// </summary>
        /// <param name="ar1">The first operand.</param>
        /// <param name="ar2">The second operand.</param>
        /// <param name="assume_unique">Skip the unique pass (the operands are flattened instead).</param>
        /// <param name="return_indices">Also compute comm1 / comm2.</param>
        /// <returns><c>[values]</c>, or <c>[values, comm1, comm2]</c> with <paramref name="return_indices"/>.</returns>
        /// <exception cref="UnicodeDecodeError">See <see cref="intersect1d(NDArray, NDArray, bool, bool)"/>.</exception>
        /// <exception cref="DTypePromotionError">See <see cref="intersect1d(NDArray, NDArray, bool, bool)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="intersect1d(NDArray, NDArray, bool, bool)"/>.</exception>
        [NDScoped]
        private static NDArray[] IntersectCore(NDArray ar1, NDArray ar2, bool assume_unique, bool return_indices)
        {
            // Text runs the statements below unchanged: np.unique and the sorts order the text (the index form with
            // NumPy's stable 'mergesort', whose order of equal strings is not the default kind's), and the concatenation
            // formats numbers into text and decodes bytes beside str as NumPy's does.
            // Boundary scope: the unique/ravel/concatenate/argsort/take temps and the adjacency
            // mask are reclaimed at exit; the returned tuple's arrays are yielded.

            NDArray ind1 = null, ind2 = null;
            if (!assume_unique)
            {
                if (return_indices)
                {
                    NDArray[] u1 = np.unique(ar1, true); ar1 = u1[0]; ind1 = u1[1];
                    NDArray[] u2 = np.unique(ar2, true); ar2 = u2[0]; ind2 = u2[1];
                }
                else
                {
                    ar1 = np.unique(ar1);
                    ar2 = np.unique(ar2);
                }
            }
            else
            {
                ar1 = np.ravel(ar1);
                ar2 = np.ravel(ar2);
            }

            long ar1Size = ar1.size;
            NDArray cat = np.concatenate((ar1, ar2), 0);   // promotes to the common dtype

            NDArray aux, auxSortIndices = null;
            if (return_indices)
            {
                auxSortIndices = np.argsort(cat, -1, kind: "mergesort");   // NumPy: argsort(aux, kind='mergesort')
                aux = np.take(cat, auxSortIndices);
            }
            else
            {
                aux = np.sort(cat);
            }

            // mask = aux[1:] == aux[:-1]  — an adjacent equal pair means the value is in both arrays.
            NDArray auxLeft = aux[":-1"];
            NDArray mask = aux["1:"] == auxLeft;
            NDArray int1d = auxLeft[mask];

            if (!return_indices)
                return new[] { int1d };

            NDArray ar1_indices = auxSortIndices[":-1"][mask];
            NDArray ar2_indices = auxSortIndices["1:"][mask] - (NDArray)ar1Size;
            if (!assume_unique)
            {
                ar1_indices = np.take(ind1, ar1_indices);
                ar2_indices = np.take(ind2, ar2_indices);
            }

            return new[] { int1d, ar1_indices, ar2_indices };
        }
    }
}
