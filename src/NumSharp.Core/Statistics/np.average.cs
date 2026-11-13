using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="average(NDArray, NDAxis?, NDArray, bool, bool)"/> — NumPy's bare-average-OR-tuple
        ///     return expressed as one type (np.unique's <see cref="UniqueResult"/> pattern): the average
        ///     (<see cref="values"/>), plus <see cref="sum_of_weights"/> when <c>returned</c> was set (null otherwise).
        /// <para>
        ///     It stands in for both NumPy return shapes: it converts <b>implicitly to <see cref="NDArray"/></b> (the bare
        ///     form: <c>NDArray avg = np.average(a);</c>) and to <see cref="NDArray"/><c>[]</c> (the present outputs), and
        ///     deconstructs: <c>var (avg, sum_of_weights) = np.average(a, returned: true);</c>.
        /// </para>
        /// </summary>
        /// <remarks>
        ///     <para>API-parity batch 11 (the owner's decision 3, result structs for NumPy's flag-dependent returns): it
        ///     replaces np.average's NDArray return and the NumSharp-only <c>np.average_returned</c>. The NDArray members
        ///     callers read on an average forward to <see cref="values"/> (size, shape, dtype, flags, the typed getters,
        ///     <c>ToArray&lt;T&gt;</c>, <c>reshape</c>, <c>array_equal</c>, <c>astype</c>), and the explicit scalar casts
        ///     of every dtype an average can have (<c>(double)np.average(a)</c>, NumPy's <c>float(np.average(a))</c>) read
        ///     a 0-d average; for anything else read <see cref="values"/> or assign the result to an
        ///     <see cref="NDArray"/>.</para>
        ///     <para>It is <see cref="IDisposable"/> as the NDArray it replaced was: <c>using var avg = np.average(a);</c>
        ///     releases the average and the sum of weights together.</para>
        ///     <para>Footgun: an extension method of <see cref="NDArray"/> (FluentAssertions' <c>Should()</c>) does not see
        ///     through the implicit conversion - call it on <see cref="values"/>. C# applies one user-defined conversion
        ///     at a time, so a cast the result does not declare (<c>(int)np.average(a)</c>) needs <see cref="values"/>:
        ///     <c>(int)np.average(a).values</c>.</para>
        /// </remarks>
        public readonly struct AverageResult : INDArrayCarrier, IDisposable
        {
            /// <summary>The average: float64 for an integer or bool operand, else the common dtype of the operand and the
            ///     weights; 0-d over every axis; a matrix operand's class as its rule gives it. Always present.</summary>
            public NDArray values { get; }

            /// <summary>The sum of the weights in the average's dtype and shape (the element count per average without
            ///     weights), or <c>null</c> when <c>returned</c> was false.</summary>
            public NDArray sum_of_weights { get; }

            /// <summary>Wraps the outputs the average computed.</summary>
            /// <param name="values">The average.</param>
            /// <param name="sum_of_weights">The sum of weights, or null without <c>returned</c>.</param>
            internal AverageResult(NDArray values, NDArray sum_of_weights)
            {
                this.values = values;
                this.sum_of_weights = sum_of_weights;
            }

            // ---- Bare-return NDArray surface (delegates to values) ----

            /// <summary>Element count of <see cref="values"/>.</summary>
            public long size => values.size;

            /// <summary>Shape of <see cref="values"/> as a <c>long[]</c>.</summary>
            public long[] shape => values.shape;

            /// <summary>The <see cref="NumSharp.Shape"/> of <see cref="values"/>.</summary>
            public Shape Shape => values.Shape;

            /// <summary>Rank of <see cref="values"/> (0 over every axis).</summary>
            public int ndim => values.ndim;

            /// <summary>The descriptor of <see cref="values"/>.</summary>
            public DType dtype => values.dtype;

            /// <summary>The storage lane of <see cref="values"/>.</summary>
            public NPTypeCode typecode => values.typecode;

            /// <summary>The flags of <see cref="values"/> (<see cref="NDArray.flags"/>): an average is a fresh array, so
            ///     <c>owndata</c> holds even for the average over no axis.</summary>
            public NDArrayFlags flags => values.flags;

            /// <summary>True if <see cref="values"/> equals <paramref name="rhs"/> element-wise (same shape).</summary>
            /// <param name="rhs">The array to compare with.</param>
            /// <returns>Whether the shapes and elements are equal.</returns>
            public bool array_equal(NDArray rhs) => values.array_equal(rhs);

            /// <summary>The elements of <see cref="values"/> as a new <typeparamref name="T"/>[] in C order.</summary>
            /// <typeparam name="T">The element type, as <see cref="NDArray.ToArray{T}"/> reads it.</typeparam>
            /// <returns>A new managed array.</returns>
            public T[] ToArray<T>() where T : unmanaged => values.ToArray<T>();

            /// <summary>The element of <see cref="values"/> at a flat index, boxed in its own type.</summary>
            /// <param name="index">The flat index.</param>
            /// <returns>The element.</returns>
            public object GetAtIndex(long index) => values.GetAtIndex(index);

            /// <summary>The element of <see cref="values"/> at a flat index, read as <typeparamref name="T"/> (the stored
            ///     value for the average's own type, NumPy's cast of it otherwise).</summary>
            /// <typeparam name="T">The type to read the element as.</typeparam>
            /// <param name="index">The flat index (C order); a negative one counts from the end.</param>
            /// <returns>The element as <typeparamref name="T"/>.</returns>
            /// <exception cref="IndexError"><paramref name="index"/> is outside the average.</exception>
            public T GetAtIndex<T>(long index) where T : unmanaged => values.GetAtIndex<T>(index);

            /// <summary>The <see cref="double"/> value of <see cref="values"/> at the given coordinates (none for a 0-d
            ///     average).</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as a double.</returns>
            public double GetDouble(params long[] indices) => values.GetDouble(indices);

            /// <summary>The <see cref="float"/> value of <see cref="values"/> at the given coordinates.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element read as a float.</returns>
            public float GetSingle(params long[] indices) => values.GetSingle(indices);

            /// <summary><see cref="values"/> with a new shape (<see cref="NDArray.reshape(long[])"/>: a view where one
            ///     exists).</summary>
            /// <param name="shape">The new dimensions.</param>
            /// <returns>The reshaped average.</returns>
            /// <exception cref="IncorrectShapeException">The size does not match.</exception>
            public NDArray reshape(params long[] shape) => values.reshape(shape);

            /// <summary><see cref="values"/> cast to <paramref name="dtype"/>
            ///     (<see cref="NDArray.astype(DType, char, string, bool)"/>).</summary>
            /// <param name="dtype">The target dtype.</param>
            /// <param name="order">The memory order of a copy ('K' keeps the average's).</param>
            /// <param name="casting">The casting rule ('unsafe' by default, as NumPy's astype).</param>
            /// <param name="copy">Whether to copy when no cast is needed.</param>
            /// <returns>The cast average.</returns>
            /// <exception cref="InvalidCastException">The cast is refused under <paramref name="casting"/>.</exception>
            public NDArray astype(DType dtype, char order = 'K', string casting = "unsafe", bool copy = true)
                => values.astype(dtype, order, casting, copy);

            // ---- Tuple-return surface ----

            /// <summary>Number of present outputs: 1, or 2 with <c>returned</c> (NumPy's tuple length).</summary>
            public int Length => sum_of_weights is null ? 1 : 2;

            /// <summary>The <paramref name="index"/>-th present output: 0 the average, 1 the sum of weights.</summary>
            /// <param name="index">0 or 1.</param>
            /// <returns>The output.</returns>
            /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is not below <see cref="Length"/>.</exception>
            public NDArray this[int index]
                => index == 0 ? values
                 : index == 1 && sum_of_weights is not null ? sum_of_weights
                 : throw new IndexOutOfRangeException("tuple index out of range");

            /// <summary>The present outputs as an <see cref="NDArray"/><c>[]</c>, NumPy's order.</summary>
            /// <returns>A new array of one or two outputs.</returns>
            public NDArray[] ToArray() => sum_of_weights is null ? new[] { values } : new[] { values, sum_of_weights };

            /// <summary>Bare-return conversion: the average (NumPy's single-array return).</summary>
            /// <param name="result">The result.</param>
            /// <returns><see cref="values"/>.</returns>
            public static implicit operator NDArray(AverageResult result) => result.values;

            /// <summary>Tuple-return conversion: the present outputs, NumPy's order.</summary>
            /// <param name="result">The result.</param>
            /// <returns>A new array of the outputs.</returns>
            public static implicit operator NDArray[](AverageResult result) => result.ToArray();

            // ---- Scalar casts of a 0-d average ----
            // One per dtype an average can have (float64 for an integer or bool operand, else float16 / float32 / float64 /
            // complex128 / decimal): C# does not chain the implicit NDArray conversion into NDArray's own explicit casts,
            // so without these `(double)np.average(a)` - the C# spelling of NumPy's float(np.average(a)) - stops compiling.

            /// <summary>NumPy's <c>float(np.average(a))</c>: the 0-d average as a <see cref="double"/>.</summary>
            /// <param name="result">The result; only <see cref="values"/> is read.</param>
            /// <returns>The average.</returns>
            /// <exception cref="TypeError">The average is not 0-d (<c>only 0-dimensional arrays can be converted to Python
            ///     scalars</c>), or it is complex (CPython's <c>float()</c> text).</exception>
            public static explicit operator double(AverageResult result) => (double)result.values;

            /// <summary>The 0-d average as a <see cref="float"/> (NumPy's <c>float()</c> rounded to float32).</summary>
            /// <param name="result">The result; only <see cref="values"/> is read.</param>
            /// <returns>The average.</returns>
            /// <exception cref="TypeError">The average is not 0-d, or it is complex.</exception>
            public static explicit operator float(AverageResult result) => (float)result.values;

            /// <summary>The 0-d average as a <see cref="Half"/> (NumPy's <c>float()</c> rounded to float16).</summary>
            /// <param name="result">The result; only <see cref="values"/> is read.</param>
            /// <returns>The average.</returns>
            /// <exception cref="TypeError">The average is not 0-d, or it is complex.</exception>
            public static explicit operator Half(AverageResult result) => (Half)result.values;

            /// <summary>The 0-d average as a <see cref="decimal"/> (NumSharp's lane). Footgun: a NaN or infinite float
            ///     average converts to 0, NumSharp's decimal convention, not an error.</summary>
            /// <param name="result">The result; only <see cref="values"/> is read.</param>
            /// <returns>The average.</returns>
            /// <exception cref="TypeError">The average is not 0-d, or it is complex.</exception>
            public static explicit operator decimal(AverageResult result) => (decimal)result.values;

            /// <summary>NumPy's <c>complex(np.average(a))</c>: the 0-d average as a <see cref="System.Numerics.Complex"/>
            ///     (a real average gets a zero imaginary part).</summary>
            /// <param name="result">The result; only <see cref="values"/> is read.</param>
            /// <returns>The average.</returns>
            /// <exception cref="TypeError">The average is not 0-d.</exception>
            public static explicit operator System.Numerics.Complex(AverageResult result) => (System.Numerics.Complex)result.values;

            /// <summary>Deconstructs a <c>returned=True</c> result: <c>var (avg, sum_of_weights) = ...</c>.</summary>
            /// <param name="values">The average.</param>
            /// <param name="sum_of_weights">The sum of weights.</param>
            /// <exception cref="InvalidOperationException">The result holds the average alone (<c>returned</c> was false):
            ///     Python's <c>not enough values to unpack (expected 2, got 1)</c>.</exception>
            public void Deconstruct(out NDArray values, out NDArray sum_of_weights)
            {
                if (this.sum_of_weights is null)
                    throw new InvalidOperationException("not enough values to unpack (expected 2, got 1)");
                values = this.values;
                sum_of_weights = this.sum_of_weights;
            }

            /// <summary>Releases the average and the sum of weights (the NDArray it replaced was disposable the same way).
            ///     Footgun: an <see cref="NDArray"/> taken from the result (the implicit conversion, an indexer, a
            ///     deconstruction) is the same object, so it is released too.</summary>
            public void Dispose()
            {
                values?.Dispose();
                // Never the same object today; the guard keeps a future engine that returns one array for both from a
                // double release.
                if (!ReferenceEquals(sum_of_weights, values))
                    sum_of_weights?.Dispose();
            }

            // Yields both outputs (an absent sum of weights is null - a safe no-op): a [NDScoped] caller returning this
            // result keeps its arrays alive past its scope.
            void INDArrayCarrier.YieldTo(NDScope scope)
            {
                scope.Returns(values);
                scope.Returns(sum_of_weights);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.average(a, axis=None, weights=None, returned=False, *, keepdims=False)</c>: the weighted average
        ///     along the given axes, <c>sum(a * weights) / sum(weights)</c>, and with <paramref name="returned"/> the sum of
        ///     the weights. When <paramref name="weights"/> is null this is the mean over the same axes.
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) averages every axis; one axis (negative
        ///     counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) averages those axes, and the empty
        ///     tuple averages none (a float copy of every element).</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or of
        ///     the averaged axes' extents in the order of <paramref name="axis"/> (NumPy transposes an unsorted tuple's
        ///     weights to the axes' order; then <paramref name="axis"/> is required). A <c>numpy.matrix</c> of weights beside
        ///     a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum is the matrix
        ///     method).</param>
        /// <param name="returned">NumPy's <c>returned</c>: also return the sum of the weights, in the average's dtype and
        ///     shape (the element count per average without weights) - <c>var (avg, sum_of_weights) = np.average(a,
        ///     returned: true);</c>.</param>
        /// <param name="keepdims">Keep the averaged axes with length 1, so the result broadcasts against
        ///     <paramref name="a"/>.</param>
        /// <returns>An <see cref="AverageResult"/>: the average (float64 for an integer or bool operand, else the common dtype
        ///     of the operand and the weights; 0-d over every axis; a <c>numpy.matrix</c> operand gives the matrix method's
        ///     answer - the scalar without an axis, otherwise a matrix that keeps the averaged axes with length 1), and with
        ///     <paramref name="returned"/> the sum of weights (a plain array for a matrix operand too). It converts
        ///     implicitly to the average's <see cref="NDArray"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (<c>axis: axis 2 is out of bounds for array of dimension
        ///     2</c>; NumPy names the argument, and checks every axis before the repeat check).</exception>
        /// <exception cref="ZeroDivisionError">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); an axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of the averaged axes' extents (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c>, whose body
        ///     <see cref="TensorEngine.Average"/> runs).
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaced the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.average(a, null)</c> ambiguous (CS0121) and had no tuple literal. API-parity batch
        ///     11 (the owner's decision 3): NumPy's <c>returned</c> in NumPy's place (before <c>keepdims</c>, which NumPy
        ///     takes by keyword only) and the <see cref="AverageResult"/> return; the NumSharp-only
        ///     <c>np.average_returned</c> went.</para></remarks>
        public static AverageResult average(NDArray a, NDAxis? axis = null, NDArray weights = null, bool returned = false, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule, which
            // reads the axis as given: None or not). The axes go as an array, an int as NumPy's one-element tuple.
            return NDArray.average(a, axis?.ToArray(), weights, returned, keepdims);
        }

        /// <summary>
        ///     <see cref="average(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axis as one <c>int</c>: the overload a
        ///     C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so
        ///     every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="average(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static AverageResult average(NDArray a, int axis, NDArray weights = null, bool returned = false, bool keepdims = false)
            => average(a, (NDAxis?)axis, weights, returned, keepdims);

        /// <summary>
        ///     <see cref="average(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axes as an <c>int[]</c>: the overload an
        ///     <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis),
        ///     forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple (<c>new
        ///     int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="average(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static AverageResult average(NDArray a, int[] axis, NDArray weights = null, bool returned = false, bool keepdims = false)
            => average(a, (NDAxis?)axis, weights, returned, keepdims);
    }
}
