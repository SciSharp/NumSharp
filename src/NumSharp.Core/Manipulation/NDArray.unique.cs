using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    /// Comparer for double that matches NumPy's sorting behavior:
    /// NaN is treated as greater than all other values (placed at end).
    /// </summary>
    internal sealed class NaNAwareDoubleComparer : IComparer<double>
    {
        public static readonly NaNAwareDoubleComparer Instance = new NaNAwareDoubleComparer();

        public int Compare(double x, double y)
        {
            // If both are NaN, they are equal
            if (double.IsNaN(x) && double.IsNaN(y)) return 0;
            // NaN is greater than any non-NaN value
            if (double.IsNaN(x)) return 1;
            if (double.IsNaN(y)) return -1;
            // For non-NaN values, use default comparison (handles -Inf, +Inf correctly)
            return x.CompareTo(y);
        }
    }

    /// <summary>
    /// Comparer for float that matches NumPy's sorting behavior:
    /// NaN is treated as greater than all other values (placed at end).
    /// </summary>
    internal sealed class NaNAwareSingleComparer : IComparer<float>
    {
        public static readonly NaNAwareSingleComparer Instance = new NaNAwareSingleComparer();

        public int Compare(float x, float y)
        {
            // If both are NaN, they are equal
            if (float.IsNaN(x) && float.IsNaN(y)) return 0;
            // NaN is greater than any non-NaN value
            if (float.IsNaN(x)) return 1;
            if (float.IsNaN(y)) return -1;
            // For non-NaN values, use default comparison (handles -Inf, +Inf correctly)
            return x.CompareTo(y);
        }
    }

    /// <summary>
    /// Comparer for Complex that matches NumPy's sorting behavior:
    /// Lexicographic compare (real, then imaginary). NaN in either component is treated
    /// as greater than all non-NaN values (placed at end).
    /// </summary>
    internal sealed class NaNAwareComplexComparer : IComparer<Complex>
    {
        public static readonly NaNAwareComplexComparer Instance = new NaNAwareComplexComparer();

        public int Compare(Complex x, Complex y)
        {
            bool xrNan = double.IsNaN(x.Real);
            bool yrNan = double.IsNaN(y.Real);
            bool xiNan = double.IsNaN(x.Imaginary);
            bool yiNan = double.IsNaN(y.Imaginary);
            bool xAnyNan = xrNan || xiNan;
            bool yAnyNan = yrNan || yiNan;
            // Any-NaN Complex values sort to end; among them, order is stable (return 0)
            if (xAnyNan && yAnyNan) return 0;
            if (xAnyNan) return 1;
            if (yAnyNan) return -1;
            // Neither has NaN — lex compare (real, imag)
            int c = x.Real.CompareTo(y.Real);
            if (c != 0) return c;
            return x.Imaginary.CompareTo(y.Imaginary);
        }
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Find the unique elements of an array (bare-return form).<br></br>
        ///
        ///     Returns the sorted unique elements. Mirrors NumPy's single-array return (no
        ///     <c>return_*</c> flag): supports <c>axis</c>-aware uniqueness and the <c>equal_nan</c>/
        ///     <c>sorted</c> keywords. For first-occurrence indices / reconstruction indices / counts,
        ///     use <see cref="unique(bool, bool, bool, int?, bool, bool)"/>.
        /// </summary>
        /// <param name="axis">Axis to operate on. If <c>null</c> (default), the array is flattened.</param>
        /// <param name="equal_nan">If <c>true</c> (default), all NaNs collapse to one output value.</param>
        /// <param name="sorted">Accepted for API parity; NumSharp always returns sorted output.</param>
        /// <returns>The sorted unique values.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> array of two or more dimensions with an axis (NumPy's structured
        ///     path refuses it).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array whose NA object is neither NaN-like nor a string, with an NA
        ///     among two or more distinct values (NumPy's sort refuses to compare it).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        public NDArray unique(int? axis = null, bool equal_nan = true, bool sorted = true)
        {
            // NumPy's axis checks for text (StringDType's structured-path refusal included).
            ThrowIfTextUnique(this, axis);
            // Delegate to the tuple path with all return flags false, then take the values. The
            // kwargs path uses the optimized sort+mask algorithm (NaN-partition for floats, plain
            // Array.Sort for integers, IL-vectorizable mask scan) and carries the axis handling —
            // substantially faster than the legacy Hashset+LongIntroSort path above ~1K elements.
            return unique(return_index: false, return_inverse: false, return_counts: false,
                          axis: axis, equal_nan: equal_nan, sorted: sorted)[0];
        }

        /// <summary>
        ///     Find the unique elements of an array with full NumPy keyword argument support.
        ///
        ///     Returns sorted unique elements; optionally returns first-occurrence indices,
        ///     reconstruction indices, and counts. Supports axis-aware uniqueness.
        /// </summary>
        /// <param name="return_index">Also return indices of <c>ar</c> (along axis, if specified)
        ///   that give the unique values.</param>
        /// <param name="return_inverse">Also return indices of the unique array
        ///   that can be used to reconstruct <c>ar</c>.</param>
        /// <param name="return_counts">Also return the number of times each unique value comes up.</param>
        /// <param name="axis">Axis to operate on. If <c>null</c> (default), the array is flattened.</param>
        /// <param name="equal_nan">If <c>true</c> (default), all NaN values are treated as equal
        ///   so only one appears in the output. If <c>false</c>, each NaN is treated as unique.</param>
        /// <param name="sorted">If <c>true</c> (default), the unique elements are sorted (NumPy 2.3).
        ///   NumSharp always returns sorted output — NumPy's <c>sorted=False</c> hash order for
        ///   integer/complex values and for text is platform-specific and not reproducible in C# — so this
        ///   parameter is accepted for API parity but does not change the result (spec-compliant).</param>
        /// <returns>An array of NDArrays in order: [values, index?, inverse?, counts?]. With <paramref name="axis"/>
        ///     null the inverse has the input's shape (NumPy's <c>inv_idx.reshape(ar.shape)</c>: <c>()</c> for a 0-d input,
        ///     <c>(0, 3)</c> for an empty <c>(0, 3)</c> one).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> array of two or more dimensions with an axis (NumPy's structured
        ///     path refuses it).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array whose NA object is neither NaN-like nor a string, where
        ///     NumPy's sort compares an NA (<see cref="UniqueVString"/>). Text computes NumPy's answer: fixed-width text through
        ///     <see cref="UniqueText"/>, <c>StringDType</c> through <see cref="UniqueVString"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public NDArray[] unique(
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
        {
            // sorted does not change the values: NumSharp cannot reproduce NumPy's non-portable sorted=False hash
            // order (integer/complex/text, values-only), so it always returns the deterministic sorted
            // result — identical AS A SET, and consistent with the unique_values family. Over StringDType it decides one
            // thing NumPy's does: without the sort, a None-like NA is never compared (and never refused).

            // Text: NumPy's axis check first. StringDType is tested first (IsString refuses it: its fixed-width readers would
            // misread the 16-byte cells) and computes NumPy's flat path (NDArray.unique.VString.cs); fixed-width text
            // computes NumPy's statements (NDArray.unique.Text.cs).
            ThrowIfTextUnique(this, axis);
            if (Backends.Strings.NDStringUfuncs.IsVString(this))
            {
                var vstringOutputs = UniqueVString(this, return_index, return_inverse, return_counts, equal_nan, sorted);
                // The flat path reshapes the inverse to the input's shape, as the numeric path.
                if (return_inverse)
                    ReshapeFlatInverseToInput(vstringOutputs, return_index ? 2 : 1);
                return vstringOutputs;
            }
            if (Backends.Strings.NDStringUfuncs.IsString(this))
            {
                var textOutputs = UniqueText(this, return_index, return_inverse, return_counts, axis);
                // The flat path (axis=None or a 1-D input) reshapes the inverse to the input's shape, as the numeric path.
                if (return_inverse && (axis is null || ndim == 1))
                    ReshapeFlatInverseToInput(textOutputs, return_index ? 2 : 1);
                return textOutputs;
            }

            // NumPy: `if axis is None or ar.ndim == 1` takes the flat path — a given axis normalized first — so a 1-D input
            // with axis 0 is the flat unique (its inverse the reshaped view below), not the slab path.
            if (axis is { } given)
            {
                int resolved = given < 0 ? given + ndim : given;
                if (resolved < 0 || resolved >= ndim)
                    throw new AxisError(given, ndim);
                if (ndim != 1)
                    return uniqueAxisKwargs(resolved, return_index, return_inverse, return_counts, equal_nan);
            }

            var outputs = uniqueFlatKwargs(return_index, return_inverse, return_counts, equal_nan);
            if (return_inverse)
                ReshapeFlatInverseToInput(outputs, return_index ? 2 : 1);
            return outputs;
        }

        /// <summary>
        ///     The checks of <c>np.unique</c> over text that come before its computation: an explicit axis must be in range
        ///     (NumPy's <c>np.moveaxis</c> re-raised as <c>AxisError(axis, ndim)</c>, or <c>normalize_axis_index</c> for a 1-D
        ///     input: the same text), and a <c>StringDType</c> array refuses the structured path of an axis over two or more
        ///     dimensions (<see cref="ThrowIfVStringUnique"/>). Does nothing for a numeric array.
        /// </summary>
        /// <param name="a">The array <c>unique</c> was called on.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <exception cref="AxisError">A string array with an out-of-range <paramref name="axis"/> (<c>axis 9 is out of
        ///     bounds for array of dimension 1</c>; a 0-d array refuses every axis).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> array of two or more dimensions with an axis (see
        ///     <see cref="ThrowIfVStringUnique"/>).</exception>
        internal static void ThrowIfTextUnique(NDArray a, int? axis)
        {
            // StringDType first: IsString refuses it loudly (its callers are fixed-width routes).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                ThrowIfVStringUnique(a, axis);
                return;
            }
            if (!Backends.Strings.NDStringUfuncs.IsString(a))
                return;
            if (axis is { } ax && (ax < -a.ndim || ax >= a.ndim))
                throw new AxisError(ax, a.ndim);
        }

        /// <summary>
        ///     The guard of <c>np.unique</c> (and the Array-API family) over NEP 55 <c>StringDType</c> text, in NumPy 2.4.2's
        ///     order: the flat path (<c>axis=None</c>, or a 1-D input whose axis passed <c>normalize_axis_index</c>) computes
        ///     (<see cref="UniqueVString"/>); an axis over two or more dimensions takes the structured path, which cannot hold
        ///     <c>StringDType</c> fields.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <exception cref="AxisError">The axis is out of bounds (<c>np.moveaxis</c>'s check re-raised as
        ///     <c>AxisError(axis, ndim)</c>, or <c>normalize_axis_index</c> for a 1-D input: the same text).</exception>
        /// <exception cref="TypeError">An axis over two or more dimensions: <c>The axis argument to unique is not supported
        ///     for dtype StringDType()</c> (the structured view of the rows refuses <c>StringDType</c> fields), or — for rows of
        ///     zero width, which skip that view — <c>Cannot change data-type for array of references.</c> (the field-less
        ///     result's view back as <c>StringDType</c>).</exception>
        private static void ThrowIfVStringUnique(NDArray a, int? axis)
        {
            if (axis is { } ax)
            {
                if (ax < -a.ndim || ax >= a.ndim)
                    throw new AxisError(ax, a.ndim);
                if (a.ndim != 1)
                {
                    // `ar = moveaxis(ar, axis, 0).reshape(n, prod(rest))`: the row width is every other axis's extent.
                    int norm = ax < 0 ? ax + a.ndim : ax;
                    long width = 1;
                    for (int d = 0; d < a.ndim; d++)
                        if (d != norm)
                            width *= a.Shape.dimensions[d];
                    if (width > 0)
                        throw new TypeError($"The axis argument to unique is not supported for dtype {a.dtype}");
                    throw new TypeError("Cannot change data-type for array of references.");
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>inv_idx.reshape(inverse_shape)</c> for the flat path (<c>axis=None</c>, or a 1-D input): the inverse
        ///     is always a VIEW with the input's shape (<c>flags.owndata</c> False, as NumPy's reshape returns) — <c>()</c> for a
        ///     0-d input (the flat path makes <c>(1,)</c>), <c>(0, 3)</c> for an empty <c>(0, 3)</c> one (it makes <c>(0,)</c>).
        /// </summary>
        /// <param name="outputs">The flat path's outputs; the inverse at <paramref name="at"/> is replaced by its reshaped view.</param>
        /// <param name="at">The inverse's position (1 without <c>return_index</c>, 2 with it).</param>
        private void ReshapeFlatInverseToInput(NDArray[] outputs, int at)
        {
            var inverse = outputs[at];
            outputs[at] = inverse.reshape(Shape.Adopt((long[])Shape.dimensions.Clone()));
            // The view holds its own reference to the inverse's buffer; the flat wrapper is released.
            inverse.Dispose();
        }

        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns></returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        protected NDArray unique<T>() where T : unmanaged, IComparable<T>
        {
            unsafe
            {
                var hashset = new Hashset<T>();
                if (Shape.IsContiguous)
                {
                    // The logical start is the base address PLUS the shape's offset: a contiguous view may keep the base
                    // address with a non-zero offset (np.ndarray over a buffer, a reconstructed view), and reading from
                    // the base alone took the elements before the view and dropped its last ones.
                    var src = (T*)this.Address + Shape.offset;
                    long len = this.size;
                    for (long i = 0; i < len; i++)
                        hashset.Add(src[i]);
                }
                else
                {
                    long len = this.size;
                    var flat = this.flat;
                    var src = (T*)flat.Address;
                    Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                    for (long i = 0; i < len; i++)
                        hashset.Add(src[getOffset(i)]);
                }

                // Allocate memory directly, copy, sort, then wrap in NDArray
                var count = hashset.LongCount;
                var memoryBlock = new UnmanagedMemoryBlock<T>(count);
                var arraySlice = new ArraySlice<T>(memoryBlock);
                Hashset<T>.CopyTo(hashset, arraySlice);

                // NumPy returns sorted unique values with NaN at end
                SortUnique<T>(memoryBlock.Address, count);

                // Create NDArray directly from ArraySlice (no additional allocation)
                return new NDArray(arraySlice, Shape.Vector(count));
            }
        }

        /// <summary>
        /// Sorts the unique values using LongIntroSort. For float/double, uses NaN-aware comparison
        /// that places NaN at the end (matching NumPy behavior).
        /// Supports long indexing for arrays exceeding int.MaxValue elements.
        /// </summary>
        private static unsafe void SortUnique<T>(T* ptr, long count) where T : unmanaged, IComparable<T>
        {
            if (typeof(T) == typeof(double))
            {
                Utilities.LongIntroSort.Sort((double*)ptr, count, NaNAwareDoubleComparer.Instance.Compare);
            }
            else if (typeof(T) == typeof(float))
            {
                Utilities.LongIntroSort.Sort((float*)ptr, count, NaNAwareSingleComparer.Instance.Compare);
            }
            else
            {
                Utilities.LongIntroSort.Sort(ptr, count);
            }
        }

    }
}
