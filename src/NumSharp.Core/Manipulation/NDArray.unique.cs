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
        /// <exception cref="StringComputationNotSupportedException">A fixed-width string array (NumPy sorts the text: the
        ///     next phase of NumSharp's string support).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        public NDArray unique(int? axis = null, bool equal_nan = true, bool sorted = true)
        {
            // NumPy's unique sorts text: the next phase of NumSharp's string support (after NumPy's axis check).
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
        ///   integer/complex values is platform-specific and not reproducible in C# — so this
        ///   parameter is accepted for API parity but does not change the result (spec-compliant).</param>
        /// <returns>An array of NDArrays in order: [values, index?, inverse?, counts?]. With <paramref name="axis"/>
        ///     null the inverse has the input's shape (NumPy's <c>inv_idx.reshape(ar.shape)</c>: <c>()</c> for a 0-d input,
        ///     <c>(0, 3)</c> for an empty <c>(0, 3)</c> one).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="StringComputationNotSupportedException">A fixed-width string array whose answer depends on the
        ///     text (NumPy sorts it: the next phase of NumSharp's string support). A string array with at most one element to
        ///     order (or slabs of zero width along <paramref name="axis"/>) is answered: the shape decides it.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html</remarks>
        public NDArray[] unique(
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
        {
            // sorted is a no-op: NumSharp cannot reproduce NumPy's non-portable sorted=False hash
            // order (integer/complex, values-only), so it always returns the deterministic sorted
            // result — identical AS A SET, and consistent with the unique_values family.
            _ = sorted;

            // Text: NumPy's axis check, then it sorts the text (the next phase of the string support) — except where the
            // shape decides the answer, which the bool stand-in computes and the input's own strings fill.
            ThrowIfTextUnique(this, axis);
            if (Backends.Strings.NDStringUfuncs.IsString(this))
                return UniqueTextByShape(this, return_index, return_inverse, return_counts, axis, equal_nan);

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
        ///     The guard of <c>np.unique</c> over text, after the one check NumPy makes before it sorts: an explicit axis
        ///     must be in range (NumPy's <c>np.moveaxis</c> re-raised as <c>AxisError(axis, ndim)</c>). NumPy then sorts the
        ///     text, the next phase of NumSharp's string support — unless the SHAPE decides the answer
        ///     (<see cref="UniqueShapeDecides"/>), in which case this returns and the caller builds it with
        ///     <see cref="UniqueTextByShape"/>. Does nothing for a numeric array.
        /// </summary>
        /// <param name="a">The array <c>unique</c> was called on.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="op">The NumPy function named in the next-phase message (<c>unique</c>, <c>unique_values</c>, …).</param>
        /// <exception cref="AxisError">A string array with an out-of-range <paramref name="axis"/> (<c>axis 9 is out of
        ///     bounds for array of dimension 1</c>; a 0-d array refuses every axis).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string array whose answer depends on the text.</exception>
        internal static void ThrowIfTextUnique(NDArray a, int? axis, string op = "unique")
        {
            if (!Backends.Strings.NDStringUfuncs.IsString(a))
                return;
            if (axis is { } ax && (ax < -a.ndim || ax >= a.ndim))
                throw new AxisError(ax, a.ndim);
            if (!UniqueShapeDecides(a, axis))
                throw Backends.Strings.NDStringUfuncs.NextPhase(op, a);
        }

        /// <summary>
        ///     Whether <c>np.unique</c>'s answer over <paramref name="a"/> is decided by the shape: at most one element (or
        ///     one slab along <paramref name="axis"/>) to order, or slabs of zero width — NumPy views each slab as a structured
        ///     scalar, and slabs with no fields all compare equal (<c>np.unique(np.zeros((3, 0), 'U2'), axis=0)</c> is one
        ///     <c>(1, 0)</c> slab with count 3).
        /// </summary>
        /// <param name="a">The array (its axis already validated).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>True when no element is compared.</returns>
        private static bool UniqueShapeDecides(NDArray a, int? axis)
        {
            if (axis is not { } ax)
                return a.size <= 1;
            int norm = ax < 0 ? ax + a.ndim : ax;
            // An empty array has either no slab or slabs of zero width; one slab compares with nothing.
            return a.size == 0 || a.Shape.dimensions[norm] <= 1;
        }

        /// <summary>
        ///     NumPy's <c>np.unique</c> over text whose answer the shape decides (<see cref="ThrowIfTextUnique"/> passed): the
        ///     index outputs (first occurrences, inverse, counts) do not depend on the values over such a shape, so they come
        ///     from the same call on a bool stand-in of the input's shape; the values are the input's own strings — none for
        ///     an empty result, the single element (flattened) for <c>axis=None</c>, the single slab (the input itself) for an
        ///     axis. With an axis, the values are built as NumPy's <c>reshape_uniq</c> builds them: a C-contiguous buffer of
        ///     the slabs, the axis first, viewed back with <c>np.moveaxis</c> (so the result is a view, like NumPy's).
        /// </summary>
        /// <param name="a">The string array.</param>
        /// <param name="return_index">Also return the first-occurrence indices.</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="equal_nan">NumPy's NaN switch (no NaN in text; passed through).</param>
        /// <returns>The outputs in NumPy's order: [values, index?, inverse?, counts?]; values in <paramref name="a"/>'s dtype.</returns>
        private static NDArray[] UniqueTextByShape(NDArray a, bool return_index, bool return_inverse, bool return_counts,
            int? axis, bool equal_nan)
        {
            using var standIn = Backends.Strings.NDStringUfuncs.BoolStandIn(a);
            var outputs = standIn.unique(return_index, return_inverse, return_counts, axis, equal_nan);
            // The stand-in's bool values are replaced (and released); every other output is int64 and kept as it is.
            using (var boolValues = outputs[0])
            {
                if (axis is not { } ax || a.ndim == 1)
                {
                    // aux[mask] in NumPy (the flat path also serves a 1-D input with an axis): a fresh 1-D array.
                    outputs[0] = boolValues.size == 0 ? new NDArray(a.dtype, Shape.Vector(0), true) : a.flatten();
                }
                else
                {
                    int norm = ax < 0 ? ax + a.ndim : ax;
                    // The slab buffer has the result's dimensions with the axis moved first: (n, *rest).
                    var dims = boolValues.Shape.dimensions;
                    var movedDims = new long[dims.Length];
                    movedDims[0] = dims[norm];
                    for (int d = 0, k = 1; d < dims.Length; d++)
                        if (d != norm)
                            movedDims[k++] = dims[d];
                    // One slab is the input's own, copied in C order; an empty result holds no element.
                    NDArray slabs;
                    if (boolValues.size == 0)
                        slabs = new NDArray(a.dtype, Shape.Adopt(movedDims), true);
                    else
                    {
                        using var moved = np.moveaxis(a, norm, 0);
                        slabs = moved.copy();
                    }
                    outputs[0] = np.moveaxis(slabs, 0, norm);
                    // The moveaxis view holds its own counted reference to the slab buffer.
                    slabs.Dispose();
                }
            }
            return outputs;
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
