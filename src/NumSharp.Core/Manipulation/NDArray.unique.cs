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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        public NDArray unique(int? axis = null, bool equal_nan = true, bool sorted = true)
        {
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
        /// <returns>An array of NDArrays in order: [values, index?, inverse?, counts?].</returns>
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

            if (axis == null)
            {
                return uniqueFlatKwargs(return_index, return_inverse, return_counts, equal_nan);
            }

            int resolved = axis.Value;
            if (resolved < 0) resolved += ndim;
            if (resolved < 0 || resolved >= ndim)
                throw new AxisError(axis.Value, ndim);

            return uniqueAxisKwargs(resolved, return_index, return_inverse, return_counts, equal_nan);
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
                    var src = (T*)this.Address;
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
