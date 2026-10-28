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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html (the computation is this array's
        ///     <see cref="TensorEngine.Unique"/> without return flags, of which this member returns the values; the
        ///     internals are reclaimed at its return).</remarks>
        public NDArray unique(int? axis = null, bool equal_nan = true, bool sorted = true)
            // With every return flag false the engine returns [values]; the facade of NumPy's bare return takes it.
            => TensorEngine.Unique(this, axis: axis, equal_nan: equal_nan, sorted: sorted)[0];

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
        ///     NumPy's sort compares an NA. Text computes NumPy's answer: fixed-width text through NumPy's statements,
        ///     <c>StringDType</c> through NumPy's flat path (see <see cref="TensorEngine.Unique"/>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html (the computation is this array's
        ///     <see cref="TensorEngine.Unique"/>; the internals are reclaimed at its return, the outputs are kept).</remarks>
        public NDArray[] unique(
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
            => TensorEngine.Unique(this, return_index, return_inverse, return_counts, axis, equal_nan, sorted);

        /// <summary>
        ///     The sorted distinct values of this array, found with a hash set: a legacy helper that no NumSharp code calls,
        ///     kept for subclasses (it is <c>protected</c> on a public, unsealed class). <c>np.unique</c> does not run it and
        ///     its answer is not NumPy's in the cases listed under remarks; call <see cref="unique(int?, bool, bool)"/> for
        ///     NumPy's answer.
        /// </summary>
        /// <typeparam name="T">The CLR type of the array's dtype. The elements are read as <typeparamref name="T"/> without
        ///     a check, so any other type reinterprets the bytes.</typeparam>
        /// <returns>A new 1-D array of the distinct values in ascending order. Values that
        ///     <see cref="EqualityComparer{T}.Default"/> calls equal are one value: every NaN is one value, and so are -0.0
        ///     and +0.0; the one kept is the first met in C order, with its bits.</returns>
        /// <remarks>
        ///     Where the answer differs from NumPy's <c>np.unique</c>:
        ///     <list type="bullet">
        ///       <item><description>a float16 NaN sorts first, because <see cref="Half.CompareTo(Half)"/> orders a NaN below
        ///         every value (float32 and float64 put it last, as NumPy does);</description></item>
        ///       <item><description>the NaN kept has the bits of the first NaN met, where NumPy's sort writes the positive
        ///         quiet NaN for float32 and float64.</description></item>
        ///     </list>
        ///     A non-contiguous array is read through <see cref="flat"/>, a raveled copy that is not disposed: its buffer
        ///     returns to the pool only when the finalizer runs.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        /// </remarks>
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

                // Ascending order. float32 / float64 put a NaN last (NumPy's place for it); every other type sorts by its
                // CompareTo, which puts a float16 NaN first.
                SortUnique<T>(memoryBlock.Address, count);

                // Create NDArray directly from ArraySlice (no additional allocation)
                return new NDArray(arraySlice, Shape.Vector(count));
            }
        }

        /// <summary>
        /// Sorts the unique values using LongIntroSort. For float/double, uses NaN-aware comparison
        /// that places NaN at the end (matching NumPy behavior). Every other type sorts by its own
        /// <see cref="IComparable{T}.CompareTo"/>, which puts a float16 NaN first. The sort is not stable,
        /// which is harmless here only because the values are distinct.
        /// Supports long indexing for arrays exceeding int.MaxValue elements.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="ptr">The first of <paramref name="count"/> values, sorted in place.</param>
        /// <param name="count">The number of values.</param>
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
