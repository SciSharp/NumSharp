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
        ///     Find the unique elements of an array.<br></br>
        ///     
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns>The sorted unique values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        public NDArray unique()
        {
            switch (typecode)
            {
#if _REGEN
	        %foreach supported_dtypes,supported_dtypes_lowercase%
	        case NPTypeCode.#1: return unique<#2>();
            %
            default: throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return unique<bool>();
                case NPTypeCode.Byte: return unique<byte>();
                case NPTypeCode.SByte: return unique<sbyte>();
                case NPTypeCode.Int16: return unique<short>();
                case NPTypeCode.UInt16: return unique<ushort>();
                case NPTypeCode.Int32: return unique<int>();
                case NPTypeCode.UInt32: return unique<uint>();
                case NPTypeCode.Int64: return unique<long>();
                case NPTypeCode.UInt64: return unique<ulong>();
                case NPTypeCode.Char: return unique<char>();
                case NPTypeCode.Half: return unique<Half>();
                case NPTypeCode.Double: return unique<double>();
                case NPTypeCode.Single: return unique<float>();
                case NPTypeCode.Decimal: return unique<decimal>();
                case NPTypeCode.Complex: return uniqueComplex();
                default: throw new NotSupportedException();
#endif
            }
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

        /// <summary>
        /// B9: Dedicated unique path for Complex, since System.Numerics.Complex does not implement
        /// IComparable&lt;Complex&gt; (prevents reuse of the generic unique&lt;T&gt;).
        /// Dedup uses EqualityComparer&lt;Complex&gt;.Default (component-wise value equality, NaN==NaN)
        /// then sorts using NumPy lex semantics with NaN at end.
        /// </summary>
        protected unsafe NDArray uniqueComplex()
        {
            var hashset = new Hashset<Complex>();
            if (Shape.IsContiguous)
            {
                var src = (Complex*)this.Address;
                long len = this.size;
                for (long i = 0; i < len; i++)
                    hashset.Add(src[i]);
            }
            else
            {
                long len = this.size;
                var flat = this.flat;
                var src = (Complex*)flat.Address;
                Func<long, long> getOffset = flat.Shape.GetOffset_1D;
                for (long i = 0; i < len; i++)
                    hashset.Add(src[getOffset(i)]);
            }

            var count = hashset.LongCount;
            var memoryBlock = new UnmanagedMemoryBlock<Complex>(count);
            var arraySlice = new ArraySlice<Complex>(memoryBlock);
            Hashset<Complex>.CopyTo(hashset, arraySlice);

            Utilities.LongIntroSort.Sort(memoryBlock.Address, count, NaNAwareComplexComparer.Instance.Compare);

            return new NDArray(arraySlice, Shape.Vector(count));
        }
    }
}
