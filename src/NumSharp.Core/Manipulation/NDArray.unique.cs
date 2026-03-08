using System;
using System.Collections.Generic;
using System.Linq;
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
                case NPTypeCode.Int16: return unique<short>();
                case NPTypeCode.UInt16: return unique<ushort>();
                case NPTypeCode.Int32: return unique<int>();
                case NPTypeCode.UInt32: return unique<uint>();
                case NPTypeCode.Int64: return unique<long>();
                case NPTypeCode.UInt64: return unique<ulong>();
                case NPTypeCode.Char: return unique<char>();
                case NPTypeCode.Double: return unique<double>();
                case NPTypeCode.Single: return unique<float>();
                case NPTypeCode.Decimal: return unique<decimal>();
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
                    int len = this.size;
                    for (int i = 0; i < len; i++)
                        hashset.Add(src[i]);

                    var dst = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)dst.Array);
                    // NumPy returns sorted unique values with NaN at end
                    SortUniqueSpan<T>((T*)dst.Address, hashset.Count);
                    return dst;
                }
                else
                {
                    int len = this.size;
                    var flat = this.flat;
                    var src = (T*)flat.Address;
                    Func<int, int> getOffset = flat.Shape.GetOffset_1D;
                    for (int i = 0; i < len; i++)
                        hashset.Add(src[getOffset(i)]);

                    var dst = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)dst.Array);
                    // NumPy returns sorted unique values with NaN at end
                    SortUniqueSpan<T>((T*)dst.Address, hashset.Count);
                    return dst;
                }
            }
        }

        /// <summary>
        /// Sorts the unique values span. For float/double, uses NaN-aware comparison
        /// that places NaN at the end (matching NumPy behavior).
        /// </summary>
        private static unsafe void SortUniqueSpan<T>(T* ptr, int count) where T : unmanaged, IComparable<T>
        {
            var span = new Span<T>(ptr, count);

            if (typeof(T) == typeof(double))
            {
                var doubleSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(span);
                doubleSpan.Sort(NaNAwareDoubleComparer.Instance.Compare);
            }
            else if (typeof(T) == typeof(float))
            {
                var floatSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(span);
                floatSpan.Sort(NaNAwareSingleComparer.Instance.Compare);
            }
            else
            {
                span.Sort();
            }
        }
    }
}
