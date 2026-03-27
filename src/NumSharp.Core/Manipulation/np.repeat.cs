using System;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Repeat elements of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <returns>Output array which has the same shape as a, except along the given axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        public static NDArray repeat(NDArray a, int repeats) => repeat(a, (long)repeats);

        /// <summary>
        ///     Repeat elements of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <returns>Output array which has the same shape as a, except along the given axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        public static NDArray repeat(NDArray a, long repeats)
        {
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");

            // Handle empty input or zero repeats
            if (a.size == 0 || repeats == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            long totalSize = a.size * repeats;
            a = a.ravel(); // After ravel(), array is guaranteed contiguous

            return a.GetTypeCode switch
            {
                NPTypeCode.Boolean => RepeatScalarTyped<bool>(a, repeats, totalSize),
                NPTypeCode.Byte => RepeatScalarTyped<byte>(a, repeats, totalSize),
                NPTypeCode.Int16 => RepeatScalarTyped<short>(a, repeats, totalSize),
                NPTypeCode.UInt16 => RepeatScalarTyped<ushort>(a, repeats, totalSize),
                NPTypeCode.Int32 => RepeatScalarTyped<int>(a, repeats, totalSize),
                NPTypeCode.UInt32 => RepeatScalarTyped<uint>(a, repeats, totalSize),
                NPTypeCode.Int64 => RepeatScalarTyped<long>(a, repeats, totalSize),
                NPTypeCode.UInt64 => RepeatScalarTyped<ulong>(a, repeats, totalSize),
                NPTypeCode.Char => RepeatScalarTyped<char>(a, repeats, totalSize),
                NPTypeCode.Single => RepeatScalarTyped<float>(a, repeats, totalSize),
                NPTypeCode.Double => RepeatScalarTyped<double>(a, repeats, totalSize),
                NPTypeCode.Decimal => RepeatScalarTyped<decimal>(a, repeats, totalSize),
                _ => throw new NotSupportedException($"Type {a.GetTypeCode} is not supported.")
            };
        }

        /// <summary>
        ///     Repeat elements of an array with per-element repeat counts.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">Array of repeat counts for each element. Must have the same size as the flattened input array.</param>
        /// <returns>A new array with each element repeated according to the corresponding count in repeats.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        public static NDArray repeat(NDArray a, NDArray repeats)
        {
            a = a.ravel();
            var repeatsFlat = repeats.ravel();

            if (a.size != repeatsFlat.size)
                throw new ArgumentException($"repeats array size ({repeatsFlat.size}) must match input array size ({a.size})");

            // Calculate total output size and validate repeat counts
            long totalSize = 0;
            for (long i = 0; i < repeatsFlat.size; i++)
            {
                // Use Convert.ToInt64 to handle any integer dtype (int32, int64, etc.)
                long count = Convert.ToInt64(repeatsFlat.GetAtIndex(i));
                if (count < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                totalSize += count;
            }

            // Handle empty result
            if (totalSize == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            return a.GetTypeCode switch
            {
                NPTypeCode.Boolean => RepeatArrayTyped<bool>(a, repeatsFlat, totalSize),
                NPTypeCode.Byte => RepeatArrayTyped<byte>(a, repeatsFlat, totalSize),
                NPTypeCode.Int16 => RepeatArrayTyped<short>(a, repeatsFlat, totalSize),
                NPTypeCode.UInt16 => RepeatArrayTyped<ushort>(a, repeatsFlat, totalSize),
                NPTypeCode.Int32 => RepeatArrayTyped<int>(a, repeatsFlat, totalSize),
                NPTypeCode.UInt32 => RepeatArrayTyped<uint>(a, repeatsFlat, totalSize),
                NPTypeCode.Int64 => RepeatArrayTyped<long>(a, repeatsFlat, totalSize),
                NPTypeCode.UInt64 => RepeatArrayTyped<ulong>(a, repeatsFlat, totalSize),
                NPTypeCode.Char => RepeatArrayTyped<char>(a, repeatsFlat, totalSize),
                NPTypeCode.Single => RepeatArrayTyped<float>(a, repeatsFlat, totalSize),
                NPTypeCode.Double => RepeatArrayTyped<double>(a, repeatsFlat, totalSize),
                NPTypeCode.Decimal => RepeatArrayTyped<decimal>(a, repeatsFlat, totalSize),
                _ => throw new NotSupportedException($"Type {a.GetTypeCode} is not supported.")
            };
        }

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        public static unsafe NDArray repeat<T>(T a, int repeats) where T : unmanaged
            => repeat(a, (long)repeats);

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        public static unsafe NDArray repeat<T>(T a, long repeats) where T : unmanaged
        {
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");

            if (repeats == 0)
                return new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(0));

            var ret = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(repeats));
            var dst = (T*)ret.Address;
            for (long j = 0; j < repeats; j++)
                dst[j] = a;
            return ret;
        }

        /// <summary>
        ///     Generic implementation for repeating with scalar repeat count.
        ///     Uses direct pointer access for performance (no allocations per element).
        /// </summary>
        private static unsafe NDArray RepeatScalarTyped<T>(NDArray a, long repeats, long totalSize) where T : unmanaged
        {
            var ret = new NDArray(a.GetTypeCode, Shape.Vector(totalSize));
            var src = (T*)a.Address;
            var dst = (T*)ret.Address;
            long srcSize = a.size;

            long outIdx = 0;
            for (long i = 0; i < srcSize; i++)
            {
                T val = src[i];
                for (long j = 0; j < repeats; j++)
                    dst[outIdx++] = val;
            }

            return ret;
        }

        /// <summary>
        ///     Generic implementation for repeating with per-element repeat counts.
        ///     Uses direct pointer access for performance (no allocations per element).
        /// </summary>
        private static unsafe NDArray RepeatArrayTyped<T>(NDArray a, NDArray repeatsFlat, long totalSize) where T : unmanaged
        {
            var ret = new NDArray(a.GetTypeCode, Shape.Vector(totalSize));
            var src = (T*)a.Address;
            var dst = (T*)ret.Address;
            long srcSize = a.size;

            long outIdx = 0;
            for (long i = 0; i < srcSize; i++)
            {
                // Use Convert.ToInt64 to handle any integer dtype (int32, int64, etc.)
                long count = Convert.ToInt64(repeatsFlat.GetAtIndex(i));
                T val = src[i];
                for (long j = 0; j < count; j++)
                    dst[outIdx++] = val;
            }

            return ret;
        }
    }
}
