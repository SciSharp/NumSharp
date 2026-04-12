using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <exception cref="ValueError">If indices_or_sections is an integer and does not result in equal division.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            // Normalize axis
            int ndim = ary.ndim;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {ndim}");

            long N = ary.shape[axis];
            if (N % indices_or_sections != 0)
                throw new ArgumentException("array split does not result in an equal division");

            return array_split(ary, indices_or_sections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, long[] indices, int axis = 0)
        {
            return array_split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, int[] indices, int axis = 0)
        {
            var longIndices = new long[indices.Length];
            for (int i = 0; i < indices.Length; i++)
                longIndices[i] = indices[i];
            return array_split(ary, longIndices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N sub-arrays along axis.
        /// If N does not divide the array equally, it returns l % n sub-arrays of size
        /// l//n + 1 and the rest of size l//n.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// The only difference between split and array_split is that array_split allows
        /// indices_or_sections to be an integer that does not equally divide the axis.
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            if (indices_or_sections <= 0)
                throw new ArgumentException("number sections must be larger than 0.");

            // Normalize axis
            int ndim = ary.ndim;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {ndim}");

            long Ntotal = ary.shape[axis];
            int Nsections = indices_or_sections;

            // Calculate division points
            // l % n sub-arrays of size l//n + 1, rest of size l//n
            long Neach_section = Ntotal / Nsections;
            long extras = Ntotal % Nsections;

            // Build division points array
            var div_points = new long[Nsections + 1];
            div_points[0] = 0;
            long cumulative = 0;
            for (int i = 0; i < Nsections; i++)
            {
                // First 'extras' sections get size Neach_section + 1
                cumulative += (i < extras) ? Neach_section + 1 : Neach_section;
                div_points[i + 1] = cumulative;
            }

            return SplitByDivPoints(ary, div_points, Nsections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// If an index exceeds the dimension of the array along axis, an empty sub-array
        /// is returned correspondingly.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, long[] indices, int axis = 0)
        {
            // Normalize axis
            int ndim = ary.ndim;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {ndim}");

            long Ntotal = ary.shape[axis];
            int Nsections = indices.Length + 1;

            // Build division points: [0] + indices + [Ntotal]
            var div_points = new long[Nsections + 1];
            div_points[0] = 0;
            for (int i = 0; i < indices.Length; i++)
            {
                div_points[i + 1] = indices[i];
            }
            div_points[Nsections] = Ntotal;

            return SplitByDivPoints(ary, div_points, Nsections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// A 1-D array of sorted integers indicating where along axis the array is split.
        /// For example, [2, 3] would result in ary[:2], ary[2:3], ary[3:].
        /// If an index exceeds the dimension of the array along axis, an empty sub-array
        /// is returned correspondingly.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int[] indices, int axis = 0)
        {
            var longIndices = new long[indices.Length];
            for (int i = 0; i < indices.Length; i++)
                longIndices[i] = indices[i];
            return array_split(ary, longIndices, axis);
        }

        /// <summary>
        /// Internal helper to split array at given division points along an axis.
        /// Matches NumPy's approach: swap axis to front, slice, swap back.
        /// </summary>
        private static NDArray[] SplitByDivPoints(NDArray ary, long[] div_points, int Nsections, int axis)
        {
            var sub_arys = new NDArray[Nsections];

            // NumPy's approach: swap target axis to axis 0, slice, then swap back
            // This works because slicing along axis 0 is straightforward
            NDArray sary = swapaxes(ary, axis, 0);

            for (int i = 0; i < Nsections; i++)
            {
                long st = div_points[i];
                long end = div_points[i + 1];

                // Build slices for axis 0 (which is the swapped target axis)
                // We want sary[st:end, ...]
                var slices = new Slice[sary.ndim];
                slices[0] = new Slice(st, end);
                for (int d = 1; d < sary.ndim; d++)
                    slices[d] = Slice.All;

                NDArray sub = sary[slices];

                // Swap axis back
                sub_arys[i] = swapaxes(sub, axis, 0);
            }

            return sub_arys;
        }
    }
}
