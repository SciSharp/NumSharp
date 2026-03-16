using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Modify a sequence in-place by shuffling its contents.
        /// </summary>
        /// <param name="x">The array to be shuffled.</param>
        /// <param name="axis">The axis along which to shuffle. Default is 0.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html
        ///     <br/>
        ///     This function shuffles the array along the specified axis.
        ///     The order of sub-arrays is changed but their contents remain the same.
        ///     <br/>
        ///     For a 2D array with axis=0, rows are shuffled.
        ///     For a 2D array with axis=1, columns are shuffled (elements within each row are reordered).
        /// </remarks>
        /// <example>
        ///     <code>
        ///     // Shuffle rows of a 2D array
        ///     var arr = np.arange(9).reshape(3, 3);
        ///     np.random.shuffle(arr);  // axis=0 by default
        ///
        ///     // Shuffle columns (within each row)
        ///     np.random.shuffle(arr, axis: 1);
        ///     </code>
        /// </example>
        public void shuffle(NDArray x, int axis = 0)
        {
            if (x.ndim == 0)
                throw new ArgumentException("cannot shuffle a 0-dimensional array", nameof(x));

            // Normalize negative axis
            if (axis < 0)
                axis += x.ndim;

            if (axis < 0 || axis >= x.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {x.ndim}");

            var n = x.shape[axis];
            if (n <= 1)
                return; // Nothing to shuffle

            // For 1D arrays, use optimized path
            if (x.ndim == 1 && x.Shape.IsContiguous)
            {
                Shuffle1DContiguous(x, n);
                return;
            }

            // For multi-dimensional arrays, use slice-based swapping
            // Fisher-Yates shuffle along the specified axis
            for (int i = n - 1; i > 0; i--)
            {
                int j = randomizer.Next(i + 1);
                if (i != j)
                {
                    SwapSlices(x, axis, i, j);
                }
            }
        }

        /// <summary>
        ///     Optimized shuffle for 1D contiguous arrays.
        /// </summary>
        private unsafe void Shuffle1DContiguous(NDArray x, int n)
        {
            var itemSize = x.dtypesize;
            var addr = (byte*)x.Address;

            // Allocate temp buffer for swapping
            var temp = stackalloc byte[itemSize];

            // Fisher-Yates shuffle
            for (int i = n - 1; i > 0; i--)
            {
                int j = randomizer.Next(i + 1);
                if (i != j)
                {
                    var ptrI = addr + (long)i * itemSize;
                    var ptrJ = addr + (long)j * itemSize;

                    // Swap elements
                    Buffer.MemoryCopy(ptrI, temp, itemSize, itemSize);
                    Buffer.MemoryCopy(ptrJ, ptrI, itemSize, itemSize);
                    Buffer.MemoryCopy(temp, ptrJ, itemSize, itemSize);
                }
            }
        }

        /// <summary>
        ///     Swap two slices along a specified axis.
        /// </summary>
        private static void SwapSlices(NDArray x, int axis, int i, int j)
        {
            // Get slices at indices i and j along the specified axis
            var sliceI = GetSliceAtIndex(x, axis, i);
            var sliceJ = GetSliceAtIndex(x, axis, j);

            // Create a temporary copy of slice i
            var temp = sliceI.copy();

            // Copy j to i
            np.copyto(sliceI, sliceJ);

            // Copy temp (original i) to j
            np.copyto(sliceJ, temp);
        }

        /// <summary>
        ///     Get a slice of the array at the specified index along the given axis.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray GetSliceAtIndex(NDArray x, int axis, int index)
        {
            // Build slice specification: all colons except for the specified axis
            var slices = new Slice[x.ndim];
            for (int d = 0; d < x.ndim; d++)
            {
                slices[d] = d == axis ? Slice.Index(index) : Slice.All;
            }
            return x[slices];
        }
    }
}
