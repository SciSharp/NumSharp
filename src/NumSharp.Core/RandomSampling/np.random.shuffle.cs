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
        /// <param name="x">The array or list to be shuffled.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
        ///     <br/>
        ///     This function only shuffles the array along the first axis of a multi-dimensional array.
        ///     The order of sub-arrays is changed but their contents remain the same.
        ///     <br/>
        ///     Note: NumPy's Generator API (rng.shuffle) supports an axis parameter, but the legacy
        ///     np.random.shuffle does not. This implementation matches the legacy API.
        /// </remarks>
        /// <example>
        ///     <code>
        ///     // 1D array - elements are shuffled
        ///     var arr = np.arange(10);
        ///     np.random.shuffle(arr);
        ///
        ///     // 2D array - rows are shuffled, contents within rows unchanged
        ///     var arr2d = np.arange(9).reshape(3, 3);
        ///     np.random.shuffle(arr2d);
        ///     // e.g. [[6,7,8], [0,1,2], [3,4,5]] - rows reordered
        ///     </code>
        /// </example>
        public void shuffle(NDArray x)
        {
            if (x.ndim == 0)
                throw new ArgumentException("cannot shuffle a 0-dimensional array", nameof(x));

            var n = x.shape[0];  // Always shuffle along first axis
            if (n <= 1)
                return; // Nothing to shuffle

            // For 1D contiguous arrays, use optimized path
            if (x.ndim == 1 && x.Shape.IsContiguous)
            {
                Shuffle1DContiguous(x, n);
                return;
            }

            // For multi-dimensional arrays, shuffle along axis 0
            // Fisher-Yates shuffle using NextInt64 for full long range support
            for (long i = n - 1; i > 0; i--)
            {
                long j = randomizer.NextLong(i + 1);
                if (i != j)
                {
                    SwapSlicesAxis0(x, i, j);
                }
            }
        }

        /// <summary>
        ///     Optimized shuffle for 1D contiguous arrays.
        /// </summary>
        private unsafe void Shuffle1DContiguous(NDArray x, long n)
        {
            // Random.Next only supports int range
            if (n > int.MaxValue)
                throw new NotSupportedException($"Shuffle not supported for arrays with size {n} > int.MaxValue");

            var itemSize = x.dtypesize;
            var addr = (byte*)x.Address;

            // Allocate temp buffer for swapping
            var temp = stackalloc byte[itemSize];

            // Fisher-Yates shuffle
            int nInt = (int)n;
            for (int i = nInt - 1; i > 0; i--)
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
        ///     Swap two slices along axis 0.
        /// </summary>
        private static void SwapSlicesAxis0(NDArray x, long i, long j)
        {
            // Get slices at indices i and j along axis 0
            var sliceI = x[i];
            var sliceJ = x[j];

            // Create a temporary copy of slice i
            var temp = sliceI.copy();

            // Copy j to i
            np.copyto(sliceI, sliceJ);

            // Copy temp (original i) to j
            np.copyto(sliceJ, temp);
        }
    }
}
