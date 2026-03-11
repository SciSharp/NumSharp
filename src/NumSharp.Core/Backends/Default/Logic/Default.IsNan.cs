using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for Not a Number.
        /// </summary>
        /// <returns>The result is returned as a boolean array.</returns>
        public override NDArray<bool> IsNan(NDArray a)
        {
            var result = new NDArray<bool>(a.Shape, true);

            // Only floating-point types can have NaN values
            // Integer types always return false for all elements
            switch (a.GetTypeCode)
            {
                case NPTypeCode.Single:
                    IsNanFloat(a, result);
                    break;
                case NPTypeCode.Double:
                    IsNanDouble(a, result);
                    break;
                default:
                    // All integer/boolean/decimal types: NaN is not possible, return all false
                    // result is already initialized to false
                    break;
            }

            return result.MakeGeneric<bool>();
        }

        private static unsafe void IsNanFloat(NDArray a, NDArray result)
        {
            var src = (float*)a.Address;
            var dst = (bool*)result.Address;
            int size = a.size;

            if (a.Shape.IsContiguous)
            {
                for (int i = 0; i < size; i++)
                    dst[i] = float.IsNaN(src[i]);
            }
            else
            {
                var iter = new NDIterator<float>(a);
                for (int i = 0; i < size; i++)
                {
                    dst[i] = float.IsNaN(iter.MoveNextReference());
                }
            }
        }

        private static unsafe void IsNanDouble(NDArray a, NDArray result)
        {
            var src = (double*)a.Address;
            var dst = (bool*)result.Address;
            int size = a.size;

            if (a.Shape.IsContiguous)
            {
                for (int i = 0; i < size; i++)
                    dst[i] = double.IsNaN(src[i]);
            }
            else
            {
                var iter = new NDIterator<double>(a);
                for (int i = 0; i < size; i++)
                {
                    dst[i] = double.IsNaN(iter.MoveNextReference());
                }
            }
        }
    }
}
