using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for finiteness (not infinity and not NaN).
        /// </summary>
        /// <param name="a"></param>
        /// <returns>The result is returned as a boolean array.</returns>
        public override NDArray<bool> IsFinite(NDArray a)
        {
            var result = new NDArray<bool>(a.Shape, true);

            // Only floating-point types can have Inf/NaN values
            // Integer types are always finite
            switch (a.GetTypeCode)
            {
                case NPTypeCode.Single:
                    IsFiniteFloat(a, result);
                    break;
                case NPTypeCode.Double:
                    IsFiniteDouble(a, result);
                    break;
                default:
                    // All integer/boolean/decimal types: always finite, fill with true
                    FillTrue(result);
                    break;
            }

            return result.MakeGeneric<bool>();
        }

        private static unsafe void FillTrue(NDArray result)
        {
            var dst = (bool*)result.Address;
            int size = result.size;
            for (int i = 0; i < size; i++)
                dst[i] = true;
        }

        private static unsafe void IsFiniteFloat(NDArray a, NDArray result)
        {
            var src = (float*)a.Address;
            var dst = (bool*)result.Address;
            int size = a.size;

            if (a.Shape.IsContiguous)
            {
                for (int i = 0; i < size; i++)
                    dst[i] = float.IsFinite(src[i]);
            }
            else
            {
                var iter = new NDIterator<float>(a);
                for (int i = 0; i < size; i++)
                {
                    dst[i] = float.IsFinite(iter.MoveNextReference());
                }
            }
        }

        private static unsafe void IsFiniteDouble(NDArray a, NDArray result)
        {
            var src = (double*)a.Address;
            var dst = (bool*)result.Address;
            int size = a.size;

            if (a.Shape.IsContiguous)
            {
                for (int i = 0; i < size; i++)
                    dst[i] = double.IsFinite(src[i]);
            }
            else
            {
                var iter = new NDIterator<double>(a);
                for (int i = 0; i < size; i++)
                {
                    dst[i] = double.IsFinite(iter.MoveNextReference());
                }
            }
        }
    }
}
