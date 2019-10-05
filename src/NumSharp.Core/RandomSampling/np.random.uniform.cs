using System;
using System.Linq;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a uniform distribution.
        ///     Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be less than high. The default value is 1.0.</param>
        /// <param name="shape">Output shape. If the given shape is, e.g., m, n, k, then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. </param>
        /// <returns>NDArray with values of type <see cref="double"/></returns>
        public NDArray uniform(double low, double high, Shape shape)
        {
            return uniform(low, high, shape.dimensions);
        }
        
        /// <summary>
        ///     Draw samples from a uniform distribution.
        ///     Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be less than high. The default value is 1.0.</param>
        /// <param name="size">Output shape. If the given shape is, e.g., m, n, k, then m * n * k samples are drawn. If size is None (default), a single value is returned if low and high are both scalars. </param>
        /// <returns>NDArray with values of type <see cref="double"/></returns>
        public NDArray uniform(double low, double high, params int[] size)
        {
            if (size == null || size.Length == 0) //return scalar
            {
                var ret = new NDArray<double>(new Shape(1));
                var data = new double[] {low + randomizer.NextDouble() * (high - low)};
                ret.ReplaceData(data);
                return ret;
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            //parallelism is prohibited to make sure the result come out presistant
            double diff = high - low;
            for (int i = 0; i < result.size; ++i)
                resultArray[i] = low + randomizer.NextDouble() * diff;

            result.ReplaceData(resultArray); //incase of a view //todo! incase of a view?
            return result;
        }

        /// <summary>
        ///     Draw samples from a uniform distribution.
        ///     Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be greater than or equal to low. The default value is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be less than high. The default value is 1.0.</param>
        /// <param name="dType">The type of the output <see cref="NDArray"/></param>
        /// <returns></returns>
        public NDArray uniform(NDArray low, NDArray high, Type dType = null)
        {
            if (!low.shape.SequenceEqual(high.shape))
                throw new IncorrectShapeException();
            dType = dType ?? (low.dtype == high.dtype ? low.dtype : throw new IncorrectTypeException());

            var ret = low + rand(low.shape).astype(dType) * (high - low);
            return dType != null ? ret.astype(dType) : ret;
        }
    }
}
