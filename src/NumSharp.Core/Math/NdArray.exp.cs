using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="dtype">The dtype of the returned NDArray</param>
        /// <returns>The base-e exponential of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html</remarks>
        public NDArray exp(Type dtype = null)
            => BackendFactory.GetEngine().Exp(this, dtype);

        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The base-e exponential of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html</remarks>
        public NDArray exp(NPTypeCode typeCode)
            => BackendFactory.GetEngine().Exp(this, typeCode);
    }
}
