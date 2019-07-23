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
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public NDArray log(Type dtype = null)
            => BackendFactory.GetEngine().Log(this, dtype);

        /// <summary>
        ///     Natural logarithm, element-wise.
        /// </summary>
        /// <param name="typeCode">The dtype of the returned NDArray</param>
        /// <returns>The natural logarithm of x, element-wise. This is a scalar NDArray.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html</remarks>
        public NDArray log(NPTypeCode typeCode)
            => BackendFactory.GetEngine().Log(this, typeCode);
    }
}
