using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Natural logarithm, element-wise.
        ///
        /// The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.The natural logarithm is logarithm in base e.
        /// </summary>
        /// <returns></returns>
        public NDArray log()
            => BackendFactory.GetEngine().Log(this);
    }
}
