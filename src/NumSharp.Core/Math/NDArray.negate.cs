using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Negates all values by performing: -x
        /// </summary>
        public NDArray negate()
        {
            return TensorEngine.Negate(this);
        }
    }
}
