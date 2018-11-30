using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray sin(NDArray nd)
        {
            return nd.sin();
        }
    }
}
