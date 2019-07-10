using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray transpose()
            => np.transpose(this);
    }
}
