using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMax(in NDArray a, int axis)
        {
            return ReduceArgMax(a, axis);
        }        
        
        public override NDArray ArgMax(in NDArray a)
        {
            return ReduceArgMax(a, null);
        }
    }
}
