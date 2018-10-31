using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray_Legacy<TData>
    {
        public NDArrayRandom Random()
        {
            var rand = new NDArrayRandom();
            return rand;
        }
    }
}
