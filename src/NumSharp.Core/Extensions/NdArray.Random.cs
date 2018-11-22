using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray_Legacy<TData>
    {
        public NumPyRandom Random()
        {
            var rand = new NumPyRandom();
            return rand;
        }
    }
}
