using System;
using System.Diagnostics;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class Int32Engine
    {
        public unsafe override NDArray Negate(in NDArray nd)
        {
            if (nd.size == 0)
                return nd.Clone(); //return new to maintain immutability.

            var @out = new NDArray(nd.dtype, nd.Shape, false);
            var out_addr = (int*)@out.Address;
            var addr = (int*)nd.Address;
            var len = nd.size;

            for (int i = 0; i < len; i++)
            {
                var val = *(addr + i);
                if (val == 0)
                    *(out_addr + i) = 0;
                else
                    *(out_addr + i) = 0 - val;
            }

            return @out;
        }
    }
}
