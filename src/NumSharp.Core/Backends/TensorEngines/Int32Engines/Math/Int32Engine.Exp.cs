using System;
using System.Diagnostics;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class Int32Engine
    {
        public unsafe override NDArray Exp(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = new NDArray<double>(nd.Shape, false);
            var len = @out.size;
            var out_addr = @out.Address;
            var addr = (int*)nd.Address;

            for (int i = 0; i < len; i++)
                *(out_addr + i) = Math.Exp(*(addr + i));

            return @out;
        }
    }
}
