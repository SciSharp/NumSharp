using System;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class SingleEngine
    {
        public unsafe override NDArray Log2(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = new NDArray<double>(nd.Shape, false);
            var len = @out.size;
            var out_addr = @out.Address;
            var addr = (float*)nd.Address;

            for (int i = 0; i < len; i++)
                *(out_addr + i) = Math.Log(*(addr + i), 2d);

            return @out;
        }
    }
}
