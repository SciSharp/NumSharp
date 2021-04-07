using System;
using System.Diagnostics;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DoubleEngine
    {
        public unsafe override NDArray Log(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var @out = new NDArray<double>(nd.Shape, false);
            var len = @out.size;
            var out_addr = @out.Address;
            var addr = (double*)nd.Address;

            for (int i = 0; i < len; i++)
                *(out_addr + i) = Math.Log(*(addr + i));

            return @out;
        }
    }
}
