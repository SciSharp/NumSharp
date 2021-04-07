using System;
using System.Diagnostics;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class Int32Engine
    {
        public unsafe override NDArray Abs(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var ret = nd.Clone();
            var len = nd.size;

            var out_addr = (int*)ret.Address;
            for (int i = 0; i < len; i++)
            {
                if (*(out_addr + i) < 0)
                    *(out_addr + i) = Math.Abs(*(out_addr + i));
            }

            return ret;
        }
    }
}
