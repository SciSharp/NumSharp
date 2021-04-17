using System;
using System.Diagnostics;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class SingleEngine
    {
        public unsafe override NDArray Log(in NDArray nd, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            if (typeCode.HasValue && typeCode < NPTypeCode.Double)
                throw new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {nameof(Sin)}");

            var @out = new NDArray<float>(nd.Shape, false);
            var len = @out.size;
            var out_addr = @out.Address;
            var addr = (float*)nd.Address;

            for (int i = 0; i < len; i++)
                *(out_addr + i) = Convert.ToSingle(Math.Log(*(addr + i)));

            return @out;
        }
    }
}
