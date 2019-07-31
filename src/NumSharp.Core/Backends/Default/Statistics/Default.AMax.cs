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
        public override NDArray AMax(in NDArray nd, int axis, Type dtype, bool keepdims = false)
        {
            return AMax(nd, axis, dtype != null ? dtype.GetTypeCode() : default(NPTypeCode?), keepdims);
        }

        public override NDArray AMax(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false)
        {
            return ReduceAMax(nd, axis, keepdims, typeCode);
        }
    }
}
