using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Mean(in NDArray nd, int axis, Type dtype, bool keepdims = false)
        {
            return Mean(nd, axis, dtype != null ? dtype.GetTypeCode() : default(NPTypeCode?), keepdims);
        }

        public override NDArray Mean(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false)
        {
            return ReduceMean(nd, axis, keepdims, typeCode);
        }
    }
}
