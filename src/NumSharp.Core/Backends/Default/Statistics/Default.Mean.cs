using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Mean(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false)
        {
            return ReduceMean(nd, axis, keepdims, dtype);
        }
    }
}
