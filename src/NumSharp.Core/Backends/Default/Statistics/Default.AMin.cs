using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray AMin(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false)
        {
            return ReduceAMin(nd, axis, keepdims, dtype);
        }
    }
}
