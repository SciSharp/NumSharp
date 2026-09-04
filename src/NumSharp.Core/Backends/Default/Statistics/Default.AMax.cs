using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray AMax(NDArray nd, int? axis = null, Type dtype = null, bool keepdims = false)
        {
            return ReduceAMax(nd, axis, keepdims, dtype);
        }
    }
}
