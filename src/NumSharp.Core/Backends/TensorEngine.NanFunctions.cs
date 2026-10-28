// The functions of numpy/lib/_nanfunctions_impl.py: nansum, nanprod, nanmin and nanmax so far; the plan's
// phase PE moves the rest of the family here (nanquantile, nanmedian, nancumsum, nanmean, nanvar, ...).

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        // NaN-aware reductions
        public abstract NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false);
    }
}
