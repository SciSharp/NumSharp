// The functions of numpy/lib/_ufunclike_impl.py that NumSharp's engine implements: isposinf and isneginf.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        // np.isposinf / np.isneginf (numpy/lib/_ufunclike_impl.py). Not ufuncs — no
        // where/dtype; the single settable side is out (positional in NumPy). Complex
        // input is rejected at the np.* layer (NumPy's signbit raises), so the engine
        // predicate never sees Complex.
        public abstract NDArray IsPosInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsNegInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
    }
}
