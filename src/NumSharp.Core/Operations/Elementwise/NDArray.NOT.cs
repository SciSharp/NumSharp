using System;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        // Scope: MakeGeneric<bool>() mints a typed ALIAS over `result`'s storage (its own ARC
        // ref); the untyped `result` wrapper is then a strand reclaimable only by a future GC +
        // finalizer pass (measured: one bucketed buffer escaped per `!arr` call). The [NDScoped]
        // weaver tracks `result`, yields the alias via scope.Returns, and disposes `result` at
        // exit — the same pattern the typed NDArray<T> &/|/^ operators already carry. `self` is
        // an input constructed before the scope opens, so it is never tracked (rule R2).
        [NDScoped]
        public static unsafe NDArray<bool> operator !(NDArray self)
        {
            // The raw-buffer fast path below walks self.Address LINEARLY (from[i]), which matches the
            // logical C-order of the elements ONLY when self is C-contiguous with offset 0. For any
            // other layout — F-contiguous, transposed, strided, a sliced view with a non-zero offset,
            // or a broadcast (stride-0) view — the linear read visits the backing buffer in memory
            // order, not logical order, and the fresh C-contiguous `result` then holds scrambled
            // values. Route those through the layout-aware logical_not ufunc, which honours strides and
            // offset (correctness over the raw-loop speed; the contiguous hot path is unchanged).
            if (!self.Shape.IsContiguous || self.Shape.offset != 0)
                return np.logical_not(self);

            var result = new NDArray(typeof(bool), self.shape);
            NpFunc.Invoke(self.GetTypeCode, NotExecute<int>, (nint)self.Address, (nint)result.Address, result.size);
            return result.MakeGeneric<bool>();
        }

        private static unsafe void NotExecute<T>(nint fromAddr, nint toAddr, long len) where T : unmanaged, IEquatable<T>
        {
            var from = (T*)fromAddr;
            var to = (bool*)toAddr;
            for (long i = 0; i < len; i++)
                *(to + i) = (*(from + i)).Equals(default);
        }
    }
}
