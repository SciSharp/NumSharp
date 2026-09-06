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
