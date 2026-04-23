using System;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
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
