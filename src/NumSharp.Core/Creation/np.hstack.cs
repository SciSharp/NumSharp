using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray hstack(params NDArray[] nps)
        {
            var np1 = nps[0];

            var npn = new NDArray[nps.Length-1];
            for (int idx = 1;idx < nps.Length;idx++)
                npn[idx-1] = nps[idx];

            return np1.hstack(npn);
        }
        public static NDArray hstack<T>(params NDArray[] nps)
        {
            var np1 = nps[0];

            var npn = new NDArray[nps.Length - 1];
            for (int idx = 1; idx < nps.Length; idx++)
                npn[idx - 1] = nps[idx];

            return np1.hstack<T>(npn);
        }
    }
}
