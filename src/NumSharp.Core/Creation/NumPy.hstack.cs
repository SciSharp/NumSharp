using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class NumPyExtensions
    {
        public static NDArray hstack(this NumPy np, params NDArray[] nps)
        {
            var np1 = nps[0];

            var npn = new NDArray[nps.Length-1];
            for (int idx = 1;idx < nps.Length;idx++)
                npn[idx-1] = nps[idx];

            return np1.hstack(npn);
        }
    }
}
