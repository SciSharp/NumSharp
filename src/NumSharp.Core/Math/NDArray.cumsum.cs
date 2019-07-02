using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray cumsum()
        {
            return np.cumsum(this);
        }
    }

    public static partial class np
    {
        public static NDArray cumsum(NDArray a)
        {
            // TODO currently no support for multidimensional a
            NDArray cs = np.zeros(a.shape[0]);
            cs[0] = a[0];
            for (int i = 1; i < a.shape[0]; i++)
            {
                cs[i] = cs[i - 1] + a[i];
            }
            return cs;
        }
    }
}
