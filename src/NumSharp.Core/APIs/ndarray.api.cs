using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray log()
             => np.log(this);

        public static NDArray operator +(NDArray x, NDArray y)
            => np.add(x, y);

        public static NDArray operator -(NDArray x, NDArray y)
            => np.subtract(x, y);

        public static NDArray operator *(NDArray x, NDArray y)
            => np.multiply(x, y);

        public static NDArray operator /(NDArray x, NDArray y)
            => np.divide(x, y);
    }
}
