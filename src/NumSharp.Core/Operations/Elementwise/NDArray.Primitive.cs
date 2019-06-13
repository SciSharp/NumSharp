using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray x, NDArray y)
            => np.add(x, y);

        public static NDArray operator -(NDArray x, NDArray y)
            => np.subtract(x, y);

        public static NDArray operator *(NDArray x, NDArray y)
            => np.multiply(x, y);

        public static NDArray operator /(NDArray x, NDArray y)
            => np.divide(x, y);

        public static NDArray operator -(NDArray x)
            => BackendFactory.GetEngine().Negate(x); //access engine directly since there is no np.negate(x)

        public static NDArray operator +(NDArray x)
            => x.copy(); //to maintain immutability.
    }
}
