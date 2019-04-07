using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray max(int axis)
            => amax(axis);

        public T max<T>()
            => amax<T>();
    }
}
