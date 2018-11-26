using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray dot(NDArray a, NDArray b)
        {
            switch (a.dtype.Name)
            {
                case "Int32":
                    return a.dot<int>(b);
            }

            return null;
        }
    }
}
