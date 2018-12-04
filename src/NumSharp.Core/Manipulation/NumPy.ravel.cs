using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.Manipulation
{
    public partial class NumPy
    {
        public NDArray ravel(Matrix mx)
        {
            var nd = new NDArray(mx.dtype, mx.size);

            switch (mx.dtype.Name)
            {
                case "Double":
                    nd.Set(mx.float64);
                    break;
                case "Int32":
                    nd.Set(mx.int32);
                    break;
            }

            return nd;
        }
    }
}
