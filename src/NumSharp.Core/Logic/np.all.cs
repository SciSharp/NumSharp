using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray all(NDArray nd, int axis = -1)
        {
            NDArray result = null;
            if(axis == -1)
            {
                switch (nd.dtype.Name)
                {
                    case "Int32":
                        {
                        }
                        break;
                    case "Int64":
                        {
                        }
                        break;
                }
            }
            else
            {
                throw new NotImplementedException($"np.prod axis {axis}");
            }

            return result;
        }
    }
}
