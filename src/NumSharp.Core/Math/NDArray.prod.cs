using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Return the product of array elements over a given axis.
        /// </summary>
        /// <param name="axis"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public NDArray prod(int axis = -1, Type dtype = null)
        {
            var nd = this;
            NDArray result = null;
            if (nd.size == 0) return 1;

            if(axis == -1)
            {
                switch (nd.dtype.Name)
                {
                    case "Int32":
                        {
                            int prod = 1;
                            for (int i = 0; i < nd.size; i++)
                                prod *= nd.Data<int>(i);
                            result = prod;
                        }
                        break;
                    case "Int64":
                        {
                            long prod = 1;
                            for (int i = 0; i < nd.size; i++)
                                prod *= nd.Data<long>(i);
                            result = prod;
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
