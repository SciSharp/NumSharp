using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Matrix or vector product between given NDArray and 2nd one.
        /// if both NDArrays are 1D, scalar product is returned independend of shape
        /// if both NDArrays are 2D matrix product is returned.
        /// </summary>
        /// <param name="nd2">2nd NDArray</param>
        /// <returns>Scalarproduct or matrix prod</returns>
        public NDArray dot(NDArray nd2)
        {
            if (ndim == 0 && nd2.ndim == 0)
            {
                switch (dtype.Name)
                {
                    case "Int32":
                        return nd2.Data<int>(0) * Data<int>(0);
                }
            }
            else if (ndim == 1 && nd2.ndim == 1)
            {
                int sum = 0;
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < size; i++)
                            sum += Data<int>(i) * nd2.Data<int>(i);
                        break;
                }
                return sum;
            }
            else if (ndim == 2 && nd2.ndim == 1)
            {
                var nd = new NDArray(dtype, new Shape(shape[0]));
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < shape[0]; i++)
                            for (int j = 0; j < nd2.shape[0]; j++)
                                nd.Data<int>()[i] += Data<int>(i, j) * nd2.Data<int>(j);
                        break;
                }
                return nd;
            }
            else if (ndim == 2 && nd2.ndim == 2)
            {
                return np.matmul(this, nd2);
            }

            throw new NotImplementedException($"dot {ndim} * {nd2.ndim}");
        }
    }    
}
