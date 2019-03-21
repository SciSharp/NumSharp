using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
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
            if(ndim == 2 && nd2.ndim == 1)
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
                var nd = new NDArray(dtype, new Shape(shape[0], nd2.shape[1]));
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < shape[1]; i++)
                            for (int j = 0; j < nd2.shape[0]; j++)
                                nd[i, j] = nd.Data<int>(i, j) + Data<int>(i, j) * nd2.Data<int>(j);
                        break;
                }
                return nd;
            }

            throw new NotImplementedException($"dot {ndim} * {nd2.ndim}");
        }
    }    
}
