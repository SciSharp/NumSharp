using NumSharp.Core.Casting;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray[] broadcast_arrays(NDArray nd1, NDArray nd2, bool subok = false)
        {
            var shape = _broadcast_shape(nd1, nd2);

            if (nd1.shape == shape && nd2.shape == shape)
            {
                return new NDArray[] { nd1, nd2 };
            }

            return new NDArray[] { _broadcast_to(nd1, shape, subok, false), _broadcast_to(nd2, shape, subok, false) };
        }

        private static Shape _broadcast_shape(NDArray nd1, NDArray nd2)
        {
            Broadcast b = np.broadcast(nd1, nd2);
            return b.shape;
        }

        private static NDArray _broadcast_to(NDArray nd, Shape shape, bool subok, bool rdonly)
        {
            double[,] table = new double[shape.Dimensions[0], shape.Dimensions[1]];
            if (nd.shape[0] == 1) 
            {// (1,2,3)
                for (int i = 0; i < shape.Dimensions[0]; i++)
                {
                    for (int j = 0; j < shape.Dimensions[1]; j++)
                    {
                        table[i, j] = Convert.ToDouble(nd.Storage.GetData(0, j)); 
                    }
                }
            }
            else if (nd.shape[1] == 1)
            {
                for (int i = 0; i < shape.Dimensions[0]; i++)
                {
                    for (int j = 0; j < shape.Dimensions[1]; j++)
                    {
                        table[i, j] = Convert.ToDouble(nd.Storage.GetData(i, 0));
                    }
                }
            }
            return np.array<double>(table);
        }
    }
}
