using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp
{
    /// <summary>
    /// NumPy bridge
    /// </summary>
    public class NumPy<T>
    {
        public NDArray<T> arange(int stop)
        {
            return arange(0, stop);
        }

        public NDArray<T> arange(int start, int stop, int step = 1)
        {
            if (typeof(T) == typeof(int))
            {
                var n = new NDArray<int>();
                n.ARange(stop, start, step);

                return n as NDArray<T>;
            }
            else if (typeof(T) == typeof(double))
            {
                var n = new NDArray<double>();
                n.ARange(stop, start, step);

                return n as NDArray<T>;
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        public NDArray<int> reshape(NDArray<int> np, params int[] shape)
        {
            np.Shape = shape;

            return np;
        }
    }
}
