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
            if(start > stop)
            {
                throw new Exception("parameters invalid");
            }

            switch (typeof(T).Name)
            {
                case "Int32":
                    {
                        var n = new NDArray<int>();
                        n.ARange(stop, start, step);
                        return n as NDArray<T>;
                    }

                case "Double":
                    {
                        var n = new NDArray<double>();
                        n.ARange(stop, start, step);
                        return n as NDArray<T>;
                    }
                default:
                    throw new NotImplementedException();
            }
        }

        public NDArray<int> reshape(NDArray<int> np, params int[] shape)
        {
            np.Shape = shape;

            return np;
        }

        public NDArray<T> zeros(params int[] shape)
        {
            switch (typeof(T).Name)
            {
                case "Int32":
                    {
                        var n = new NDArray<int>();
                        n.Zeros(shape);
                        return n as NDArray<T>;
                    }

                case "Double":
                    {
                        var n = new NDArray<double>();
                        n.Zeros(shape);
                        return n as NDArray<T>;
                    }
                default:
                    throw new NotImplementedException();
            }
        }
    }
}
