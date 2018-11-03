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

        public NDArray<T> array(T[] data)
        {
            var n = new NDArray<T>();
            n.Data = data;
            n.Shape[0] = data.Length;

            return n;
        }

        public NDArray<T> array(T[][] data)
        {
            int size = data.Length * data[0].Length;
            var all = new T[size];

            int idx = 0;
            for (int row = 0; row < data.Length; row++)
            {
                for (int col = 0; col < data[row].Length; col++)
                {
                    all[idx] = data[row][col];
                    idx++;
                }
            }

            var n = new NDArray<T>();
            n.Data = all;
            n.Shape = new List<int> { data.Length, data[0].Length };

            return n;
        }
    }
}
