using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp
{
    /// <summary>
    /// API bridge between NumSharp and Python NumPy  
    /// </summary>
    public class NumPy<T>
    {
        public NDArray<double> absolute(NDArray<double> np)
        {
            return np.Absolute();
        }

        public NDArray<double> amax(NDArray<double> np, int? axis = null)
        {
            return np.AMax(axis);
        }

        public NDArray<double> amin(NDArray<double> np, int? axis = null)
        {
            return np.AMin(axis);
        }

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
                        n.arange(stop, start, step);
                        return n as NDArray<T>;
                    }

                case "Double":
                    {
                        var n = new NDArray<double>();
                        n.arange(stop, start, step);
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
            n.Shape = new Shape(new int[] { data.Length });

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
            n.Shape = new Shape(new int[] { data.Length, data[0].Length });

            return n;
        }

        public NDArray<double> hstack(params NDArray<double>[] nps)
        {
            var n = new NDArray<double>();
            return n.HStack(nps);
        }

        /// <summary>
        /// Returns num evenly spaced samples, calculated over the interval [start, stop].
        /// </summary>
        /// <param name="start"></param>
        /// <param name="stop"></param>
        /// <param name="num"></param>
        /// <returns></returns>
        public NDArray<double> linspace(double start, double stop, int num = 50)
        {
            return new NDArray<double>().linspace(start, stop, num);
        }

        public NDArrayRandom random 
        {
            get
            {
                return new NDArrayRandom();
            }
        }

        public NDArray<int> reshape(NDArray<int> np, params int[] shape)
        {
            np.Shape = new Shape(shape);

            return np;
        }

        public NDArray<double> sin(NDArray<double> nd)
        {
            return nd.sin();
        }

        public NDArray<double> vstack(params NDArray<double>[] nps)
        {
            var n = new NDArray<double>();
            return n.VStack(nps);
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
