using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
using NumSharp.Backends;

namespace NumSharp
{
	public static partial class np
	{
        public static NDArray array(Array array, Type dtype = null, int ndim = 1)
        {
            dtype = (dtype == null) ? array.GetType().GetElementType() : dtype;

            var nd = new NDArray(dtype, new Shape(new int[] { array.Length }));

            if ((array.Rank == 1) && ( !array.GetType().GetElementType().IsArray ))
			{
                nd.SetData(array); 
            }
            else 
            {
                throw new Exception("Method is not implemeneted for multidimensional arrays or jagged arrays.");
            }
            
            return nd;
        }

        public static NDArray array<T>(T[][] data)
        {
            var array = data.SelectMany(inner => inner).ToArray();

            return new NDArray(array, new Shape(data.Length, data[0].Length));
        }

        public static NDArray array<T>(T[][][] data)
        {
            var array = data.SelectMany(inner => inner
                .SelectMany(innerInner => innerInner))
                .ToArray();

            return new NDArray(array, new Shape(data.Length, data[0].Length, data[0][0].Length));
        }

        public static NDArray array<T>(T[][][][] data)
        {
            var array = data.SelectMany(inner => inner
                .SelectMany(innerInner => innerInner
                .SelectMany(innerInnerInner => innerInnerInner)))
                .ToArray();

            return new NDArray(array, new Shape(data.Length, data[0].Length, data[0][0].Length));
        }

        public static NDArray array<T>(T[,] data)
        {
            var array = data.Cast<T>().ToArray();

            return new NDArray(array, new Shape(data.GetLength(0), data.GetLength(1)));
        }

        public static NDArray array<T>(T[,,] data)
        {
            var array = data.Cast<T>().ToArray();

            return new NDArray(array, new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2)));
        }

        public static NDArray array<T>(T[,,,] data)
        {
            var array = data.Cast<T>().ToArray();

            return new NDArray(data.Cast<T>().ToArray(), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3)));
        }

        public static NDArray array<T>(params T[] data)
        {
            var nd = new NDArray(typeof(T), data.Length);

            nd.Array = data;

            return nd;
        }
    }
}
