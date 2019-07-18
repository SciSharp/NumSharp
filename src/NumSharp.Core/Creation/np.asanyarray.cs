using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray asanyarray(string data)
        {
            var nd = new NDArray(typeof(string), new int[0]);
            nd.ReplaceData(new string[] {data});
            return nd;
        }

        public static NDArray asanyarray<T>(T data) where T : struct
        {
            var nd = new NDArray(typeof(T), new int[0]);
            nd.ReplaceData(new T[] {data});
            return nd;
        }

        public static NDArray asanyarray(string[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(string),new Shape(data.Length));
            nd.ReplaceData(data);
            return nd;
        }

        public static NDArray asanyarray<T>(T[] data, int ndim = 1) where T : struct
        {
            var nd = new NDArray(typeof(T), new Shape(data.Length));
            nd.ReplaceData(data);
            return nd;
        }
    }
}
