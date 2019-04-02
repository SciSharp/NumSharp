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
            nd.Storage.SetData(new string[] { data });
            return nd;
        }

        public static NDArray asanyarray<T>(T data) where T : struct
        {
            var nd = new NDArray(typeof(T), new int[0]);
            nd.Storage.SetData(new T[] { data });
            return nd;
        }

        public static NDArray asanyarray(string[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(string), data.Length);
            nd.Storage.SetData(data);
            return nd;
        }

        public static NDArray asanyarray<T>(T[] data, int ndim = 1) where T : struct
        {
            var nd = new NDArray(typeof(T), data.Length);
            nd.Storage.SetData(data);
            return nd;
        }
    }
}
