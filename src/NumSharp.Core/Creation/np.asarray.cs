using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp
{
    public static partial class np
    {
        public static NDArray asarray(string data)
        {
            var nd = new NDArray(typeof(string), new int[0]);
            nd.SetData(new string[] { data });
            return nd;
        }

        public static NDArray asarray<T>(T data) where T : struct
        {
            var nd = new NDArray(typeof(T), new int[0]);
            nd.SetData(new T[] { data });
            return nd;
        }

        public static NDArray asarray(string[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(string), data.Length);
            nd.SetData(data);
            return nd;
        }

        public static NDArray asarray<T>(T[] data, int ndim = 1) where T : struct
        {
            var nd = new NDArray(typeof(T), data.Length);
            nd.SetData(data);
            return nd;
        }
    }
}
