using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray asarray(double[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(double), data.Length);
            nd.Storage.SetData(data);
            return nd;
        }

        public static NDArray asarray(float[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(float), data.Length);
            nd.Storage.SetData(data);
            return nd;
        }
        /*
        public static NDArray asarray(matrix mx, int ndim = 1)
        {
            var nd = new NDArray(mx.dtype, mx.shape);
            nd.Storage = mx.Storage;
            return nd;
        }
        */
    }
}
