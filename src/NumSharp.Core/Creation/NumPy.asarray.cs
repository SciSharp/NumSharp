using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray asarray(double[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(double), data.Length);
            nd.Storage.Set(data);
            return nd;
        }

        public NDArray asarray(float[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(float), data.Length);
            nd.Storage.Set(data);
            return nd;
        }

        public NDArray asarray(matrix mx, int ndim = 1)
        {
            var nd = new NDArray(mx.dtype, mx.shape);
            nd.Storage = mx.Storage;
            return nd;
        }
    }
}
