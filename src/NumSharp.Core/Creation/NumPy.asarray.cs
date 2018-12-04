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
        public NDArray asarray(Matrix mx, int ndim = 1)
        {
            var nd = new NDArray(mx.dtype, mx.shape);
            nd.Storage = mx.Storage;
            return nd;
        }
    }
}
