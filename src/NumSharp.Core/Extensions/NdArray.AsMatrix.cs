using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static Matrix AsMatrix(this NDArray nd)
        {
            var npAsMatrix = new Matrix(nd);

            npAsMatrix.reshape(nd.shape);

            return npAsMatrix;
        }
    }
}
