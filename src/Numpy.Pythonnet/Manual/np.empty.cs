using System;
using System.Collections.Generic;
using System.Text;
using Numpy.Models;

namespace Numpy
{

    public static partial class np
    {

        /// <summary>
        /// Create an array.
        /// 
        /// <param name="shape">
        /// The shape of the empty ndarray
        /// </param>
        /// <returns>
        /// An array object satisfying the specified requirements.
        /// </returns>
        public static NDarray empty(params int[] shape)
        {
            return NumPy.Instance.empty(new Shape(shape));
        }
    }
}
