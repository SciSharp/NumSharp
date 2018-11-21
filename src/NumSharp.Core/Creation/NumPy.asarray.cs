using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
	public static partial class NumPyExtensions
	{
        public static NDArrayGeneric<T> asarray<T>(this NumPyGeneric<T> np, IEnumerable<T> array, int ndim = 1)
        {
            return np.array(array);
        }
    }
}
