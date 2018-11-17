using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp
{
	public static partial class NumPyExtensions
	{
        public static NDArray<T> asarray<T>(this NumPy<T> np, IEnumerable<T> array, int ndim = 1)
        {
            return np.array(array);
        }
    }
}
