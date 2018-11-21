using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NumPyExtensions
    {
        public static NDArrayGeneric<T> delete<T>(this NumPy<T> np, NDArrayGeneric<T> nd, IEnumerable<T> delete)
        {            
            var nd1 = np.array(nd.Data.Where(x => !delete.Contains(x)));
            nd1.Shape = new Shape(nd1.Data.Length);

            return nd1;
        }
    }
}
