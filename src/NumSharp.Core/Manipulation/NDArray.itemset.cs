using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// 
        /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.itemset.html.
        /// </summary>
        public void itemset<T>(Shape shape, T val)
        {
            return;
            //SetData<T>(val, shape);
        }

        public void itemset<T>(int[] shape, T val)
        {
            return;
            //SetData<T>(val, shape);
        }

        public void itemset<T>(int index, T val)
        {
            return;
            //Data<T>()[index] = val;
        }
    }
}
