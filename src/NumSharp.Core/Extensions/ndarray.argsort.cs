using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray argsort<T>(int axis = -1)
        {
            if(ndim == 1)
            {
                var map = new Dictionary<int, T>();
                for (int i = 0; i < size; i++)
                    map[i] = Data<T>(i);
                var array = map.OrderBy(x => x.Value).Select(x => x.Key).ToArray();
                return np.array(array, typeof(T), ndim);
            }
            else
            {
                throw new NotImplementedException("argsort");
            }
        }
    }
}
