using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public int ArgMax()
        {
            var np = this;

            var max = np.Data.Max();

            return np.Data.IndexOf(max);
        }
    }
}
