using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public NDArray<TData> Delete(IEnumerable<TData> delete)
        {
            var np = this;
            
            return np.Array(np.Data.Where(x => !delete.Contains(x)));
        }
    }
}
