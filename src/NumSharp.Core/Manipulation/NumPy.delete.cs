using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray delete<T>(IEnumerable<T> delete)
        {
            var np = new NumPy();
            var nd1 = np.array(Data<T>().Where(x => !delete.Contains(x)).ToArray());
            nd1.Storage.Shape = new Shape(nd1.Size);

            return nd1;
        }
    }
}
