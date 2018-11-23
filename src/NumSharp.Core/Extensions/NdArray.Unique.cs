using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray unique<T>()
        {
            var nd = new NDArray(dtype, Shape);
            nd.Set(Data<T>().Distinct().ToArray());

            return nd;
        }
    }
}
