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
            var nd = new NDArray(dtype);
            var data = Data<T>().Distinct().ToArray();
            nd.Set(data);
            nd.Storage.Shape = new Shape(data.Length);

            return nd;
        }
    }
}
