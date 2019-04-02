using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray unique<T>()
        {
            var nd = new NDArray(dtype);
            var data = Storage.GetData<T>().Distinct().ToArray();
            nd.Storage.SetData(data);
            
            nd.Storage.Reshape(data.Length);

            return nd;
        }
    }
}
