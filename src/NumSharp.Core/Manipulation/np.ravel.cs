using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Manipulation
{
    public partial class NumPy
    {
        public NDArray ravel(matrix mx)
        {
            var nd = new NDArray(mx.dtype, mx.size);

            switch (mx.dtype.Name)
            {
                case "Double":
                    nd.Storage.SetData(mx.Storage.GetData<double>());
                    break;
                case "Int32":
                    nd.Storage.SetData(mx.Storage.GetData<int>());
                    break;
            }

            return nd;
        }
    }
}
