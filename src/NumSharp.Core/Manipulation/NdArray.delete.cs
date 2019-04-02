using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray ravel()
        {
            var nd = copy();
            nd.reshape(Storage.Shape.Size);
            return nd;
        }
    }
}
