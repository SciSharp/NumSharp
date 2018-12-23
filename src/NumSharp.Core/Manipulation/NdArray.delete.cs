using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray ravel()
        {
            Storage.Reshape(Storage.Shape.Size);
            return this;
        }
    }
}
