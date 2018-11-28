using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public static bool operator !=(NDArray np, object obj)
        {
            return !(np == obj);
        }
    }
}
