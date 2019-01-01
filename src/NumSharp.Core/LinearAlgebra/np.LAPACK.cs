using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.LAPACKProvider
{
    public enum LAPACKProvider
    {
        NetLib    
    }
}
namespace NumSharp.Core
{
    public static partial class np
    {
        public static NumSharp.Core.LAPACKProvider.LAPACKProvider LAPACKProvider = NumSharp.Core.LAPACKProvider.LAPACKProvider.NetLib;
    }
}
