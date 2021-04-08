using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log2(in NDArray nd, Type dtype) => Log2(nd, dtype?.GetTypeCode());

        public override NDArray Log2(in NDArray nd, NPTypeCode? typeCode = null)
        {
            throw new NotImplementedException("");
        }
    }
}
