using System;
using DecimalMath;
using NumSharp.Utilities;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log(in NDArray nd, Type dtype) => Log(nd, dtype?.GetTypeCode());

        public override NDArray Log(in NDArray nd, NPTypeCode? typeCode = null)
        {
            throw new NotImplementedException("");
        }
    }
}
