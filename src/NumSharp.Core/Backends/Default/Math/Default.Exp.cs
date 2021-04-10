using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Exp(in NDArray nd, Type dtype) => Exp(nd, dtype?.GetTypeCode());

        public override NDArray Exp(in NDArray nd, NPTypeCode? typeCode = null)
        {
            throw new NotImplementedException("");
        }
    }
}
