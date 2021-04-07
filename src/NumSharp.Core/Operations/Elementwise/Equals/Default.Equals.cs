#if _REGEN_TEMPLATE
%template "./Add/Default.Add.#1.cs" for every supported_dtypes, supported_dtypes_lowercase
#endif

using System;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray<bool> Compare(in NDArray x, in NDArray y)
        {
            throw new NotImplementedException("");
        }
    }
}
