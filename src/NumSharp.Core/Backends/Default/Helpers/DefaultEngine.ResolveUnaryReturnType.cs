using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        [MethodImpl((MethodImplOptions)768)]
        public NPTypeCode ResolveUnaryReturnType(in NDArray nd, Type @override) => ResolveUnaryReturnType(nd, @override?.GetTypeCode());

        [MethodImpl((MethodImplOptions)768)]
        public NPTypeCode ResolveUnaryReturnType(in NDArray nd, NPTypeCode? @override)
        {
            if (!@override.HasValue)
                return nd.GetTypeCode.GetComputingType();

            var over = @override.Value;
            if (over < NPTypeCode.Single)
                throw new IncorrectTypeException($"No loop matching the specified signature and casting was found for ufunc {nameof(Sin)}");

            return over;
        }
    }
}
