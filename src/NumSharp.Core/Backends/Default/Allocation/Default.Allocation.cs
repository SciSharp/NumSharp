using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override UnmanagedStorage GetStorage(Type dtype)
        {
            return new UnmanagedStorage(dtype) {Engine = this};
        }

        // The descriptor form (NumPy's PyArray_NewFromDescr): the descriptor is stored as-is, so a
        // parametric one (a future datetime64[ns]) reaches the array's dtype intact.
        public override UnmanagedStorage GetStorage(DType descr)
        {
            return new UnmanagedStorage(descr) {Engine = this};
        }

        // NPTypeCode is DefaultEngine's internal dtype currency — kept as a (non-abstract) helper
        // the whole engine calls directly; the TensorEngine abstraction exposes only the Type form.
        public UnmanagedStorage GetStorage(NPTypeCode typeCode)
        {
            return new UnmanagedStorage(typeCode) {Engine = this};
        }
    }
}
