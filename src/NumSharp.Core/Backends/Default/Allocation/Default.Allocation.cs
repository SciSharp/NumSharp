using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override UnmanagedStorage GetStorage(Type dtype)
        {
            return new UnmanagedStorage(dtype) {Engine = this};
        }

        public override UnmanagedStorage GetStorage(NPTypeCode typeCode)
        {
            return new UnmanagedStorage(typeCode) {Engine = this};
        }
    }
}
