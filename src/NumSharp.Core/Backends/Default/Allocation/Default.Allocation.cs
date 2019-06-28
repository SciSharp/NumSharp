using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public IStorage GetStorage(Type dtype)
        {
            return new TypedArrayStorage(dtype) { Engine = this };
        }

        public IStorage GetStorage(NPTypeCode typeCode)
        {
            return new TypedArrayStorage(typeCode) { Engine = this };
        }
    }
}
