using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override IStorage GetStorage(Type dtype)
        {
            return new TypedArrayStorage(dtype) { Engine = this };
        }

        public override IStorage GetStorage(NPTypeCode typeCode)
        {
            return new TypedArrayStorage(typeCode) { Engine = this };
        }
    }
}
