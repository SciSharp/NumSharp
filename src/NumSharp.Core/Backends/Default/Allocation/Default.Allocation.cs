using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override IStorage GetStorage(Type dtype)
            => BackendFactory.GetStorage(dtype);

        public override IStorage GetStorage(NPTypeCode typeCode)
            => BackendFactory.GetStorage(typeCode);
    }
}
