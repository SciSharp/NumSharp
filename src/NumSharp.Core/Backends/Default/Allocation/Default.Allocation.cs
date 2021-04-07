using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override IStorage GetStorage(Type dtype)
            => dtype switch
            {
                _ => new StorageOfInt32()
            };

        public override IStorage GetStorage(NPTypeCode typeCode)
            => typeCode switch
            {
                _ => new StorageOfInt32()
            };
    }
}
