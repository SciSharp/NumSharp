using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override IStorage GetStorage(Type dtype)
            => dtype.Name switch
            {
                "Boolean" => new StorageOfBoolean(),
                _ => new StorageOfInt32()
            };

        public override IStorage GetStorage(NPTypeCode typeCode)
            => typeCode switch
            {
                NPTypeCode.Boolean => new StorageOfBoolean(),
                _ => new StorageOfInt32()
            };
    }
}
