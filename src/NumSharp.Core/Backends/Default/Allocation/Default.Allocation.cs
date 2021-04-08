using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override IStorage GetStorage(Type dtype)
            => dtype.Name switch
            {
                "Boolean" => new StorageOfBoolean(),
                "Int32" => new StorageOfInt32(),
                "Single" => new StorageOfSingle(),
                "Double" => new StorageOfDouble(),
                _ => throw new NotImplementedException("")
            };

        public override IStorage GetStorage(NPTypeCode typeCode)
            => GetStorage(typeCode.AsType());
    }
}
