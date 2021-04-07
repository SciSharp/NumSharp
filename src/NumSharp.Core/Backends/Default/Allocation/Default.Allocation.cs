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
                "Double" => new StorageOfDouble(),
                _ => throw new NotImplementedException("")
            };

        public override IStorage GetStorage(NPTypeCode typeCode)
            => typeCode switch
            {
                NPTypeCode.Boolean => new StorageOfBoolean(),
                NPTypeCode.Int32 => new StorageOfInt32(),
                NPTypeCode.Double => new StorageOfDouble(),
                _ => throw new NotImplementedException("")
            };
    }
}
