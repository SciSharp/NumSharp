using System;
using System.Diagnostics;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        [DebuggerNonUserCode]
        public static IStorage GetStorage(Type type)
            => type.Name switch
            {
                "Boolean" => new StorageOfBoolean(),
                "Byte" => new StorageOfByte(),
                "Int32" => new StorageOfInt32(),
                "Int64" => new StorageOfInt64(),
                "Single" => new StorageOfSingle(),
                "Double" => new StorageOfDouble(),
                _ => throw new NotImplementedException("")
            };

        [DebuggerNonUserCode]
        public static IStorage GetStorage(NPTypeCode typeCode)
            => GetStorage(typeCode.AsType());

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(Type type)
            => type.Name switch
            {
                "Boolean" => EngineCache<BooleanEngine>.Value,
                "Byte" => EngineCache<ByteEngine>.Value,
                "Int32" => EngineCache<Int32Engine>.Value,
                "Int64" => EngineCache<Int64Engine>.Value,
                "Single" => EngineCache<SingleEngine>.Value,
                "Double" => EngineCache<DoubleEngine>.Value,
                _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
            };

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(NPTypeCode type)
            => GetEngine(type.AsType());

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(BackendType backendType = BackendType.Default)
        {
            switch (backendType)
            {
                case BackendType.Default:
                    return EngineCache<DefaultEngine>.Value;
                default:
                    throw new ArgumentOutOfRangeException(nameof(backendType), backendType, null);
            }
        }

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine<T>() where T : TensorEngine, new()
        {
            return EngineCache<T>.Value;
        }

        private static class EngineCache<T> where T : TensorEngine, new()
        {
            public static readonly T Value = new T();
        }
    }
}
