using System;
using System.Diagnostics;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        [DebuggerNonUserCode]
        public static IStorage GetStorage(object x, NPTypeCode? typeCode = null)
        {
            if (!typeCode.HasValue || x.GetType() == typeCode.Value.AsType())
            {
                return x switch
                {
                    bool bool_x => new StorageOfBoolean(bool_x),
                    byte byte_x => new StorageOfByte(byte_x),
                    int int_x => new StorageOfInt32(int_x),
                    long int_x => new StorageOfInt64(int_x),
                    float float_x => new StorageOfSingle(float_x),
                    double double_x => new StorageOfDouble(double_x),
                    _ => throw new NotImplementedException("")
                };
            }
            else
            {
                return typeCode switch
                {
                    NPTypeCode.Boolean => new StorageOfBoolean(Convert.ToBoolean(x)),
                    NPTypeCode.Byte => new StorageOfByte(Convert.ToByte(x)),
                    NPTypeCode.Int32 => new StorageOfInt32(Convert.ToInt32(x)),
                    NPTypeCode.Int64 => new StorageOfInt64(Convert.ToInt64(x)),
                    NPTypeCode.Single => new StorageOfSingle(Convert.ToSingle(x)),
                    NPTypeCode.Double => new StorageOfDouble(Convert.ToDouble(x)),
                    _ => throw new NotImplementedException("")
                };
            }
        }

        [DebuggerNonUserCode]
        public static IStorage GetStorage(Type type)
        {
            if (type.IsArray)
                return GetStorage(type.GetElementType());
            else
                return type.Name switch
                {
                    "Boolean" => new StorageOfBoolean(),
                    "Char" => new StorageOfChar(),
                    "Byte" => new StorageOfByte(),
                    "Int32" => new StorageOfInt32(),
                    "Int64" => new StorageOfInt64(),
                    "Single" => new StorageOfSingle(),
                    "Double" => new StorageOfDouble(),
                    _ => throw new NotImplementedException("")
                };
        }

        [DebuggerNonUserCode]
        public static IStorage GetStorage<T>(T x) where T : unmanaged
            => x switch
            {
                bool bool_x => new StorageOfBoolean(bool_x),
                char char_x => new StorageOfChar(char_x),
                byte byte_x => new StorageOfByte(byte_x),
                int int_x => new StorageOfInt32(int_x),
                long int_x => new StorageOfInt64(int_x),
                float float_x => new StorageOfSingle(float_x),
                double double_x => new StorageOfDouble(double_x),
                _ => throw new NotImplementedException("")
            };

        [DebuggerNonUserCode]
        public static IStorage GetStorage<T>(T[] x, Shape? shape)
            => x switch
            {
                bool[] bool_x => new StorageOfBoolean(bool_x, shape),
                char[] char_x => new StorageOfChar(char_x),
                byte[] byte_x => new StorageOfByte(byte_x),
                int[] int_x => new StorageOfInt32(int_x, shape),
                long[] int_x => new StorageOfInt64(int_x, shape),
                float[] float_x => new StorageOfSingle(float_x, shape),
                double[] double_x => new StorageOfDouble(double_x, shape),
                _ => throw new NotImplementedException("")
            };

        [DebuggerNonUserCode]
        public static IStorage GetStorage(ValueType x)
            => x switch
            {
                bool bool_x => new StorageOfBoolean(bool_x),
                char char_x => new StorageOfChar(char_x),
                byte byte_x => new StorageOfByte(byte_x),
                int int_x => new StorageOfInt32(int_x),
                long long_x => new StorageOfInt64(long_x),
                float float_x => new StorageOfSingle(float_x),
                double double_x => new StorageOfDouble(double_x),
                _ => throw new NotImplementedException("")
            };

        [DebuggerNonUserCode]
        public static IStorage GetStorage(NPTypeCode typeCode)
            => GetStorage(typeCode.AsType());

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(Type type)
        {
            if (type.IsArray)
                return GetEngine(type.GetElementType());
            else
                return type.Name switch
                {
                    "Boolean" => EngineCache<BooleanEngine>.Value,
                    "Char" => EngineCache<ByteEngine>.Value,
                    "Byte" => EngineCache<ByteEngine>.Value,
                    "Int32" => EngineCache<Int32Engine>.Value,
                    "Int64" => EngineCache<Int64Engine>.Value,
                    "Single" => EngineCache<SingleEngine>.Value,
                    "Double" => EngineCache<DoubleEngine>.Value,
                    _ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
                };
        }

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
