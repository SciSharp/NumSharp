using System;
using System.Diagnostics;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(Type type)
        {
            switch (type.Name)
            {
                case "Boolean":
                    return EngineCache<BooleanEngine>.Value;
                case "Int32":
                    return EngineCache<Int32Engine>.Value;
                case "Double":
                    return EngineCache<DoubleEngine>.Value;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                    return EngineCache<BooleanEngine>.Value;
                case NPTypeCode.Int32:
                    return EngineCache<Int32Engine>.Value;
                case NPTypeCode.Double:
                    return EngineCache<DoubleEngine>.Value;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

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
