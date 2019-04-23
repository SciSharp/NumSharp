using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        private static Dictionary<BackendType, ITensorEngine> cache
            = new Dictionary<BackendType, ITensorEngine>();

        public static ITensorEngine GetEngine(BackendType backendType = BackendType.SIMD)
        {
            if(!cache.ContainsKey(backendType))
            {
                switch (backendType)
                {
                    case BackendType.MKL:
                    case BackendType.SIMD:
                        cache[backendType] = new SimdEngine();
                        break;
                    case BackendType.ArrayFire:
                        cache[backendType] = new ArrayFireEngine();
                        break;
                    default:
                        throw new NotImplementedException($"Storage {backendType} not found.");
                }
            }

            return cache[backendType];
        }

        public static IStorage GetStorage(Type dtype, StorageType storage = StorageType.TypedArray)
        {
            switch (storage)
            {
                case StorageType.Array:
                    return new ArrayStorage(dtype);
                case StorageType.TypedArray:
                    return new TypedArrayStorage(dtype);
                default:
                    return new TypedArrayStorage(dtype);
            }
        }
    }
}
