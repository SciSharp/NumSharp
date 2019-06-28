using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        private static Dictionary<BackendType, ITensorEngine> cache
            = new Dictionary<BackendType, ITensorEngine>();

        [DebuggerNonUserCode]
        public static ITensorEngine GetEngine(BackendType backendType = BackendType.Default)
        {
            if (!cache.ContainsKey(backendType))
            {
                switch (backendType)
                {
                    case BackendType.Default:
                        return cache[backendType] = new DefaultEngine();
                    case BackendType.MKL:
                    case BackendType.SIMD:
                        return cache[backendType] = new SimdEngine();
                    case BackendType.ArrayFire:
                        return cache[backendType] = new ArrayFireEngine();
                    default:
                        throw new NotImplementedException($"Storage {backendType} not found.");
                }
            }

            return cache[backendType];
        }
    }
}
