using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        public static ITensorEngine GetEngine(BackendType backendType = BackendType.Default)
        {
            switch (backendType)
            {
                case BackendType.Default:
                    return new DefaultEngine();
                case BackendType.SIMD:
                    return new SimdEngine();
                case BackendType.ArrayFire:
                    return new ArrayFireEngine();
            }

            throw new NotImplementedException($"Storage {backendType} not found.");
        }
    }
}
