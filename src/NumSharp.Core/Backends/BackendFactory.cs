using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        public static IStorage GetEngine<T>(BackendType backendType)
        {
            //if (backendType == 0)
                backendType = BackendType.ManagedArray;

            switch (backendType)
            {
                case BackendType.ManagedArray:
                    return new ManagedArray.ManagedArrayEngine();
                case BackendType.VectorT:
                    return new VectorT.VectorTEngine<int>();
            }

            throw new NotImplementedException($"Storage {backendType} not found.");
        }
    }
}
