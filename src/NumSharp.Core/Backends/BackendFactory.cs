using NumSharp.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        public static IStorage GetEngine<T>() 
            where T: IStorage
        {
            switch (typeof(T).Name)
            {
                case "VectorTEngine":
                    return new VectorT.VectorTEngine();
                case "ManagedArrayEngine":
                    return new VectorT.VectorTEngine();
            }

            throw new NotImplementedException($"Storage {typeof(T).Name} not found.");
        }

        public static IStorage GetEngine(BackendType backendType)
        {
            if (backendType == 0)
                backendType = BackendType.ManagedArray;

            switch (backendType)
            {
                case BackendType.ManagedArray:
                    return new ManagedArray.ManagedArrayEngine();
                case BackendType.VectorT:
                    return new VectorT.VectorTEngine();
            }

            throw new NotImplementedException($"Storage {backendType} not found.");
        }
    }
}
