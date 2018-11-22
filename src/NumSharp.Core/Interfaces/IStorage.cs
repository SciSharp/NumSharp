using System;
using System.Collections;

namespace NumSharp.Core
{
    public interface IStorage : IEnumerable, IEnumerator
    {
        Type dtype {get;set;}
        Array GetData();
        T[] GetData<T>(); 
        T GetData<T>(params int[] indexer);
        void SetData(Array value);
        void SetData<T>(T[] value);
        void SetData<T>(T value, params int[] indexer);
    }
}