using System.Collections;

namespace NumSharp.Backends.Unmanaged
{
    public interface IUnmanagedArray : ICollection
    {
        unsafe void* Address { get; }
        int ItemLength { get; }
    }
}
