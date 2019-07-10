using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp.Memory.Pooling
{
    public class ScalarMemoryPool
    {
        public static readonly InternalBufferManager.PooledBufferManager Instance = new InternalBufferManager.PooledBufferManager(131072, Marshal.SizeOf<Complex>());
    }
}
