using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp.Memory.Pooling
{
    public class ScalarMemoryPool
    {
        public static readonly StackedMemoryPool Instance = new StackedMemoryPool(131072);
    }
}
