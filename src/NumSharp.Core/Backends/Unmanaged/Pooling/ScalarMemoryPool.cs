namespace NumSharp.Unmanaged.Memory
{
    public class ScalarMemoryPool
    {
        public static readonly StackedMemoryPool Instance = new StackedMemoryPool(131072);
    }
}
