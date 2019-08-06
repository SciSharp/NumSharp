namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMax(in NDArray a, int axis)
        {
            return ReduceArgMax(a, axis);
        }        
        
        public override NDArray ArgMax(in NDArray a)
        {
            return ReduceArgMax(a, null);
        }
    }
}
