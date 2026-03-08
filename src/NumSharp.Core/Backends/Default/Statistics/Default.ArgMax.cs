namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMax(in NDArray a, int axis, bool keepdims = false)
        {
            return ReduceArgMax(a, axis, keepdims);
        }

        public override NDArray ArgMax(in NDArray a)
        {
            return ReduceArgMax(a, null);
        }
    }
}
