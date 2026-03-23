namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMax(NDArray a, int axis, bool keepdims = false)
        {
            return ReduceArgMax(a, axis, keepdims);
        }

        public override NDArray ArgMax(NDArray a)
        {
            return ReduceArgMax(a, null);
        }
    }
}
