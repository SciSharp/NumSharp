namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMin(in NDArray a, int axis, bool keepdims = false)
        {
            return ReduceArgMin(a, axis, keepdims);
        }

        public override NDArray ArgMin(in NDArray a)
        {
            return ReduceArgMin(a, null);
        }
    }
}
