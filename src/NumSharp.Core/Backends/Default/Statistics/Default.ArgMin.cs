namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMin(NDArray a, int axis, bool keepdims = false)
        {
            return ReduceArgMin(a, axis, keepdims);
        }

        public override NDArray ArgMin(NDArray a)
        {
            return ReduceArgMin(a, null);
        }
    }
}
