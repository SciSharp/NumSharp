namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ArgMin(in NDArray a, int axis)
        {
            return ReduceArgMin(a, axis);
        }        
        
        public override NDArray ArgMin(in NDArray a)
        {
            return ReduceArgMin(a, null);
        }
    }
}
