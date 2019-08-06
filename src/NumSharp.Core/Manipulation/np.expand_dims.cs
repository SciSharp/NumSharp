namespace NumSharp
{
    public static partial class np
    {
        public static NDArray expand_dims(NDArray a, int axis)
        {
            //test if the ndarray is empty.
            if (a.size == 0 || a.Shape.IsEmpty)
                return a;

            return new NDArray(a.Storage.Alias(a.Shape.ExpandDimension(axis)));
        }
    }
}
