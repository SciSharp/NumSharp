namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray transpose()
            => np.transpose(this);
    }
}
