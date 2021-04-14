namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray x, NDArray y) => np.add(x, y);
        public static NDArray operator -(NDArray x, NDArray y) => np.subtract(x, y);
        public static NDArray operator *(NDArray x, NDArray y) => np.multiply(x, y);
        public static NDArray operator /(NDArray x, NDArray y) => np.divide(x, y);
        public static NDArray operator %(NDArray x, NDArray y) => np.mod(x, y);
        public static NDArray operator -(NDArray x) => np.negative(x);
        public static NDArray operator +(NDArray x) => x.copy(); //to maintain immutability like numpy does.
    }
}
