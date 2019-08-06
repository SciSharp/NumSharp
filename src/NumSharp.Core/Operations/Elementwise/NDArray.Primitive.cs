namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray left, NDArray right) => np.add(left, right);
        public static NDArray operator -(NDArray x, NDArray y) => np.subtract(x, y);
        public static NDArray operator *(NDArray x, NDArray y) => np.multiply(x, y);
        public static NDArray operator /(NDArray x, NDArray y) => np.divide(x, y);
        public static NDArray operator -(NDArray x) => x.TensorEngine.Negate(x); //access engine directly since there is no np.negate(x)

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        /// <remarks>
        ///     a = np.arange(10)
        ///     print(a is a)
        ///     print(a is (+a))
        /// </remarks>
        public static NDArray operator +(NDArray x) => x.copy(); //to maintain immutability.

        //byte
        public static NDArray operator +(byte left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, byte right) => np.add(left, Scalar(right));
        public static NDArray operator -(byte left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, byte right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(byte left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, byte right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(byte left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, byte right) => np.divide(left, Scalar(right));


#if _REGEN
	%foreach except(supported_dtypes,"Boolean", "Byte"), except(supported_dtypes_lowercase, "bool", "byte")%
        //#2
        public static NDArray operator +(#2 left, NDArray right) => np.add((NDArray)left, right);
        public static NDArray operator +(NDArray left, #2 right) => np.add(left, (NDArray) right);
        public static NDArray operator -(#2 left, NDArray right) => np.subtract((NDArray)left, right);
        public static NDArray operator -(NDArray left, #2 right) => np.subtract(left, (NDArray) right);
        public static NDArray operator *(#2 left, NDArray right) => np.multiply((NDArray)left, right);
        public static NDArray operator *(NDArray left, #2 right) => np.multiply(left, (NDArray) right);
        public static NDArray operator /(#2 left, NDArray right) => np.divide((NDArray)left, right);
        public static NDArray operator /(NDArray left, #2 right) => np.divide(left, (NDArray) right);

    %
#else

        //short
        public static NDArray operator +(short left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, short right) => np.add(left, right);
        public static NDArray operator -(short left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, short right) => np.subtract(left, right);
        public static NDArray operator *(short left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, short right) => np.multiply(left, right);
        public static NDArray operator /(short left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, short right) => np.divide(left, right);

        //ushort
        public static NDArray operator +(ushort left, NDArray right) => np.add((NDArray)left, right);
        public static NDArray operator +(NDArray left, ushort right) => np.add(left, (NDArray) right);
        public static NDArray operator -(ushort left, NDArray right) => np.subtract((NDArray)left, right);
        public static NDArray operator -(NDArray left, ushort right) => np.subtract(left, (NDArray) right);
        public static NDArray operator *(ushort left, NDArray right) => np.multiply((NDArray)left, right);
        public static NDArray operator *(NDArray left, ushort right) => np.multiply(left, (NDArray) right);
        public static NDArray operator /(ushort left, NDArray right) => np.divide((NDArray)left, right);
        public static NDArray operator /(NDArray left, ushort right) => np.divide(left, (NDArray) right);

        //int
        public static NDArray operator +(int left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, int right) => np.add(left, right);
        public static NDArray operator -(int left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, int right) => np.subtract(left, right);
        public static NDArray operator *(int left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, int right) => np.multiply(left, right);
        public static NDArray operator /(int left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, int right) => np.divide(left, right);

        //uint
        public static NDArray operator +(uint left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, uint right) => np.add(left, right);
        public static NDArray operator -(uint left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, uint right) => np.subtract(left, right);
        public static NDArray operator *(uint left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, uint right) => np.multiply(left, right);
        public static NDArray operator /(uint left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, uint right) => np.divide(left, right);

        //long
        public static NDArray operator +(long left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, long right) => np.add(left, right);
        public static NDArray operator -(long left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, long right) => np.subtract(left, right);
        public static NDArray operator *(long left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, long right) => np.multiply(left, right);
        public static NDArray operator /(long left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, long right) => np.divide(left, right);

        //ulong
        public static NDArray operator +(ulong left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, ulong right) => np.add(left, right);
        public static NDArray operator -(ulong left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, ulong right) => np.subtract(left, right);
        public static NDArray operator *(ulong left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, ulong right) => np.multiply(left, right);
        public static NDArray operator /(ulong left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, ulong right) => np.divide(left, right);

        //char
        public static NDArray operator +(char left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, char right) => np.add(left, right);
        public static NDArray operator -(char left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, char right) => np.subtract(left, right);
        public static NDArray operator *(char left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, char right) => np.multiply(left, right);
        public static NDArray operator /(char left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, char right) => np.divide(left, right);

        //double
        public static NDArray operator +(double left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, double right) => np.add(left, right);
        public static NDArray operator -(double left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, double right) => np.subtract(left, right);
        public static NDArray operator *(double left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, double right) => np.multiply(left, right);
        public static NDArray operator /(double left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, double right) => np.divide(left, right);

        //float
        public static NDArray operator +(float left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, float right) => np.add(left, right);
        public static NDArray operator -(float left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, float right) => np.subtract(left, right);
        public static NDArray operator *(float left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, float right) => np.multiply(left, right);
        public static NDArray operator /(float left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, float right) => np.divide(left, right);

        //decimal
        public static NDArray operator +(decimal left, NDArray right) => np.add(left, right);
        public static NDArray operator +(NDArray left, decimal right) => np.add(left, right);
        public static NDArray operator -(decimal left, NDArray right) => np.subtract(left, right);
        public static NDArray operator -(NDArray left, decimal right) => np.subtract(left, right);
        public static NDArray operator *(decimal left, NDArray right) => np.multiply(left, right);
        public static NDArray operator *(NDArray left, decimal right) => np.multiply(left, right);
        public static NDArray operator /(decimal left, NDArray right) => np.divide(left, right);
        public static NDArray operator /(NDArray left, decimal right) => np.divide(left, right);
#endif




    }
}
