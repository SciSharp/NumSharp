namespace NumSharp
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray x, NDArray y) => x.TensorEngine.Add(x, y);
        public static NDArray operator -(NDArray x, NDArray y) => x.TensorEngine.Subtract(x, y);
        public static NDArray operator *(NDArray x, NDArray y) => x.TensorEngine.Multiply(x, y);
        public static NDArray operator /(NDArray x, NDArray y) => x.TensorEngine.Divide(x, y);
        public static NDArray operator %(NDArray x, NDArray y) => x.TensorEngine.Mod(x, y);
        public static NDArray operator -(NDArray x) => x.TensorEngine.Negate(x);
        public static NDArray operator +(NDArray x) => x.copy(); //to maintain immutability like numpy does.


#if _REGEN1
    %mod = "%"
	%foreach supported_dtypes, supported_dtypes_lowercase%

        //#2
        public static NDArray operator +(#2 left, NDArray right) => np.add(#(True|"Scalar(left)"|"(NDArray)left"), right);
        public static NDArray operator +(NDArray left, #2 right) => np.add(left, #(True|"Scalar(right)"|"(NDArray)right"));
        public static NDArray operator -(#2 left, NDArray right) => np.subtract(#(True|"Scalar(left)"|"(NDArray)left"), right);
        public static NDArray operator -(NDArray left, #2 right) => np.subtract(left, #(True|"Scalar(right)"|"(NDArray)right"));
        public static NDArray operator *(#2 left, NDArray right) => np.multiply(#(True|"Scalar(left)"|"(NDArray)left"), right);
        public static NDArray operator *(NDArray left, #2 right) => np.multiply(left, #(True|"Scalar(right)"|"(NDArray)right"));
        public static NDArray operator /(#2 left, NDArray right) => np.divide(#(True|"Scalar(left)"|"(NDArray)left"), right);
        public static NDArray operator /(NDArray left, #2 right) => np.divide(left, #(True|"Scalar(right)"|"(NDArray)right"));
        public static NDArray operator #(mod)(#2 left, NDArray right) => np.mod(#(True|"Scalar(left)"|"(NDArray)left"), right);
        public static NDArray operator #(mod)(NDArray left, #2 right) => np.mod(left, #(True|"Scalar(right)"|"(NDArray)right"));
    %
#else

        //bool
        public static NDArray operator +(bool left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, bool right) => np.add(left, Scalar(right));
        public static NDArray operator -(bool left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, bool right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(bool left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, bool right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(bool left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, bool right) => np.divide(left, Scalar(right));
        public static NDArray operator %(bool left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, bool right) => np.mod(left, Scalar(right));

        //byte
        public static NDArray operator +(byte left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, byte right) => np.add(left, Scalar(right));
        public static NDArray operator -(byte left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, byte right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(byte left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, byte right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(byte left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, byte right) => np.divide(left, Scalar(right));
        public static NDArray operator %(byte left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, byte right) => np.mod(left, Scalar(right));

        //int
        public static NDArray operator +(int left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, int right) => np.add(left, Scalar(right));
        public static NDArray operator -(int left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, int right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(int left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, int right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(int left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, int right) => np.divide(left, Scalar(right));
        public static NDArray operator %(int left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, int right) => np.mod(left, Scalar(right));

        //long
        public static NDArray operator +(long left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, long right) => np.add(left, Scalar(right));
        public static NDArray operator -(long left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, long right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(long left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, long right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(long left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, long right) => np.divide(left, Scalar(right));
        public static NDArray operator %(long left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, long right) => np.mod(left, Scalar(right));

        //float
        public static NDArray operator +(float left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, float right) => np.add(left, Scalar(right));
        public static NDArray operator -(float left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, float right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(float left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, float right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(float left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, float right) => np.divide(left, Scalar(right));
        public static NDArray operator %(float left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, float right) => np.mod(left, Scalar(right));

        //double
        public static NDArray operator +(double left, NDArray right) => np.add(Scalar(left), right);
        public static NDArray operator +(NDArray left, double right) => np.add(left, Scalar(right));
        public static NDArray operator -(double left, NDArray right) => np.subtract(Scalar(left), right);
        public static NDArray operator -(NDArray left, double right) => np.subtract(left, Scalar(right));
        public static NDArray operator *(double left, NDArray right) => np.multiply(Scalar(left), right);
        public static NDArray operator *(NDArray left, double right) => np.multiply(left, Scalar(right));
        public static NDArray operator /(double left, NDArray right) => np.divide(Scalar(left), right);
        public static NDArray operator /(NDArray left, double right) => np.divide(left, Scalar(right));
        public static NDArray operator %(double left, NDArray right) => np.mod(Scalar(left), right);
        public static NDArray operator %(NDArray left, double right) => np.mod(left, Scalar(right));
#endif




    }
}
