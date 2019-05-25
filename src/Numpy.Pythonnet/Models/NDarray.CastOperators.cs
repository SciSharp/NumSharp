using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Numpy
{
    public partial class NDarray
    {
        public static implicit operator NDarray(Array array)
        {
            if (array == null)
                return null;
            switch (array)
            {
                case byte[] a: return np.array(a);
                case bool[] a: return np.array(a);
                case short[] a: return np.array(a);
                case int[] a: return np.array(a);
                case long[] a: return np.array(a);
                case float[] a: return np.array(a);
                case double[] a: return np.array(a);
                case byte[,] a: return np.array(a);
                case bool[,] a: return np.array(a);
                case short[,] a: return np.array(a);
                case int[,] a: return np.array(a);
                case long[,] a: return np.array(a);
                case float[,] a: return np.array(a);
                case double[,] a: return np.array(a);
                case byte[,,] a: return np.array(a);
                case bool[,,] a: return np.array(a);
                case short[,,] a: return np.array(a);
                case int[,,] a: return np.array(a);
                case long[,,] a: return np.array(a);
                case float[,,] a: return np.array(a);
                case double[,,] a: return np.array(a);
            }
            throw new InvalidOperationException($"Unable to cast {array.GetType()} to NDarray");
        }

        // these must be explicit or we have bad side effects
        public static explicit operator NDarray(bool d) => np.asarray(d);
        public static explicit operator NDarray(byte d) => np.asarray(d);
        public static explicit operator NDarray(short d) => np.asarray(d);
        public static explicit operator NDarray(int d) => np.asarray(d);
        public static explicit operator NDarray(long d) => np.asarray(d);
        public static explicit operator NDarray(float d) => np.asarray(d);
        public static explicit operator NDarray(double d) => np.asarray(d);

        // these must be explicit or we have bad side effects
        public static explicit operator bool(NDarray a) => np.asscalar<bool>(a);
        public static explicit operator byte(NDarray a) => np.asscalar<byte>(a);
        public static explicit operator short(NDarray a) => np.asscalar<short>(a);
        public static explicit operator int(NDarray a) => np.asscalar<int>(a);
        public static explicit operator long(NDarray a) => np.asscalar<long>(a);
        public static explicit operator float(NDarray a) => np.asscalar<float>(a);
        public static explicit operator double(NDarray a) => np.asscalar<double>(a);



    }
}
