using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Type conversion operators for NDArray.
    ///
    ///     NumPy alignment (see OPERATOR_ALIGNMENT.md Section 7):
    ///     - scalar → NDArray: IMPLICIT (always safe, no data loss)
    ///     - NDArray → scalar: EXPLICIT (may fail if ndim != 0, matches NumPy's int(arr) pattern)
    /// </summary>
    public partial class NDArray
    {
        // ===== scalar → NDArray: IMPLICIT (safe, creates 0-d array) =====

        public static implicit operator NDArray(bool d) => NDArray.Scalar<bool>(d);
        public static implicit operator NDArray(byte d) => NDArray.Scalar<byte>(d);
        public static implicit operator NDArray(short d) => NDArray.Scalar<short>(d);
        public static implicit operator NDArray(ushort d) => NDArray.Scalar<ushort>(d);
        public static implicit operator NDArray(int d) => NDArray.Scalar<int>(d);
        public static implicit operator NDArray(uint d) => NDArray.Scalar<uint>(d);
        public static implicit operator NDArray(long d) => NDArray.Scalar<long>(d);
        public static implicit operator NDArray(ulong d) => NDArray.Scalar<ulong>(d);
        public static implicit operator NDArray(char d) => NDArray.Scalar<char>(d);
        public static implicit operator NDArray(float d) => NDArray.Scalar<float>(d);
        public static implicit operator NDArray(double d) => NDArray.Scalar<double>(d);
        public static implicit operator NDArray(decimal d) => NDArray.Scalar<decimal>(d);
        public static implicit operator NDArray(Complex d) => NDArray.Scalar<Complex>(d);

        // ===== NDArray → scalar: EXPLICIT (requires 0-d, matches NumPy's int(arr)) =====

        public static explicit operator bool(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<bool>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator byte(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<byte>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator short(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<short>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator ushort(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<ushort>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator int(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<int>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator uint(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<uint>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator long(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<long>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator ulong(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<ulong>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator char(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<char>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator float(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<float>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator double(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<double>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator decimal(NDArray nd)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            return Converts.ChangeType<decimal>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator string(NDArray d) => d.ToString(false);
    }
}
