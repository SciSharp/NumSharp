using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Type conversion operators for NDArray.
    ///
    ///     NumPy alignment (see OPERATOR_ALIGNMENT.md Section 7):
    ///     - scalar → NDArray: IMPLICIT (always safe, no data loss)
    ///     - NDArray → scalar: EXPLICIT (may fail if ndim != 0, matches NumPy's int(arr) pattern)
    ///
    ///     Complex source guard:
    ///     - Any explicit NDArray → non-complex scalar cast from a Complex-typed array throws
    ///       <see cref="TypeError"/>. This matches Python's <c>int(complex)</c>/<c>float(complex)</c>
    ///       TypeError and treats NumPy's ComplexWarning (silent imaginary drop) as a hard error,
    ///       since NumSharp has no warning mechanism. Use <c>np.real</c> explicitly before casting.
    /// </summary>
    public partial class NDArray
    {
        // ===== scalar → NDArray: IMPLICIT (safe, creates 0-d array) =====

        public static implicit operator NDArray(bool d) => NDArray.Scalar<bool>(d);
        public static implicit operator NDArray(sbyte d) => NDArray.Scalar<sbyte>(d);
        public static implicit operator NDArray(byte d) => NDArray.Scalar<byte>(d);
        public static implicit operator NDArray(short d) => NDArray.Scalar<short>(d);
        public static implicit operator NDArray(ushort d) => NDArray.Scalar<ushort>(d);
        public static implicit operator NDArray(int d) => NDArray.Scalar<int>(d);
        public static implicit operator NDArray(uint d) => NDArray.Scalar<uint>(d);
        public static implicit operator NDArray(long d) => NDArray.Scalar<long>(d);
        public static implicit operator NDArray(ulong d) => NDArray.Scalar<ulong>(d);
        public static implicit operator NDArray(char d) => NDArray.Scalar<char>(d);
        public static implicit operator NDArray(Half d) => NDArray.Scalar<Half>(d);
        public static implicit operator NDArray(float d) => NDArray.Scalar<float>(d);
        public static implicit operator NDArray(double d) => NDArray.Scalar<double>(d);
        public static implicit operator NDArray(decimal d) => NDArray.Scalar<decimal>(d);
        public static implicit operator NDArray(Complex d) => NDArray.Scalar<Complex>(d);

        // ===== NDArray → scalar: EXPLICIT (requires 0-d, matches NumPy's int(arr)) =====

        /// <summary>
        ///     Validates preconditions common to every NDArray → scalar cast:
        ///     <list type="bullet">
        ///       <item><c>ndim == 0</c> (only 0-d arrays can be converted to Python scalars, per NumPy 2.x).</item>
        ///       <item>If target is non-complex, source must not be complex (TypeError, per Python's
        ///             <c>int(complex)</c>/<c>float(complex)</c> semantics).</item>
        ///       <item>A string source: complex is NumPy's cast refusal, every other target parses or tests the text
        ///             (the next phase of the string support).</item>
        ///     </list>
        /// </summary>
        /// <param name="nd">The array being converted.</param>
        /// <param name="targetType">The C# target type's name, quoted in the messages.</param>
        /// <param name="targetIsComplex">True for the <see cref="Complex"/> conversion.</param>
        /// <exception cref="IncorrectShapeException"><paramref name="nd"/> is not 0-d.</exception>
        /// <exception cref="TypeError">A complex source to a non-complex target, or a string source to complex (NumPy's
        ///     <c>Unable to convert dtype('&lt;U2') to complex</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string source to any other target: NumPy parses the
        ///     text (int, float) or tests it for emptiness (bool).</exception>
        private static void EnsureCastableToScalar(NDArray nd, string targetType, bool targetIsComplex)
        {
            if (nd.ndim != 0)
                throw new IncorrectShapeException("only 0-d arrays can be converted to scalar");
            if (!targetIsComplex && nd.typecode == NPTypeCode.Complex)
                throw new TypeError($"can't convert complex to {targetType}");
            if (nd.typecode == NPTypeCode.String)
            {
                // complex(arr) never reads the text: NumPy's array_complex first asks whether the descriptor casts to
                // complex128 under same_kind, and a string does not (TypeError "Unable to convert dtype('<U2') to
                // complex"), before any parsing.
                if (targetIsComplex)
                    throw new TypeError($"Unable to convert {nd.dtype.ToString(true)} to complex");
                // NumPy's int(arr) / float(arr) parse a 0-d string array's text and bool(arr) tests it for emptiness:
                // text computation, the next phase of the string support. (The explicit string conversion prints the
                // text.)
                throw Backends.Strings.NDStringUfuncs.NextPhase($"Converting to {targetType}", nd);
            }
        }

        public static explicit operator bool(NDArray nd)
        {
            EnsureCastableToScalar(nd, "bool", targetIsComplex: false);
            return Converts.ChangeType<bool>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator sbyte(NDArray nd)
        {
            EnsureCastableToScalar(nd, "sbyte", targetIsComplex: false);
            return Converts.ChangeType<sbyte>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator byte(NDArray nd)
        {
            EnsureCastableToScalar(nd, "byte", targetIsComplex: false);
            return Converts.ChangeType<byte>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator short(NDArray nd)
        {
            EnsureCastableToScalar(nd, "short", targetIsComplex: false);
            return Converts.ChangeType<short>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator ushort(NDArray nd)
        {
            EnsureCastableToScalar(nd, "ushort", targetIsComplex: false);
            return Converts.ChangeType<ushort>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator int(NDArray nd)
        {
            EnsureCastableToScalar(nd, "int", targetIsComplex: false);
            return Converts.ChangeType<int>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator uint(NDArray nd)
        {
            EnsureCastableToScalar(nd, "uint", targetIsComplex: false);
            return Converts.ChangeType<uint>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator long(NDArray nd)
        {
            EnsureCastableToScalar(nd, "long", targetIsComplex: false);
            return Converts.ChangeType<long>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator ulong(NDArray nd)
        {
            EnsureCastableToScalar(nd, "ulong", targetIsComplex: false);
            return Converts.ChangeType<ulong>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator char(NDArray nd)
        {
            EnsureCastableToScalar(nd, "char", targetIsComplex: false);
            return Converts.ChangeType<char>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator float(NDArray nd)
        {
            EnsureCastableToScalar(nd, "float", targetIsComplex: false);
            return Converts.ChangeType<float>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator double(NDArray nd)
        {
            EnsureCastableToScalar(nd, "double", targetIsComplex: false);
            return Converts.ChangeType<double>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator decimal(NDArray nd)
        {
            EnsureCastableToScalar(nd, "decimal", targetIsComplex: false);
            return Converts.ChangeType<decimal>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator Half(NDArray nd)
        {
            EnsureCastableToScalar(nd, "half", targetIsComplex: false);
            return Converts.ChangeType<Half>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator Complex(NDArray nd)
        {
            // Complex target: no source-type restriction. ndim==0 still required.
            EnsureCastableToScalar(nd, "complex", targetIsComplex: true);
            return Converts.ChangeType<Complex>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator string(NDArray d) => d.ToString(false);
    }
}
