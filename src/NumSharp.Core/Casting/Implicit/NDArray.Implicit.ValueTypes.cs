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
        //
        // The integer primitives, float, double and Complex are NumPy's Python int / float / complex (the C# boundary
        // map), so their 0-d array is marked WasPythonLiteral (NumPy's NPY_ARRAY_WAS_PYTHON_LITERAL): a store packs it
        // with NumPy's range checks (a[0] = 1000 on an int8 array raises) where an ndarray value is cast. bool is not
        // marked (NumPy marks only an exact int / float / complex, and a bool stores like its cast). Half / char are NumPy
        // scalars in the map: their 0-d array is marked WasNumPyScalar (range-checked into a SIGNED integer, cast into the
        // rest). decimal (no NumPy analog) is a plain 0-d array.

        /// <summary>A 0-d bool array (not a Python-literal mark: NumPy stores a bool like its cast).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(bool d) => NDArray.Scalar<bool>(d);
        /// <summary>A 0-d int8 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(sbyte d) => NDArray.PythonLiteral<sbyte>(d);
        /// <summary>A 0-d uint8 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(byte d) => NDArray.PythonLiteral<byte>(d);
        /// <summary>A 0-d int16 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(short d) => NDArray.PythonLiteral<short>(d);
        /// <summary>A 0-d uint16 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(ushort d) => NDArray.PythonLiteral<ushort>(d);
        /// <summary>A 0-d int32 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(int d) => NDArray.PythonLiteral<int>(d);
        /// <summary>A 0-d uint32 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(uint d) => NDArray.PythonLiteral<uint>(d);
        /// <summary>A 0-d int64 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(long d) => NDArray.PythonLiteral<long>(d);
        /// <summary>A 0-d uint64 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(ulong d) => NDArray.PythonLiteral<ulong>(d);
        /// <summary>A 0-d char array marked as a NumPy uint16 scalar (see <see cref="WasNumPyScalar"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(char d) => NDArray.NumPyScalarLiteral<char>(d);
        /// <summary>A 0-d float16 array marked as a NumPy float16 scalar (see <see cref="WasNumPyScalar"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(Half d) => NDArray.NumPyScalarLiteral<Half>(d);
        /// <summary>A 0-d float32 array marked as a Python float literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(float d) => NDArray.PythonLiteral<float>(d);
        /// <summary>A 0-d float64 array marked as a Python float literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(double d) => NDArray.PythonLiteral<double>(d);
        /// <summary>A 0-d decimal array (NumSharp-only, a NumPy-scalar-like value, not a Python-literal mark).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(decimal d) => NDArray.Scalar<decimal>(d);
        /// <summary>A 0-d complex128 array marked as a Python complex literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(Complex d) => NDArray.PythonLiteral<Complex>(d);

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
