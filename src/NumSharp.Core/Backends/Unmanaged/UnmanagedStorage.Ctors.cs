using System;
using System.Globalization;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        // ---------------------------------------------------------------------------------------------
        // ctors.c (np-layer dtype plan, Appendix D's UnmanagedStorage.Ctors.cs): PyArray_ArangeObj. np.arange's overloads
        // hand it Python ints (long) or Python floats (double); the length comes from Python's arithmetic on them, and the
        // array is filled by its dtype's slots - start and start + step through setitem, the rest through fill - as NumPy
        // fills it. No ufunc runs, so no TensorEngine member is involved.
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The ceiling of <c>(double)NPY_MAX_INTP</c>: <c>2**63</c>, the value the range check of
        ///     <c>_arange_safe_ceil_to_intp</c> rounds <c>NPY_MAX_INTP</c> to.
        /// </summary>
        private const double ArangeMaxIntpAsDouble = 9223372036854775808.0;

        /// <summary>NumPy's text when the length does not fit npy_intp (the OverflowError NumPy re-raises as a ValueError).</summary>
        private const string ArangeMaxSize = "Maximum allowed size exceeded";

        /// <summary>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>) for Python-float bounds: the length from Python's float
        ///     arithmetic (<c>_calc_length</c>), then the array filled by its dtype's <c>setitem</c> and <c>fill</c> slots.
        /// </summary>
        /// <param name="start">The start, a Python float.</param>
        /// <param name="stop">The stop (excluded), a Python float.</param>
        /// <param name="step">The step, a Python float.</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's discovery from Python floats: float64.</param>
        /// <returns>A fresh 1-D array of the dtype (its native form): <c>start</c>, <c>start + step</c> stored as Python
        ///     floats, then <c>start + i * delta</c> in the dtype's own arithmetic, with <c>delta</c> the difference of the
        ///     first two stored elements.</returns>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet (datetime64 and timedelta64,
        ///     which NumPy hands to <c>datetime_arange</c>).</exception>
        /// <exception cref="TypeError">The dtype has no fill function (<c>S</c>, <c>U</c>, <c>StringDType</c>: <c>arange() not
        ///     supported for inputs with DType &lt;class 'numpy.dtypes.StrDType'&gt;.</c>), or it is bool and the result has
        ///     more than two elements (NumPy's <c>BOOL_fill</c>).</exception>
        /// <exception cref="ZeroDivisionError">A zero step (NumPy's <c>ZeroDivisionError: float division by zero</c>).</exception>
        /// <exception cref="ValueError">A NaN length (<c>arange: cannot compute length</c>), a length past npy_intp
        ///     (<c>Maximum allowed size exceeded</c>), or an allocation past the byte limit (<see cref="AllocationGuard"/>'s
        ///     <c>array is too big</c>); a store's own ValueError (a NaN into an integer dtype).</exception>
        /// <exception cref="OverflowError">The first or second element does not fit the dtype (NumPy's
        ///     <c>Python integer 130 out of bounds for int8</c>, an infinity into an integer dtype).</exception>
        internal static NDArray ArangeObj(double start, double stop, double step, DType dtype)
        {
            var descr = ArangeDescr(dtype ?? NumSharp.DType.Double);
            long length = ArangeLength(start, stop, step);
            // _calc_length forms next = start + step only for a positive length; it is stored only for two elements or more.
            return ArangeFill(descr, length, start, length > 1 ? start + step : 0.0);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>) for Python-int bounds: the length from Python's exact integer
        ///     arithmetic and correctly rounded true division (<c>_calc_length</c>), then the array filled by its dtype's
        ///     <c>setitem</c> and <c>fill</c> slots.
        /// </summary>
        /// <param name="start">The start, a Python int.</param>
        /// <param name="stop">The stop (excluded), a Python int.</param>
        /// <param name="step">The step, a Python int.</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's discovery from Python ints within int64: int64.</param>
        /// <returns>A fresh 1-D array of the dtype (its native form): <c>start</c>, <c>start + step</c> stored as Python
        ///     ints, then <c>start + i * delta</c> in the dtype's own arithmetic.</returns>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet.</exception>
        /// <exception cref="TypeError">The dtype has no fill function, or it is bool and the result has more than two
        ///     elements.</exception>
        /// <exception cref="ZeroDivisionError">A zero step (NumPy's <c>ZeroDivisionError: division by zero</c>).</exception>
        /// <exception cref="ValueError">A length past npy_intp (<c>Maximum allowed size exceeded</c>), or an allocation past
        ///     the byte limit (<c>array is too big</c>).</exception>
        /// <exception cref="OverflowError">The first or second element does not fit the dtype.</exception>
        internal static NDArray ArangeObj(long start, long stop, long step, DType dtype)
        {
            var descr = ArangeDescr(dtype ?? NumSharp.DType.Int64);
            long length = ArangeLength((BigInteger)start, stop, step);
            // A Python int's sum is exact; with two elements or more it lies between start and stop, so it fits a long.
            return ArangeFill(descr, length, start, length > 1 ? (object)((BigInteger)start + step) : null);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>) over Python OBJECTS - the body of np.arange's NDArray
        ///     signature: the dtype discovered as int64 promoted with each argument's own descriptor, the length from the
        ///     arguments' own arithmetic (<c>_calc_length</c>), then the array filled by its dtype's <c>setitem</c> and
        ///     <c>fill</c> slots.
        /// </summary>
        /// <param name="start">The start, after <c>array_arange</c>'s reading (not null): a C# literal stands for the Python
        ///     number it was written as (<see cref="NDArray.WasPythonLiteral"/>, <see cref="NDArray.WasPythonBool"/>), any
        ///     other array for itself (a NumPy scalar, a 0-d array, an array with dimensions).</param>
        /// <param name="stop">The stop (excluded), or null: then <paramref name="start"/> is the stop and the Python int 0 the
        ///     start (<c>np.arange(5)</c>).</param>
        /// <param name="step">The step, or null for the Python int 1.</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's discovery (<c>np.arange(5)</c> int64,
        ///     <c>np.arange(np.float32(3.5))</c> float64, a uint64 value float64, a complex complex128).</param>
        /// <returns>A fresh 1-D array of the dtype (its native form): <c>start</c>, <c>start + step</c>, then <c>start + i *
        ///     delta</c> in the dtype's own arithmetic.</returns>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet (datetime64,
        ///     timedelta64).</exception>
        /// <exception cref="TypeError">The dtype has no fill function (a text dtype, which a string argument also discovers:
        ///     <c>arange() not supported for inputs with DType ...</c>); bool with more than two elements; a Python complex
        ///     length into a real dtype (<c>must be real number, not complex</c>); an argument with dimensions (<c>only
        ///     0-dimensional arrays can be converted to Python scalars</c>); a NumPy operator's own refusal (<c>numpy
        ///     boolean subtract ...</c>).</exception>
        /// <exception cref="ZeroDivisionError">A zero Python step (<c>division by zero</c>, <c>float division by zero</c>,
        ///     <c>complex division by zero</c>; a NumPy scalar's zero divides to an infinity instead, the next
        ///     ValueError).</exception>
        /// <exception cref="ValueError">A NaN length (<c>arange: cannot compute length</c>); a length past npy_intp or any
        ///     OverflowError of the length's arithmetic (<c>Maximum allowed size exceeded</c>); an argument with more than one
        ///     element or none (NumPy's truth-value texts); an allocation past the byte limit; a store's ValueError.</exception>
        /// <exception cref="OverflowError">The first or second element does not fit the dtype (<c>Python integer 1000 out of
        ///     bounds for int8</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>_calc_length</c> computes <c>next = stop - start</c>, <c>next != 0</c>, <c>val = next / step</c>,
        ///     <c>val == 0</c> and <c>start + step</c> on the Python objects it was handed, so two arithmetics meet: Python
        ///     numbers run CPython's (<see cref="PyScalar"/>: exact ints, so <c>np.arange(2**64 - 6, 2**64 - 1, 2)</c> has 3
        ///     elements), and the moment a NumPy scalar or array takes part the operation is NumPy's own, NEP 50 included
        ///     (NumSharp's operators): <c>np.arange(0, 1, np.float32(0.1))</c> divides in float32, and
        ///     <c>np.arange(np.int8(-128), np.int8(127), np.int8(100))</c> wraps <c>stop - start</c> in int8 and is empty, as
        ///     in NumPy. A complex dtype takes the smaller of the ceilings of the quotient's two parts
        ///     (<c>np.arange(1+2j)</c> is <c>[0j]</c>, <c>np.arange(np.complex128(3))</c> is empty).
        ///     </para>
        ///     <para>
        ///     The operation temporaries are this call's own (<c>[NDScoped]</c>); only the result escapes. One NumSharp limit:
        ///     a Python int outside the int64 / uint64 ranges cannot become a NumPy operand, and raises the
        ///     <c>Maximum allowed size exceeded</c> NumPy raises for an integer partner (a float partner would convert it).
        ///     </para>
        /// </remarks>
        [NDScoped /* the operands of the length's arithmetic and its results are this call's own */]
        internal static NDArray ArangeObj(NDArray start, NDArray stop, NDArray step, DType dtype)
        {
            if (dtype is null)
            {
                // "intentionally made to be at least NPY_LONG": int64 promoted with each argument's own descriptor, the
                // stop's and the step's only when they were given.
                dtype = NumSharp.DType.Int64;
                dtype = ArangePromote(dtype, start);
                if (stop is not null)
                    dtype = ArangePromote(dtype, stop);
                if (step is not null)
                    dtype = ArangePromote(dtype, step);
            }
            var descr = ArangeDescr(dtype);
            // PyTypeNum_ISCOMPLEX(dtype->type_num): the length then reads both parts of a complex quotient.
            bool cmplx = descr.typecode == NPTypeCode.Complex;

            // A missing step is the Python int 1; a missing stop makes the start the stop and the Python int 0 the start.
            ArangeValue first = ArangeValue.Of(start);
            ArangeValue delta = step is null ? ArangeValue.Python(PyScalar.Int(BigInteger.One)) : ArangeValue.Of(step);
            ArangeValue last;
            if (stop is null)
            {
                last = first;
                first = ArangeValue.Python(PyScalar.Int(BigInteger.Zero));
            }
            else
            {
                last = ArangeValue.Of(stop);
            }

            long length;
            ArangeValue next = default;
            try
            {
                length = ArangeLength(first, last, delta, cmplx);
                // _calc_length forms next = start + step only for a positive length (its own OverflowError included).
                if (length > 0)
                    next = ArangeValue.Apply(BinaryOp.Add, first, delta);
            }
            catch (OverflowError)
            {
                // PyArray_ArangeObj turns any OverflowError of _calc_length into this ValueError.
                throw new ValueError(ArangeMaxSize);
            }
            return ArangeFill(descr, length, first.Element, length > 1 ? next.Element : null);
        }

        /// <summary>
        ///     <c>PyArray_DescrFromObject(op, dtype)</c>: the descriptor <c>np.asarray(op)</c> would have, promoted with the
        ///     running one (<see cref="DTypePromotion.PromoteTypes"/>, NumPy's <c>PyArray_PromoteTypes</c>).
        /// </summary>
        /// <param name="dtype">The running descriptor.</param>
        /// <param name="argument">The argument.</param>
        /// <returns>The promoted descriptor.</returns>
        /// <exception cref="DTypePromotionError">The two descriptors have no common one.</exception>
        private static DType ArangePromote(DType dtype, NDArray argument)
        {
            // A C# literal is the Python number it was written as: its array's dtype is the number's default (int64, uint64
            // for an int past the int64 range, float64, complex128), whatever the literal's C# width.
            DType discovered = argument.WasPythonLiteral ? DTypePromotion.ResultType(argument.Nep50Operand) : argument.dtype;
            return DTypePromotion.PromoteTypes(dtype, discovered);
        }

        /// <summary>
        ///     <c>_calc_length</c> over Python objects: <c>(stop - start) / step</c> in the objects' own arithmetic; a complex
        ///     quotient (with a complex dtype) gives the smaller ceiling of its two parts, else the underflow test and the
        ///     ceiling of the float the quotient converts to.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step.</param>
        /// <param name="cmplx">Whether the dtype is complex.</param>
        /// <returns>The length; zero or negative means an empty array.</returns>
        /// <exception cref="ZeroDivisionError">A zero Python step.</exception>
        /// <exception cref="OverflowError">A Python int too large for a float, or for a NumPy operand (the caller re-raises
        ///     it as <c>Maximum allowed size exceeded</c>).</exception>
        /// <exception cref="ValueError">A NaN quotient or a ceiling outside npy_intp; an argument with more than one element
        ///     or none (its truth value).</exception>
        /// <exception cref="TypeError">A Python complex quotient with a real dtype, an argument with dimensions, a NumPy
        ///     operator's refusal.</exception>
        private static long ArangeLength(in ArangeValue start, in ArangeValue stop, in ArangeValue step, bool cmplx)
        {
            ArangeValue next = ArangeValue.Apply(BinaryOp.Subtract, stop, start);
            bool nextIsNonzero = next.IsNonzero();
            ArangeValue val = ArangeValue.Apply(BinaryOp.Divide, next, step);
            bool valIsZero = val.IsZero();

            // PyComplex_Check(val): a Python complex, or a complex128 scalar (np.complex128 subclasses Python's complex).
            if (cmplx && val.IsComplexScalar)
            {
                Complex quotient = val.AsComplex();
                long len = ArangeSafeCeilToIntp(quotient.Real);
                long tmp = ArangeSafeCeilToIntp(quotient.Imaginary);
                return Math.Min(len, tmp);
            }

            double value = val.AsPyFloat();
            // NumPy's "underflow and divide-by-inf" check, as in the float form (no such check in the complex branch).
            if (valIsZero && nextIsNonzero)
                return double.IsNegative(value) ? 0 : 1;
            return ArangeSafeCeilToIntp(value);
        }

        /// <summary>
        ///     One argument of <see cref="ArangeObj(NDArray, NDArray, NDArray, DType)"/> as NumPy's <c>_calc_length</c> sees
        ///     it: a Python number (a C# literal's value, or the result of Python arithmetic), or a NumPy object (a NumPy
        ///     scalar, a 0-d array or an array with dimensions, or the result of an operation one took part in).
        /// </summary>
        private readonly struct ArangeValue
        {
            /// <summary>The NumPy object, or null for a Python number.</summary>
            private readonly NDArray _array;

            /// <summary>The Python number when <see cref="_array"/> is null.</summary>
            private readonly PyScalar _py;

            /// <summary>Records one value.</summary>
            /// <param name="array">The NumPy object, or null.</param>
            /// <param name="py">The Python number (read only when <paramref name="array"/> is null).</param>
            private ArangeValue(NDArray array, PyScalar py)
            {
                _array = array;
                _py = py;
            }

            /// <summary>A Python number.</summary>
            /// <param name="py">The number.</param>
            /// <returns>The value.</returns>
            public static ArangeValue Python(PyScalar py) => new ArangeValue(null, py);

            /// <summary>
            ///     The value an argument stands for: a literal's Python number (a C# <c>float</c> is a Python float, a
            ///     <c>ulong</c> a Python int by its value, a <c>bool</c> a Python bool), any other array itself.
            /// </summary>
            /// <param name="argument">The argument (not null).</param>
            /// <returns>The value.</returns>
            public static ArangeValue Of(NDArray argument)
            {
                if (argument.WasPythonBool)
                    return Python(PyScalar.Bool((bool)argument.Storage.GetAtIndex(0)));
                if (!argument.WasPythonLiteral)
                    return new ArangeValue(argument, default);
                // The literal's own C# value: the marks are set only on the 0-d array a number literal converted into.
                return Python(argument.Storage.GetAtIndex(0) switch
                {
                    double d => PyScalar.Float(d),
                    float f => PyScalar.Float(f),
                    Complex z => PyScalar.Cplx(z),
                    ulong u => PyScalar.Int(u),
                    object v => PyScalar.Int(Convert.ToInt64(v, CultureInfo.InvariantCulture)),
                });
            }

            /// <summary>
            ///     <c>a op b</c> the way Python evaluates it on these objects: two Python numbers run CPython's arithmetic
            ///     (<see cref="PyScalar.Apply"/>); a NumPy object on either side makes it NumPy's operation (NumSharp's
            ///     operators, NEP 50 with the Python operand weak), whose result is a NumPy object.
            /// </summary>
            /// <param name="op">Add, Subtract or Divide (true division).</param>
            /// <param name="a">The left operand.</param>
            /// <param name="b">The right operand.</param>
            /// <returns>The result.</returns>
            /// <exception cref="ZeroDivisionError">Python division by zero.</exception>
            /// <exception cref="OverflowError">CPython's int-to-float overflow; a Python int that fits no NumPy integer;
            ///     a weak Python int out of a NumPy integer operand's bounds.</exception>
            /// <exception cref="TypeError">A NumPy operator's refusal (bool subtraction).</exception>
            /// <exception cref="NotSupportedException">Another operator.</exception>
            public static ArangeValue Apply(BinaryOp op, in ArangeValue a, in ArangeValue b)
            {
                if (a._array is null && b._array is null)
                    return Python(PyScalar.Apply(op, a._py, b._py));
                NDArray x = a.AsOperand(), y = b.AsOperand();
                NDArray result = op switch
                {
                    BinaryOp.Add => x + y,
                    BinaryOp.Subtract => x - y,
                    // NDArray's '/' is NumPy's true division (int / int is float64).
                    BinaryOp.Divide => x / y,
                    _ => throw new NotSupportedException(op.ToString()),
                };
                return new ArangeValue(result, default);
            }

            /// <summary>
            ///     The operand a NumPy operation receives for this value: the NumPy object itself, or the WEAK literal array
            ///     of the Python number (NEP 50 adopts the other operand's dtype).
            /// </summary>
            /// <returns>The operand array.</returns>
            /// <exception cref="OverflowError">A Python int outside the int64 and uint64 ranges.</exception>
            private NDArray AsOperand()
            {
                if (_array is not null)
                    return _array;
                if (_py.IsBool)
                    return NDArray.PythonBoolLiteral(!_py.I.IsZero);
                switch (_py.Kind)
                {
                    case PyKind.Float:
                        return NDArray.PythonLiteral(_py.F);
                    case PyKind.Complex:
                        return NDArray.PythonLiteral(_py.C);
                    default:
                        if (_py.I >= long.MinValue && _py.I <= long.MaxValue)
                            return NDArray.PythonLiteral((long)_py.I);
                        if (_py.I.Sign > 0 && _py.I <= ulong.MaxValue)
                            return NDArray.PythonLiteral((ulong)_py.I);
                        // No NumSharp literal holds it (NumPy would convert it for a float partner): the caller's
                        // "Maximum allowed size exceeded", NumPy's answer for an integer partner.
                        throw new OverflowError("Python int too large to convert to C long");
                }
            }

            /// <summary><c>PyObject_RichCompareBool(x, 0, Py_NE)</c>: Python's <c>x != 0</c>, or the truth of NumPy's
            ///     elementwise comparison.</summary>
            /// <returns>Whether the value is nonzero (a NaN is).</returns>
            /// <exception cref="ValueError">An array of more than one element, or none (NumPy's truth-value texts).</exception>
            public bool IsNonzero()
                => _array is null ? _py.IsNonZero : (bool)(_array != NDArray.PythonLiteral(0L));

            /// <summary><c>PyObject_RichCompareBool(x, 0, Py_EQ)</c>: Python's <c>x == 0</c>, or the truth of NumPy's
            ///     elementwise comparison.</summary>
            /// <returns>Whether the value is zero (a NaN is not).</returns>
            /// <exception cref="ValueError">An array of more than one element, or none.</exception>
            public bool IsZero()
                => _array is null ? !_py.IsNonZero : (bool)(_array == NDArray.PythonLiteral(0L));

            /// <summary><c>PyComplex_Check</c>: a Python complex, or a complex128 scalar (a 0-d result; an array with
            ///     dimensions is no complex object).</summary>
            public bool IsComplexScalar
                => _array is null ? _py.Kind == PyKind.Complex : _array.ndim == 0 && _array.typecode == NPTypeCode.Complex;

            /// <summary><c>PyComplex_RealAsDouble</c> / <c>ImagAsDouble</c>: the value as a complex (only when
            ///     <see cref="IsComplexScalar"/>).</summary>
            /// <returns>The complex value.</returns>
            public Complex AsComplex()
                => _array is null ? _py.C : (Complex)_array.Storage.GetAtIndex(0);

            /// <summary>
            ///     <c>PyFloat_AsDouble</c>: a Python float itself; a NumPy scalar's <c>float()</c> (a complex128 scalar's
            ///     real part, as NumPy's complex <c>__float__</c>, whose ComplexWarning NumSharp does not raise).
            /// </summary>
            /// <returns>The double.</returns>
            /// <exception cref="TypeError">A Python complex (<c>must be real number, not complex</c>), or an array with
            ///     dimensions (<c>only 0-dimensional arrays can be converted to Python scalars</c>).</exception>
            public double AsPyFloat()
            {
                if (_array is null)
                {
                    // A quotient is a float or a complex (true division never leaves an int).
                    if (_py.Kind == PyKind.Complex)
                        throw new TypeError("must be real number, not complex");
                    return _py.AsDouble();
                }
                if (_array.ndim == 0 && _array.typecode == NPTypeCode.Complex)
                    return ((Complex)_array.Storage.GetAtIndex(0)).Real;
                // NumPy's float(arr): the 0-d test (NumPy 2.4's TypeError for an array with dimensions) and the element.
                return (double)_array;
            }

            /// <summary>The element <c>setitem</c> stores: the NumPy object itself (a 0-d array casts, as NumPy's setitem
            ///     of a 0-d array), or the Python number boxed as the store paths read it.</summary>
            public object Element
            {
                get
                {
                    if (_array is not null)
                        return _array;
                    if (_py.IsBool)
                        return !_py.I.IsZero;
                    return _py.Kind switch
                    {
                        PyKind.Float => _py.F,
                        PyKind.Complex => _py.C,
                        _ => _py.I,
                    };
                }
            }
        }

        /// <summary>
        ///     The checks <c>PyArray_ArangeObj</c> makes on the dtype before it computes anything: the datetime pair goes to
        ///     <c>datetime_arange</c> in NumPy (refused here: no storage yet), and a dtype whose class has no fill function
        ///     is a TypeError.
        /// </summary>
        /// <param name="dtype">The requested or discovered dtype.</param>
        /// <returns><paramref name="dtype"/>.</returns>
        /// <exception cref="NotSupportedException">The class has no NumSharp storage yet.</exception>
        /// <exception cref="TypeError">The class has no fill function.</exception>
        private static DType ArangeDescr(DType dtype)
        {
            if (!dtype.Meta.HasStorage)
                dtype.GetTypeCode(); // raises the descriptor layer's "no NumSharp storage yet" refusal
            // NumPy: if (!funcs->fill) "arange() not supported for inputs with DType %S." with the class object's repr.
            if (!dtype.Meta.HasFill)
                throw new TypeError($"arange() not supported for inputs with DType {dtype.Meta}.");
            return dtype;
        }

        /// <summary>
        ///     <c>_calc_length</c> over Python floats: <c>(stop - start) / step</c> in float arithmetic, an underflowed
        ///     quotient of a nonzero difference counted by its sign, else its ceiling as npy_intp.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step.</param>
        /// <returns>The length; zero or negative means an empty array.</returns>
        /// <exception cref="ZeroDivisionError">A zero step (Python's <c>float division by zero</c>).</exception>
        /// <exception cref="ValueError">A NaN quotient, or a ceiling outside npy_intp.</exception>
        private static long ArangeLength(double start, double stop, double step)
        {
            double next = stop - start;
            // Python's comparison: a NaN difference is "nonzero".
            bool nextIsNonzero = next != 0;
            double value = PyScalar.FloatDiv(next, step);
            // NumPy's "underflow and divide-by-inf" check: a zero quotient of a NONZERO difference is one element for a
            // positive zero and none for a negative one.
            if (value == 0 && nextIsNonzero)
                return double.IsNegative(value) ? 0 : 1;
            return ArangeSafeCeilToIntp(value);
        }

        /// <summary>
        ///     <c>_calc_length</c> over Python ints: the exact difference, Python's correctly rounded true division by the
        ///     step, then the same tests as the float form.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step.</param>
        /// <returns>The length; zero or negative means an empty array.</returns>
        /// <exception cref="ZeroDivisionError">A zero step (Python's <c>division by zero</c>).</exception>
        /// <exception cref="ValueError">A quotient past the float range or a ceiling outside npy_intp (NumPy re-raises both
        ///     OverflowErrors as <c>Maximum allowed size exceeded</c>).</exception>
        private static long ArangeLength(BigInteger start, BigInteger stop, BigInteger step)
        {
            BigInteger next = stop - start;
            double value;
            try
            {
                value = PyScalar.IntTrueDivide(next, step);
            }
            catch (OverflowError)
            {
                // PyArray_ArangeObj turns any OverflowError of _calc_length into this ValueError.
                throw new ValueError(ArangeMaxSize);
            }
            if (value == 0 && !next.IsZero)
                return double.IsNegative(value) ? 0 : 1;
            return ArangeSafeCeilToIntp(value);
        }

        /// <summary>
        ///     NumPy's <c>_arange_safe_ceil_to_intp</c>: the ceiling, refused when it is NaN or outside
        ///     <c>[(double)NPY_MIN_INTP, (double)NPY_MAX_INTP]</c>, then cast to npy_intp.
        /// </summary>
        /// <param name="value">The quotient <c>(stop - start) / step</c>.</param>
        /// <returns>The ceiling as a long; <see cref="long.MinValue"/> for a ceiling of exactly <c>2**63</c>.</returns>
        /// <exception cref="ValueError">A NaN (<c>arange: cannot compute length</c>), or a ceiling out of range (NumPy's
        ///     OverflowError, which <c>PyArray_ArangeObj</c> reports as <c>Maximum allowed size exceeded</c>).</exception>
        /// <remarks>
        ///     <c>(double)NPY_MAX_INTP</c> rounds to <c>2**63</c>, so a ceiling of exactly <c>2**63</c> passes the range check,
        ///     and C's <c>(npy_intp)</c> cast of it is <c>INT64_MIN</c> under x86-64's <c>cvttsd2si</c> (the "integer
        ///     indefinite" value): a negative length, which np.arange turns into an empty array (<c>np.arange(0, 2**63)</c> is
        ///     <c>[]</c> in NumPy on x86). .NET 9+ saturates that conversion instead, so the case is decided explicitly.
        /// </remarks>
        private static long ArangeSafeCeilToIntp(double value)
        {
            double ivalue = Math.Ceiling(value);
            // The condition inverted, as NumPy's, so a NaN fails it.
            if (double.IsNaN(ivalue))
                throw new ValueError("arange: cannot compute length");
            if (!(-ArangeMaxIntpAsDouble <= ivalue && ivalue <= ArangeMaxIntpAsDouble))
                throw new ValueError(ArangeMaxSize);
            return ivalue == ArangeMaxIntpAsDouble ? long.MinValue : (long)ivalue;
        }

        /// <summary>
        ///     The rest of <c>PyArray_ArangeObj</c>: allocate the result, store <paramref name="start"/> through the dtype's
        ///     <c>setitem</c>, then <paramref name="next"/>, then <c>fill</c> the remaining elements.
        /// </summary>
        /// <param name="dtype">The dtype (its class has a fill function).</param>
        /// <param name="length">The length; zero or less gives an empty array.</param>
        /// <param name="start">The first element's value (a Python float or int).</param>
        /// <param name="next">The second element's value, <c>start + step</c> (read only when the length is 2 or more).</param>
        /// <returns>The array.</returns>
        /// <exception cref="ValueError">The allocation is past the byte limit, or a store's ValueError.</exception>
        /// <exception cref="OverflowError">A store does not fit the dtype.</exception>
        /// <exception cref="TypeError">A store's TypeError, or bool's fill.</exception>
        /// <remarks>The allocation is released when a store or the fill fails: nothing else holds it yet, as NumPy's
        ///     <c>Py_DECREF(range)</c> on its failure path.</remarks>
        private static unsafe NDArray ArangeFill(DType dtype, long length, object start, object next)
        {
            if (length <= 0)
                length = 0;
            var range = new NDArray(dtype, Shape.Vector(length), false);
            if (length == 0)
                return range;
            try
            {
                // PyArray_DESCR(range): the descriptor the allocation installed (the native form of a byte-swapped request,
                // whose bytes NumSharp stores natively), and the buffer a StringDType class would need for its allocator.
                var descr = range.dtype;
                var meta = descr.Meta;
                var storage = range.Storage;
                byte* data = storage.Address;
                meta.setitem(descr, start, data, storage.InternalArray);
                if (length == 1)
                    return range;
                meta.setitem(descr, next, data + descr.itemsize, storage.InternalArray);
                if (length == 2)
                    return range;
                // HasFill was checked before the length; PyArray_Arange keeps this test for the same answer.
                if (!meta.fill(descr, data, length))
                    throw new ValueError("no fill-function for data-type.");
                return range;
            }
            catch
            {
                range.Dispose();
                throw;
            }
        }

        // ---------------------------------------------------------------------------------------------
        // The allocations behind np.full, np.full_like and np.ones (ctors.c; np-layer dtype plan, Appendix D's
        // UnmanagedStorage.Ctors.cs): a new array of a storage lane with one element stored into every slot. NumPy
        // writes it as `a = empty(...); copyto(a, value, casting='unsafe')`; the array object's NDArray.full /
        // full_like / ones (Backends/NDArray.Numeric.cs) decide the dtype and the route and call these for the storage
        // lanes.
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The storage-lane core behind <see cref="NDArray.full"/> and so <see cref="np.full(Shape, object, DType, char, string)"/>:
        ///     a new array of <paramref name="shape"/>'s dimensions in <paramref name="typeCode"/>'s lane, with the fill
        ///     value's element (<see cref="FullFillElement"/>) in every slot. For storage lanes only: the text classes take
        ///     their own fill routes, and a lane-sharing C-type class is the caller's to label afterwards.
        /// </summary>
        /// <param name="shape">The dimensions. A fresh dense layout is kept; a view's layout (a step, an offset, a broadcast)
        ///     is re-laid out in C order (<see cref="Shape.ForFreshAllocation"/>).</param>
        /// <param name="fill_value">The fill value, a scalar; null is Python's <c>None</c>.</param>
        /// <param name="typeCode">The storage lane.</param>
        /// <returns>A fresh, owned, writeable array with every element the fill value's element.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="typeCode"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="ValueError">A negative dimension, or a byte count past the limit
        ///     (<see cref="AllocationGuard"/>).</exception>
        /// <exception cref="OverflowError">A Python int out of the lane's integer range.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        internal static NDArray FullCore(Shape shape, object fill_value, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            // This path allocates from shape.size directly instead of going through
            // UnmanagedStorage.Allocate, so it needs the dimension guard explicitly: for
            // (2^62, 2^62) the element count ITSELF wraps to 0, and a zero-element request is
            // one the allocator happily satisfies. Checking dimensions rather than the wrapped
            // product is the only way to see it.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            object fill = FullFillElement(fill_value, typeCode, shape.size);

            // The wrapping constructor installs the shape as given, so the fresh-layout rule of the allocating funnel has
            // to be applied here: a view's Shape (np.full(x.Shape, v) for x = a[::4]) would otherwise address past the
            // `size`-element buffer, and a read-only source's Shape would make the new array read-only.
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, fill), shape.ForFreshAllocation()));
        }

        /// <summary>
        ///     The element <c>np.full</c> / <c>np.full_like</c> fill with — NumPy's <c>copyto(a, fill_value,
        ///     casting='unsafe')</c> of a scalar (<see cref="UnmanagedStorage.TryCopytoUnsafeScalarElement"/>), boxed as
        ///     <paramref name="typeCode"/>'s exact CLR type.
        /// </summary>
        /// <param name="fill_value">The caller's fill value; <c>null</c> is Python's <c>None</c>.</param>
        /// <param name="typeCode">The new array's dtype.</param>
        /// <param name="size">The new array's element count: <c>None</c> is converted only when there is an element to
        ///     write.</param>
        /// <returns>The boxed element; for a value that is not a scalar (a list, an array with dimensions), the house
        ///     conversion <see cref="Converts.ChangeType(object, NPTypeCode)"/> of it.</returns>
        /// <exception cref="OverflowError">A Python int out of the dtype's integer range (NumPy's OverflowError texts):
        ///     the copy is unsafe, but the Python scalar is packed first.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) array.</exception>
        /// <remarks>
        ///     <para>A value of the dtype's own C# type is already the stored element (<c>np.full(3, 0.5)</c>,
        ///     <c>np.full(3, 7, dtype: np.int32)</c>): it returns as is, with no packing.</para>
        ///     <para><c>None</c> reaches NumPy's <c>copyto(a, None, casting='unsafe')</c> as a 0-d OBJECT array, not a
        ///     Python scalar: nothing packs it up front, and the object-to-dtype cast runs once per element written — so
        ///     an empty result never converts it (<c>np.full((0,), None, dtype=int)</c> returns an empty array) while one
        ///     element of an integer dtype raises <c>int(None)</c>'s TypeError (probed 2.4.2). It used to fall to the
        ///     house conversion below, which turned null into ZERO.</para>
        ///     <para>The same holds for every value but a Python int, float or complex (<see cref="IsPackedBeforeCopy"/>):
        ///     NumPy's copyto packs only those before the copy, and casts anything else per element written, so text that
        ///     does not parse into an EMPTY numeric result raises nothing (<c>np.full((0,), 'x', dtype=int)</c> is an empty
        ///     array, probed 2.4.2; it raised the parse's ValueError).</para>
        /// </remarks>
        internal static object FullFillElement(object fill_value, NPTypeCode typeCode, long size)
        {
            // An EMPTY result writes no element, so NumPy's copyto converts only what it converts BEFORE the copy: a Python
            // int, float or complex is packed first (np.full((0,), 300, dtype=np.int8) raises), while None, text and every
            // other value is cast per element written. The dtype's zero stands for the element (a fresh slot is zeroed).
            if (size == 0 && !IsPackedBeforeCopy(fill_value))
                return ReductionInitial.Box(new byte[ReductionInitial.SlotBytes], typeCode);

            if (fill_value is null)
                return ReductionInitial.BoxNone(typeCode);

            // Only a number qualifies (Type.GetTypeCode unwraps an array type to its element type).
            if (NDArray.IsStoreScalar(fill_value) && fill_value.GetType().GetTypeCode() == typeCode)
                return fill_value;

            Span<byte> element = stackalloc byte[ReductionInitial.SlotBytes];
            if (UnmanagedStorage.TryCopytoUnsafeScalarElement(fill_value, typeCode, element))
                return ReductionInitial.Box(element, typeCode);
            return Converts.ChangeType(fill_value, typeCode);
        }

        /// <summary>
        ///     Whether NumPy's <c>copyto(a, value, casting='unsafe')</c> converts <paramref name="value"/> BEFORE its copy: a
        ///     Python int, float or complex is packed into <c>npy_find_descr_for_scalar</c>'s descriptor up front (its range
        ///     checked even for an empty result), while every other value - None, text, a NumPy scalar, an array - is cast
        ///     per element written, so it is never converted for an empty result.
        /// </summary>
        /// <param name="value">The fill value, in the C# boundary map: a C# integer, <see cref="BigInteger"/>,
        ///     <c>float</c>, <c>double</c> or <see cref="Complex"/> is a Python scalar, and so is a C# literal's 0-d array
        ///     (<see cref="NDArray.WasPythonLiteral"/>); null is <c>None</c>.</param>
        /// <returns>True for the Python scalars <see cref="TryCopytoUnsafeScalarElement"/> packs first.</returns>
        private static bool IsPackedBeforeCopy(object value)
            => value is sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger or float or double or Complex
               || value is NDArray { WasPythonLiteral: true };

        /// <summary>
        ///     The storage-lane core behind every public <c>ones(..., dtype)</c> overload (which all take the single
        ///     <see cref="NumSharp.DType"/> descriptor parameter, like NumPy's <c>dtype=</c>) and <c>ones_like</c>: NumPy's
        ///     <c>copyto(a, 1, casting='unsafe')</c> for a class whose elements are plain bytes, the Python int 1 converted
        ///     ONCE by the class's <c>setitem</c> and then written into every element by the lane fill.
        /// </summary>
        /// <param name="shape">Shape of the new array. A dense layout (C, F or a KEEPORDER permutation at offset 0) is
        ///     kept, which is how <c>ones_like(a, order='K')</c> gets <c>a</c>'s memory order; any view layout is re-laid out
        ///     in C order (<see cref="Shape.ForFreshAllocation"/>).</param>
        /// <param name="dtype">The array's descriptor: a class with a storage lane whose element fits
        ///     <see cref="ReductionInitial.SlotBytes"/> (every numeric class). The text classes take
        ///     <c>NDArray.OnesOfText</c> instead: a fixed-width string has no lane fill, and a <c>StringDType</c> element is
        ///     a reference into its array's allocator.</param>
        /// <returns>A fresh, owned, writeable array of <paramref name="shape"/>'s dimensions with every element the class's
        ///     one (True for bool, 1+0j for complex, U+0001 for NumSharp's char, 1 otherwise).</returns>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows (<see cref="AllocationGuard"/>).</exception>
        /// <exception cref="NotSupportedException">The class has no storage yet (datetime64): its <c>setitem</c> and its
        ///     storage lane do not exist.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ones.html. The one was a switch over the storage lanes
        ///     (a typed constant per lane); the class's <c>setitem</c> answers it now (<c>PyArray_Pack</c> of the Python
        ///     int 1), and <c>getitem</c> boxes it as the lane's exact CLR type for the fill. A non-native descriptor stores
        ///     swapped bytes and reads them back unswapped, so the fill value is the native one, as NumSharp stores it.
        /// </remarks>
        internal static unsafe NDArray OnesCore(Shape shape, DType dtype)
        {
            // GetTypeCode raises the descriptive "no storage yet" for a class without one, before anything is allocated.
            var typeCode = dtype.GetTypeCode();
            // NumPy's copyto(a, 1, casting='unsafe') converts the Python int once and writes that one element into every
            // slot: the class's setitem gives the element (for bool the True that NumPy's cast of the packed int64 1
            // gives), stored into a stack slot once; getitem boxes it as the lane's CLR type for the lane fill.
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            dtype.Meta.setitem(dtype, 1L, slot, null);
            object one = dtype.Meta.getitem(dtype, slot, null);

            // Allocates from shape.size directly (not via UnmanagedStorage.Allocate), so the
            // dimension guard has to be explicit here — see np.full.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            // The storage constructor installs the shape as given, so the fresh-layout rule of the allocating funnel has
            // to be applied here: a view's Shape (np.ones(x.Shape) for x = a[::4]) would otherwise address past the
            // `size`-element buffer, and a read-only source's Shape would make the new array read-only. It must be the
            // storage constructor, as in np.full: the public wrap constructor new NDArray(IArraySlice, Shape) reads
            // dimensions only (NumPy's ndarray(shape, buffer=...)), which would turn the kept F / KEEPORDER layout of
            // ones_like(a, order='K') into C order.
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, one), shape.ForFreshAllocation()));
        }

        // ---------------------------------------------------------------------------------------------
        // arrayobject.c's array_new, the constructor numpy.ndarray(shape, dtype, buffer, offset, strides, order), over
        // ctors.c's PyArray_NewFromDescr_int (np-layer dtype plan, Appendix D's UnmanagedStorage.Ctors.cs): the checks in
        // NumPy's order, then an uninitialized allocation or an array laid over the buffer's bytes (NDBuffer, the port of
        // PyArray_BufferConverter). np.ndarray is a facade over it.
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's text when <c>PyArray_CheckStrides</c> refuses the strides (and NumSharp's when its own bounds
        ///     check refuses a strided layout NumPy let through).</summary>
        private const string NdarrayStridesIncompatible = "strides is incompatible with shape of requested array and size of buffer";

        /// <summary>NumPy's text when a buffer holds fewer bytes than the array needs.</summary>
        private const string NdarrayBufferTooSmall = "buffer is too small for requested array";

        /// <summary>
        ///     NumPy's <c>array_new</c> (<c>numpy/_core/src/multiarray/arrayobject.c</c>), the body of
        ///     <see cref="np.ndarray(Shape, DType, object, long, long[], char?)"/> once <c>None</c> was refused as the shape:
        ///     the default dtype, the unsized-string refusal, the buffer resolved (and pinned), the order, the strides'
        ///     length and their bounds, then an uninitialized allocation (<see cref="NdarrayAllocate"/>) or an array over
        ///     the buffer's bytes (<see cref="NdarrayOverBuffer"/>).
        /// </summary>
        /// <param name="shape">The dimensions (only they are read; not <c>default(Shape)</c>, which the facade
        ///     refuses).</param>
        /// <param name="dtype">The element type, or null for float64 (NumPy's <c>NPY_DEFAULT_TYPE</c>).</param>
        /// <param name="buffer">The memory to use, or null to allocate (see the facade for the accepted kinds).</param>
        /// <param name="offset">The byte offset of the first element into <paramref name="buffer"/>; ignored without
        ///     one.</param>
        /// <param name="strides">The BYTE stride of every axis, or null for <paramref name="order"/>'s.</param>
        /// <param name="order"><c>'C'</c> or <c>'F'</c> (<c>'A'</c> and <c>'K'</c> mean C); ignored with strides.</param>
        /// <returns>A new array that owns its uninitialized memory, or a view of the buffer (a decoded native copy for a
        ///     non-native dtype).</returns>
        /// <exception cref="TypeError"><paramref name="buffer"/> is not a buffer, or it is too small without
        ///     strides.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is invalid; <paramref name="strides"/> has the wrong length
        ///     or does not fit the buffer; a dimension is negative or the byte count overflows; an <see cref="NDArray"/>
        ///     buffer is not contiguous; a negative offset puts elements before the buffer.</exception>
        /// <exception cref="BufferError">A <see cref="np.MemoryView"/> buffer over a non-contiguous array.</exception>
        /// <exception cref="NotSupportedException">A stride that splits an element, a dtype without storage, an unsized
        ///     string dtype, or a <c>StringDType</c> pairing NumSharp refuses.</exception>
        /// <remarks>A failed call releases a managed buffer's pin before the exception leaves: nothing holds the buffer
        ///     yet. After the hand-over to the result the release does nothing.</remarks>
        internal static NDArray ArrayNew(Shape shape, DType dtype, object buffer, long offset, long[] strides, char order)
        {
            // NPY_DEFAULT_TYPE. A descriptor without storage fails here, before a buffer is pinned.
            dtype ??= NumSharp.DType.Double;
            NPTypeCode typeCode = dtype.GetTypeCode();

            // An unsized descriptor (NumPy's PyDataType_ISUNSIZED: an item size of 0; in NumSharp only the S and U classes'
            // unsized descriptors): NumPy's array_new passes _NPY_ARRAY_ALLOW_EMPTY_STRING, so it builds an S0/U0 array
            // (every element '' and every stride 0) - and over a buffer with the shape (-1,) it divides by the zero item size.
            // A NumSharp string buffer holds at least one character per element, so the request is refused here, before a
            // managed buffer is pinned, rather than silently widened to S1/U1 (np.empty's rule, which np.ndarray skips).
            if (dtype.itemsize == 0)
                throw new NotSupportedException(
                    $"np.ndarray with the unsized string dtype {dtype.ToString(true)}: NumPy builds a zero-width (itemsize 0) " +
                    "array here, which NumSharp's fixed-width string buffers cannot hold. Give a width: 'S1', '<U5'.");

            // PyArray_BufferConverter: the buffer is resolved (and a managed buffer pinned) before the order is parsed.
            NDBuffer buf = NDBuffer.Acquire(buffer);
            try
            {
                // array_new asks only whether the order is NPY_FORTRANORDER; 'A' and 'K' are the C layout.
                bool fortran = OrderResolver.ResolveBufferOrder(order) == 'F';
                int itemSize = dtype.itemsize;
                long[] dims = shape.dimensions;

                if (strides != null)
                {
                    if (strides.Length != dims.Length)
                        throw new ValueError("strides, if given, must be the same length as shape");

                    // Without a buffer NumPy checks the strides against the array's own size at offset 0.
                    if (!Shape.CheckStrides(itemSize, dims, buf?.Length ?? 0, buf is null ? 0 : offset, strides))
                        throw new ValueError(NdarrayStridesIncompatible);
                }

                if (buf is null)
                    return NdarrayAllocate(dims, dtype, typeCode, strides, fortran);

                return NdarrayOverBuffer(buf, dims, dtype, typeCode, offset, strides, fortran);
            }
            catch
            {
                // A failed call must not keep a managed buffer pinned. After the hand-over this does nothing.
                buf?.Release();
                throw;
            }
        }

        /// <summary>
        ///     The no-buffer branch of <see cref="np.ndarray(Shape, DType, object, long, long[], char?)"/>: NumPy's
        ///     <c>PyArray_NewFromDescr_int</c> with no data pointer.
        /// </summary>
        /// <param name="dims">The dimensions (not yet validated).</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="typeCode">Its storage type.</param>
        /// <param name="strides">The checked byte strides, or null for the order's.</param>
        /// <param name="fortran">True for the column-major layout (ignored with strides).</param>
        /// <returns>A new array that owns its uninitialized memory.</returns>
        /// <exception cref="ValueError">A negative dimension, or a byte count that overflows.</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements splits an element.</exception>
        private static NDArray NdarrayAllocate(long[] dims, DType dtype, NPTypeCode typeCode, long[] strides, bool fortran)
        {
            int itemSize = dtype.itemsize;
            AllocationGuard.CheckDimensions(dims, itemSize);
            long size = ProductOfValidated(dims);

            if (strides is null)
            {
                // np.empty's allocation in C or F order. The shape's dimension array is immutable, so it is shared.
                var nd = new NDArray(dtype, new Shape(dims, fortran ? 'F' : 'C', copy: false), fillZeros: false);
                // NumPy zeroes every stride of an array it allocates empty (ctors.c: "Make sure all the strides are 0").
                if (size == 0 && dims.Length > 0)
                    nd.Storage.SetShapeUnsafe(new Shape(dims, new long[dims.Length], 0, 0));
                return nd;
            }

            long[] elementStrides;
            if (size == 0)
            {
                // The given strides are zeroed too: an empty allocation has no element to address.
                elementStrides = new long[dims.Length];
            }
            else if (!Shape.TryElementStrides(strides, dims, itemSize, out elementStrides, out int badAxis))
            {
                throw StrideSplitsElement(strides, badAxis, itemSize);
            }

            // CheckStrides kept every element inside prod(dims) elements from offset 0, so that is the whole buffer.
            var layout = new Shape(dims, elementStrides, 0, size);
            // NumPy leaves an array it creates with zero strides writeable; NumSharp's flags call a stride-0 axis a
            // broadcast and clear WRITEABLE, so it is set back (as setflags(write=True) on a broadcast does).
            if (!layout.IsWriteable)
                layout = layout.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);

            // A string array allocates its descriptor's bytes (the lane alone has no width); a non-native string descriptor
            // is allocated native, as np.empty allocates it. A class whose items hold references (PyDataType_REFCHK;
            // StringDType is the only one) allocates zeroed cells and its own allocator (NumPy's NPY_NEEDS_INIT: every
            // element reads as '').
            IArraySlice slice = dtype.Meta.HasReferences
                ? VStringArraySlice.Allocate(dtype, size)
                : typeCode == NPTypeCode.String
                ? StringArraySlice.Allocate(Backends.Strings.NDStringStorage.AllocationDescr(dtype), size, false)
                : ArraySlice.Allocate(typeCode, size, false);
            var storage = new UnmanagedStorage(slice, layout);
            return new NDArray(storage);
        }

        /// <summary>
        ///     The buffer branch of <see cref="np.ndarray(Shape, DType, object, long, long[], char?)"/>: NumPy's shape
        ///     <c>-1</c> and too-small checks, the dimensions, NumSharp's exact bounds check, then a view over the bytes.
        /// </summary>
        /// <param name="buf">The resolved buffer; its pin is handed to the result or released by the caller.</param>
        /// <param name="dims">The dimensions (not yet validated; <c>(-1,)</c> means "as many as fit").</param>
        /// <param name="dtype">The element type the bytes are read as.</param>
        /// <param name="typeCode">Its storage type.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="strides">The checked byte strides, or null for the order's.</param>
        /// <param name="fortran">True for the column-major layout (ignored with strides).</param>
        /// <returns>A view of the buffer; for a non-native dtype, a decoded native copy.</returns>
        /// <exception cref="TypeError">The buffer is too small (without strides).</exception>
        /// <exception cref="ValueError">A negative dimension, an overflowing byte count, or a layout outside the buffer
        ///     (strides, or a negative offset).</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements splits an element.</exception>
        private static unsafe NDArray NdarrayOverBuffer(NDBuffer buf, long[] dims, DType dtype, NPTypeCode typeCode, long offset, long[] strides, bool fortran)
        {
            int itemSize = dtype.itemsize;
            long length = buf.Length;

            if (dims.Length == 1 && dims[0] == -1)
            {
                // array_new: the shape (-1,) is every whole element after the offset. C division truncates toward zero, so
                // an offset past the end gives a negative count, which the dimension check below reports.
                dims = new[] { unchecked(length - offset) / itemSize };
            }
            else if (strides is null && length < unchecked(offset + itemSize * Shape.MultiplyListWrapping(dims)))
            {
                // NumPy's arithmetic wraps here; the exact check below catches what the wrap lets through.
                throw new TypeError(NdarrayBufferTooSmall);
            }

            AllocationGuard.CheckDimensions(dims, itemSize);

            long[] byteStrides = strides ?? Shape.FillByteStrides(dims, itemSize, fortran);

            // NumSharp's exact bounds check (NumPy has none past its own): every byte an element occupies must lie in
            // [0, length). An empty array occupies nothing, so any offset NumPy accepted is kept for it.
            Shape.ExactByteExtent(itemSize, dims, byteStrides, out Int128 lower, out Int128 upper);
            if (upper != lower && ((Int128)offset + lower < 0 || (Int128)offset + upper > length))
            {
                if (strides != null)
                    throw new ValueError(NdarrayStridesIncompatible);
                if (offset < 0)
                    throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"offset must be non-negative and no greater than buffer length ({length})"));
                throw new TypeError(NdarrayBufferTooSmall);
            }

            if (!Shape.TryElementStrides(byteStrides, dims, itemSize, out long[] elementStrides, out int badAxis))
                throw StrideSplitsElement(byteStrides, badAxis, itemSize);

            // Items that hold references on either side of the buffer (the requested class's PyDataType_REFCHK, or the
            // cells of a StringDType array buffer): NumPy lays any dtype over any bytes, NumSharp lays only a StringDType
            // view over a StringDType array's cells (sharing their allocator). Raised here, after every check NumPy makes,
            // as the exact bounds check above is.
            if (dtype.Meta.HasReferences || buf.VStringCells is not null)
                return NdarrayOverVStringCells(buf, dims, dtype, offset, elementStrides, (long)(upper - lower));

            // Every check has passed: from here on the buffer belongs to the result.
            NDArray view = NdarrayViewBytes(buf, dims, dtype, typeCode, offset, elementStrides, (long)lower, (long)(upper - lower));
            if (dtype.isnative)
                return view;

            // NumSharp stores native values only, so a non-native dtype is decoded into a new array (np.frombuffer's
            // rule): copy the bytes, swap them, and drop the view (which releases the buffer if nothing else holds it).
            NDArray decoded = view.copy();
            view.Dispose();
            decoded.byteswap(inplace: true);
            return decoded;
        }

        /// <summary>
        ///     Lays the dimensions over the buffer's bytes without a copy, as ONE storage: over the buffer array's own slice
        ///     when the dtype and its element grid match (an ordinary view), else over a forwarding alias of the span the
        ///     array occupies. The flags are NumPy's for an array over a buffer.
        /// </summary>
        /// <param name="buf">The checked buffer (its pin is handed over here).</param>
        /// <param name="dims">The validated dimensions.</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="typeCode">Its storage type (the native type for a non-native dtype).</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="elementStrides">The checked strides, in elements.</param>
        /// <param name="lowerBytes">The lowest byte the array occupies, relative to its first element (0 or negative).</param>
        /// <param name="spanBytes">The number of bytes from the lowest to one past the highest occupied byte (0 when
        ///     empty).</param>
        /// <returns>The view; it does not own its data, and its base is the buffer's root.</returns>
        private static unsafe NDArray NdarrayViewBytes(NDBuffer buf, long[] dims, DType dtype, NPTypeCode typeCode, long offset,
            long[] elementStrides, long lowerBytes, long spanBytes)
        {
            int itemSize = dtype.itemsize;
            // NumPy: offset + (char *)buffer.ptr. For an empty array this may point outside the buffer; nothing is read.
            byte* data = unchecked(buf.Start + offset);

            // A string dtype's elements are its descriptor's width, which the lane does not carry: a buffer pinned here is
            // rooted as plain bytes (the base is a uint8 array over the whole buffer, NumPy's non-array base object), and
            // the result is a string slice over its occupied span. Its descriptor is the native one (a non-native request
            // is decoded by the caller after this view is built).
            bool isString = typeCode == NPTypeCode.String;
            DType stringDescr = isString ? Backends.Strings.NDStringStorage.AllocationDescr(dtype) : null;
            IArraySlice arcOwner;
            UnmanagedStorage root = isString
                ? buf.TakeRoot(NPTypeCode.Byte, 1, out arcOwner)
                : buf.TakeRoot(typeCode, itemSize, out arcOwner);

            IArraySlice slice;
            long first;
            byte* ownerStart = (byte*)arcOwner.Address;
            // The buffer array's own slice is shared only when it holds the SAME elements: for strings that is the same
            // descriptor (class and width) - the lane is shared by every width, and an S4 view of a U1 array, or a U1 view
            // of a U2 array, must read the bytes, not the owner's elements.
            bool sameElements = arcOwner.TypeCode == typeCode
                                && (!isString || ((StringArraySlice)arcOwner).Descr == stringDescr);
            if (spanBytes > 0 && sameElements && data >= ownerStart && (data - ownerStart) % itemSize == 0)
            {
                // The buffer array's own element type and grid: share its slice, as an ordinary view does (its reference
                // counting is the owner's already). The exact bounds check put every element inside it.
                slice = arcOwner;
                first = (data - ownerStart) / itemSize;
            }
            else
            {
                // Another dtype, an offset off the grid, or an empty array: an alias of exactly the occupied span, whose
                // reference counting forwards to the buffer's owner, so the buffer lives while any array over it does.
                // A string alias is the span's bytes read as elements of the descriptor (spanBytes is a whole number of
                // elements: it runs from the lowest element's first byte to one past the highest element's last).
                slice = isString
                    ? new StringArraySlice(ArraySlice.WrapShared<byte>(data + lowerBytes, spanBytes, arcOwner), stringDescr)
                    : ArraySlice.WrapShared(typeCode, data + lowerBytes, spanBytes / itemSize, arcOwner);
                first = -lowerBytes / itemSize;
            }

            // NumPy's WRITEABLE for an array over a buffer: exactly when the buffer is writable (a stride of 0 does not
            // clear it, while NumSharp's flags call a stride-0 axis a read-only broadcast). ALIGNED needs no work here:
            // flags.aligned reads the memory's real alignment (UnmanagedStorage.IsDataAligned), which for whole-element
            // strides is NumPy's raw_array_is_aligned of the data pointer.
            var layout = new Shape(dims, elementStrides, first, slice.Count);
            layout = buf.Writable
                ? layout.WithFlags(flagsToSet: ArrayFlags.WRITEABLE)
                : layout.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            return new NDArray(UnmanagedStorage.OverBuffer(slice, layout, root));
        }

        /// <summary>
        ///     The buffer branch when <c>StringDType</c> is on either side: a <c>StringDType</c> <paramref name="dtype"/> over a
        ///     <c>StringDType</c> array's cells is a VIEW that shares the cells and their allocator (NumPy's answer: a view
        ///     whose base is the buffer array, writeable when it is); every other pairing is refused.
        /// </summary>
        /// <param name="buf">The checked buffer (its pin, if any, is released by the caller on a refusal).</param>
        /// <param name="dims">The validated dimensions.</param>
        /// <param name="dtype">The requested element type.</param>
        /// <param name="offset">The byte offset of the first element.</param>
        /// <param name="elementStrides">The checked strides, in 16-byte cells for a <c>StringDType</c> request.</param>
        /// <param name="spanBytes">The number of bytes the array occupies (0 when empty).</param>
        /// <returns>The <c>StringDType</c> view.</returns>
        /// <exception cref="NotSupportedException">A non-<c>StringDType</c> dtype over a <c>StringDType</c> array's cells, a
        ///     <c>StringDType</c> dtype over any other buffer, or an offset that splits a cell.</exception>
        /// <remarks>
        ///     <para>
        ///     <b>Why the refusals (divergences from NumPy, which lays any dtype over any bytes).</b> A <c>StringDType</c> cell
        ///     holds its text inline only up to 15 bytes; a longer text lives in the array's allocator and the cell holds an
        ///     arena offset or a heap pointer. Reading the cells as numbers is harmless, but a WRITABLE view of them (NumPy's)
        ///     lets any write turn a cell into a pointer anywhere, and a <c>StringDType</c> array over raw bytes reads cells
        ///     with no allocator behind them (NumPy reads garbage, or raises <c>MemoryError: Failed to load string</c>, for a
        ///     long string). The bytes themselves stay reachable as a copy: <c>a.tobytes()</c> is exactly the cells NumPy's
        ///     view would read.
        ///     </para>
        ///     <para>
        ///     <b>The view's descriptor</b> is the requested one (its <c>na_object</c> and <c>coerce</c>), over the buffer's
        ///     cells and allocator. NumPy gives the array the requested descriptor's OWN allocator, so a long text written by
        ///     another descriptor is unreadable there; NumSharp reads every text correctly (a safety divergence).
        ///     </para>
        /// </remarks>
        private static unsafe NDArray NdarrayOverVStringCells(NDBuffer buf, long[] dims, DType dtype, long offset, long[] elementStrides, long spanBytes)
        {
            VStringArraySlice cells = buf.VStringCells;
            if (dtype.Meta is not VStringDTypeMeta)
                throw new NotSupportedException(
                    $"np.ndarray cannot read the bytes of a StringDType array as {dtype.ToString(true)}: its 16-byte cells hold " +
                    "references into the array's allocator, and a writable view of them could corrupt it (NumPy lays the view " +
                    $"anyway). Read a copy of the cells instead: np.frombuffer(a.tobytes(), \"{dtype.str}\").");
            if (cells is null)
                throw new NotSupportedException(
                    $"np.ndarray cannot lay {dtype.ToString(true)} over a buffer that is not a StringDType array: a StringDType " +
                    "element is a 16-byte cell whose text can live in its array's allocator, and these bytes have none (NumPy " +
                    "builds the array anyway and reads garbage, or fails, for a long string). Lay it over a StringDType array, " +
                    "or build it from text: np.array(values, dtype).");

            const int cellBytes = Backends.Strings.VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            long first = 0;
            if (spanBytes > 0)
            {
                // Element [0, ..., 0] must start on a cell of the buffer array: a misaligned offset would read the halves
                // of two cells as one (NumPy does, and fails or reads garbage). An empty array reads nothing, so any offset
                // NumPy accepted stays legal for it.
                long delta = (long)(buf.Start + offset - (byte*)cells.Address);
                if (delta % cellBytes != 0)
                    throw new NotSupportedException(
                        string.Create(CultureInfo.InvariantCulture, $"np.ndarray: the byte offset {offset} splits a 16-byte StringDType cell; a StringDType view must start ") +
                        "on a cell of its buffer (NumPy reads the straddled bytes as a cell).");
                first = delta / cellBytes;
            }

            // The requested descriptor's NA object and coercion govern the view; the cells and their allocator are the
            // buffer array's, shared (reference counting forwards to the cells' block either way).
            VStringArraySlice slice = dtype.Equals(cells.Descr) ? cells : new VStringArraySlice(cells.Cells, cells.Allocator, dtype);

            // Every check has passed: the buffer belongs to the result (its base is the buffer array's owner).
            UnmanagedStorage root = buf.TakeRoot(NPTypeCode.String, cellBytes, out _);
            var layout = new Shape(dims, elementStrides, first, slice.Count);
            // NumPy's WRITEABLE for an array over a buffer: exactly when the buffer is (a stride of 0 does not clear it).
            layout = buf.Writable
                ? layout.WithFlags(flagsToSet: ArrayFlags.WRITEABLE)
                : layout.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return new NDArray(UnmanagedStorage.OverBuffer(slice, layout, root));
        }

        /// <summary>The element count of dimensions that <see cref="AllocationGuard.CheckDimensions(long[], int)"/> has
        ///     accepted (non-negative, and the byte count fits a <see cref="long"/>, so the count does too).</summary>
        /// <param name="dims">The validated dimensions.</param>
        /// <returns>Their product; 1 for a 0-d shape.</returns>
        private static long ProductOfValidated(long[] dims)
        {
            long size = 1;
            for (int i = 0; i < dims.Length; i++)
                size *= dims[i];
            return size;
        }

        /// <summary>The refusal of a stride NumSharp cannot store: it addresses elements and is not a whole number of
        ///     them.</summary>
        /// <param name="byteStrides">The byte strides.</param>
        /// <param name="axis">The first axis whose stride splits an element.</param>
        /// <param name="itemSize">The element size.</param>
        /// <returns>The exception to throw.</returns>
        private static NotSupportedException StrideSplitsElement(long[] byteStrides, int axis, int itemSize)
            => new NotSupportedException(
                string.Create(CultureInfo.InvariantCulture, $"The byte stride {byteStrides[axis]} of axis {axis} is not a multiple of the element size {itemSize}. ") +
                "NumSharp stores strides in whole elements, so it cannot lay out this array (NumPy accepts it as an unaligned array).");
    }
}
