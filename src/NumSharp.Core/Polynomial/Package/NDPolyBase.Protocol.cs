using System;
using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyBase.Protocol.cs — Python's binary-operator protocol for the series classes
// =============================================================================
//
// NumPy's operators are Python methods, and what `p + x` does is decided by CPython's protocol around them:
//
//   * ABCPolyBase.__add__ / __sub__ / __mul__ / __divmod__ first run _get_coefficients(other) — a series of
//     another class, domain, window or symbol raises TypeError/ValueError right there — then call the module
//     function inside `try: ... except Exception: return NotImplemented` (__divmod__ re-raises ZeroDivisionError);
//     __truediv__ refuses anything that is not a numbers.Number (bool excluded) and otherwise IS __floordiv__;
//     __pow__ has no try at all; the reflected methods skip _get_coefficients, and __rtruediv__ always declines.
//   * When a method returns NotImplemented, CPython tries the other operand's reflected slot — never for an
//     operand of the same type — and then its sequence fallbacks: `+` concatenates and `*` repeats a list, tuple
//     or str, and `str % x` is printf formatting. A NumPy scalar's or ndarray's reflected slot runs the ufunc,
//     which refuses a series (`__array_ufunc__ = None`) with its own TypeError. Only when everything declined
//     does Python raise `unsupported operand type(s) for +: 'A' and 'B'`.
//
// PyBinary reproduces that sequence for any pair of operands, and every C# operator, every dunder method and the
// series-valued evaluation (convert) go through it. An operand is classified ONCE (Operand): the house boundary
// map (C# numbers are Python scalars; Half / char / decimal and unmarked 0-d arrays NumPy scalars; the
// NDArray literal marks recover what a C# literal was), plus CPython's type name for the error texts.
//
// One deliberate difference (M2): where NumSharp's module function refuses an argument NumPy turns into an OBJECT
// array (NotSupportedException: a Python int past uint64, a list holding None), NumPy computes with Python objects
// and may succeed, so the refusal propagates instead of becoming NotImplemented — except for None and other
// non-numeric objects, whose object arithmetic always fails in NumPy too.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>The binary operators of NumPy's series protocol (each a forward / reflected method pair).</summary>
    internal enum PyBinOp : byte
    {
        /// <summary><c>+</c> (<c>__add__</c> / <c>__radd__</c>).</summary>
        Add,
        /// <summary><c>-</c> (<c>__sub__</c> / <c>__rsub__</c>).</summary>
        Sub,
        /// <summary><c>*</c> (<c>__mul__</c> / <c>__rmul__</c>).</summary>
        Mul,
        /// <summary><c>/</c> (<c>__truediv__</c> / <c>__rtruediv__</c>).</summary>
        TrueDiv,
        /// <summary><c>//</c> (<c>__floordiv__</c> / <c>__rfloordiv__</c>).</summary>
        FloorDiv,
        /// <summary><c>%</c> (<c>__mod__</c> / <c>__rmod__</c>).</summary>
        Mod,
        /// <summary><c>divmod()</c> (<c>__divmod__</c> / <c>__rdivmod__</c>).</summary>
        DivMod,
    }

    /// <summary>What a Python operand is, as far as the series protocol and CPython's fallbacks care.</summary>
    internal enum PyOperandKind : byte
    {
        /// <summary>A series (one of the six classes).</summary>
        Series,
        /// <summary>A Python <c>int</c> (any C# integer, <see cref="BigInteger"/>).</summary>
        PyInt,
        /// <summary>A Python <c>float</c> (C# <c>double</c> / <c>float</c>).</summary>
        PyFloat,
        /// <summary>A Python <c>complex</c> (<see cref="Complex"/>).</summary>
        PyComplex,
        /// <summary>A Python <c>bool</c> — not a <c>numbers.Number</c> for <c>/</c>, no common type for the series code.</summary>
        PyBool,
        /// <summary>A numeric NumPy scalar (<c>np.float64</c>, <c>np.int8</c>, ...; Half / char / decimal, unmarked 0-d arrays).</summary>
        NumPyNumber,
        /// <summary><c>np.bool_</c>: a NumPy scalar that is not a <c>numbers.Number</c>.</summary>
        NumPyBool,
        /// <summary>An ndarray (rank ≥ 1, or an explicit 0-d <c>np.array(x)</c>), a typed C# array.</summary>
        NdArray,
        /// <summary>A Python list (<c>object[]</c>, a jagged array, any other enumerable).</summary>
        List,
        /// <summary>A Python tuple (<see cref="ITuple"/>).</summary>
        Tuple,
        /// <summary>A Python <c>str</c> (or <c>np.str_</c>, <c>bytes</c>): a sequence that concatenates, repeats and formats.</summary>
        Str,
        /// <summary>Python's <c>None</c> (C# null).</summary>
        None,
        /// <summary>Any other object (CPython knows no arithmetic for it).</summary>
        Other,
    }

    /// <summary>One operand of a series operator: its kind, the value the module functions receive, and its Python names.</summary>
    internal readonly struct PyOperand
    {
        /// <summary>The kind.</summary>
        public readonly PyOperandKind Kind;
        /// <summary>The value the module functions receive for it (NumPy's argument): a boxed C# number for a Python scalar,
        ///     a boxed <see cref="PolyNumber"/> for a NumPy scalar, the array, the list / tuple / str object, null for None.</summary>
        public readonly object Arg;
        /// <summary>The number (a Python or NumPy scalar, or the array of an ndarray operand); default for other kinds.</summary>
        public readonly PolyNumber Number;
        /// <summary>The series (<see cref="PyOperandKind.Series"/> only).</summary>
        public readonly NDPolyBase Series;
        /// <summary>CPython's <c>tp_name</c>: <c>int</c>, <c>numpy.float64</c>, <c>numpy.ndarray</c>, <c>list</c>,
        ///     <c>Polynomial</c>, ... — what <c>unsupported operand type(s)</c> quotes.</summary>
        public readonly string TypeName;

        /// <summary>Creates an operand.</summary>
        /// <param name="kind">The kind.</param><param name="arg">The module-function argument.</param>
        /// <param name="number">The number, if any.</param><param name="series">The series, if any.</param>
        /// <param name="typeName">CPython's type name.</param>
        public PyOperand(PyOperandKind kind, object arg, in PolyNumber number, NDPolyBase series, string typeName)
        {
            Kind = kind;
            Arg = arg;
            Number = number;
            Series = series;
            TypeName = typeName;
        }

        /// <summary><c>isinstance(x, numbers.Number) and not isinstance(x, bool)</c>: the operands <c>/</c> accepts.</summary>
        public bool IsNumber => Kind is PyOperandKind.PyInt or PyOperandKind.PyFloat or PyOperandKind.PyComplex or PyOperandKind.NumPyNumber;

        /// <summary><c>str(type(x))</c>: <c>&lt;class 'int'&gt;</c>, <c>&lt;class 'numpy.polynomial.polynomial.Polynomial'&gt;</c>.</summary>
        public string ClassRepr => Kind == PyOperandKind.Series ? Series.Kind.ClassRepr : "<class '" + TypeName + "'>";
    }

    public abstract partial class NDPolyBase
    {
        // ---------------------------------------------------------------------------------------------
        //  Operand classification
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Classifies a C# value as the Python operand it stands for: a series; a C# number (Python int / float / complex,
        ///     a bool a Python bool); a <see cref="Half"/> / <c>char</c> / <c>decimal</c> (NumPy scalars); an
        ///     <see cref="NDArray"/> by its literal marks (a converted C# literal is the Python scalar it was, an unmarked 0-d
        ///     array a NumPy scalar — the house convention for a scalar operand —, an explicit <c>np.array(x)</c> or a rank ≥ 1
        ///     array an ndarray); a str; a tuple; a typed C# array (an ndarray); any other enumerable (a list); null (None).
        /// </summary>
        /// <param name="o">The value.</param>
        /// <returns>The operand.</returns>
        internal static PyOperand Operand(object o)
        {
            switch (o)
            {
                case null:
                    return new PyOperand(PyOperandKind.None, null, default, null, "NoneType");
                case NDPolyBase s:
                    return new PyOperand(PyOperandKind.Series, null, default, s, s.Kind.Name);
                case PolyNumber p:
                    return OfNumber(p);
                case NDArray nd:
                    return OfArray(nd);
                case bool b:
                    return new PyOperand(PyOperandKind.PyBool, b, PolyNumber.FromPython(PyScalar.Bool(b)), null, "bool");
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return new PyOperand(PyOperandKind.PyInt, o, PolyNumber.FromObject(o), null, "int");
                case float or double:
                    return new PyOperand(PyOperandKind.PyFloat, o, PolyNumber.FromObject(o), null, "float");
                case Complex:
                    return new PyOperand(PyOperandKind.PyComplex, o, PolyNumber.FromObject(o), null, "complex");
                case Half or char or decimal:
                    return OfNumber(PolyNumber.FromObject(o));
                case string s:
                    return new PyOperand(PyOperandKind.Str, s, default, null, "str");
                case ITuple:
                    return new PyOperand(PyOperandKind.Tuple, o, default, null, "tuple");
            }
            // A typed C# array (or Memory<T>) of a dtype is an ndarray (the house map); every other array and enumerable is a
            // Python list.
            if (PolySequence.IsArrayLike(o))
            {
                NDArray nd = np.asanyarray(o);
                return new PyOperand(PyOperandKind.NdArray, nd, PolyNumber.FromArray(nd), null, "numpy.ndarray");
            }
            if (PolySequence.IsSequence(o))
                return new PyOperand(PyOperandKind.List, o, default, null, "list");
            return new PyOperand(PyOperandKind.Other, o, default, null, o.GetType().Name);
        }

        /// <summary>The operand an <see cref="NDArray"/> stands for (see <see cref="Operand"/>).</summary>
        /// <param name="nd">The array.</param>
        /// <returns>The operand.</returns>
        private static PyOperand OfArray(NDArray nd)
        {
            // A C# literal converted at a call site: the Python object it was.
            if (nd.WasPythonLiteral)
                return Operand(nd.Storage.GetAtIndex(0));
            if (nd.WasPythonBool)
                return Operand(nd.GetBoolean());
            if (nd.WasPythonStr)
                return new PyOperand(PyOperandKind.Str, nd.PythonStrText, default, null, "str");
            if (nd.WasPythonBytes)
                return new PyOperand(PyOperandKind.Str, nd, default, null, "bytes");
            if (nd.WasNumPyScalar || (nd.ndim == 0 && !nd.IsExplicitZeroDimArray))
            {
                // NumPy's np.str_ is a str (concatenation, repetition and formatting all behave as a str's).
                if (nd.typecode == NPTypeCode.String)
                    return new PyOperand(PyOperandKind.Str, nd, default, null, "numpy.str_");
                return OfNumber(PolyNumber.ScalarOf(nd));
            }
            return new PyOperand(PyOperandKind.NdArray, nd, PolyNumber.FromArray(nd), null, "numpy.ndarray");
        }

        /// <summary>The operand a <see cref="PolyNumber"/> is (a value the package computed, or a classified C# number).</summary>
        /// <param name="p">The number.</param>
        /// <returns>The operand.</returns>
        private static PyOperand OfNumber(in PolyNumber p)
        {
            switch (p.Kind)
            {
                case PolyNumberKind.Python:
                {
                    if (p.Py.IsBool)
                        return new PyOperand(PyOperandKind.PyBool, !p.Py.I.IsZero, p, null, "bool");
                    var kind = p.Py.Kind switch
                    {
                        PyKind.Int => PyOperandKind.PyInt,
                        PyKind.Float => PyOperandKind.PyFloat,
                        _ => PyOperandKind.PyComplex,
                    };
                    return new PyOperand(kind, p.ToObject(), p, null, kind switch
                    {
                        PyOperandKind.PyInt => "int",
                        PyOperandKind.PyFloat => "float",
                        _ => "complex",
                    });
                }
                case PolyNumberKind.Scalar:
                    return new PyOperand(p.Dtype == NPTypeCode.Boolean ? PyOperandKind.NumPyBool : PyOperandKind.NumPyNumber,
                        p, p, null, NumPyScalarTypeName(p));
                default:
                    return new PyOperand(PyOperandKind.NdArray, p.Array, p, null, "numpy.ndarray");
            }
        }

        /// <summary>CPython's <c>tp_name</c> of a NumPy scalar: <c>numpy.float64</c>, <c>numpy.bool</c>, <c>numpy.uint16</c> for
        ///     NumSharp's char (the house map), <c>numpy.complex64</c> for an emulated complex64, <c>numsharp.decimal</c>.</summary>
        /// <param name="p">The scalar.</param>
        /// <returns>The name.</returns>
        internal static string NumPyScalarTypeName(in PolyNumber p)
        {
            if (p.IsComplex64)
                return "numpy.complex64";
            return p.Dtype switch
            {
                NPTypeCode.Boolean => "numpy.bool",
                NPTypeCode.Char => "numpy.uint16",
                NPTypeCode.Decimal => "numsharp.decimal",
                NPTypeCode.String => "numpy.str_",
                var t => "numpy." + t.AsNumpyDtypeName(),
            };
        }

        /// <summary>
        ///     The Python type a NumPy scalar becomes in an OBJECT-dtype ufunc loop (NumPy's getitem of the cast to object):
        ///     <c>float</c> for every real float, <c>complex</c> for a complex, <c>int</c> for an integer, <c>bool</c> — what
        ///     CPython's error quotes when that loop multiplies it by a non-numeric object.
        /// </summary>
        /// <param name="p">The scalar (a Python value names its own type).</param>
        /// <returns>The type name.</returns>
        internal static string ObjectLoopTypeName(in PolyNumber p)
        {
            if (p.IsPython)
                return p.Py.IsBool ? "bool" : p.Py.Kind switch { PyKind.Int => "int", PyKind.Float => "float", _ => "complex" };
            return p.Dtype switch
            {
                NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double => "float",
                NPTypeCode.Complex => "complex",
                NPTypeCode.Boolean => "bool",
                NPTypeCode.Decimal => "decimal.Decimal",
                _ => "int",
            };
        }

        // ---------------------------------------------------------------------------------------------
        //  The protocol
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <see cref="PyBinary"/> of two C# values inside one <see cref="NDScope"/>: the operator bodies call it. The
        ///     result's arrays are the new series' own (detached), so the scope reclaims only the conversions and the
        ///     module functions' intermediates.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The result: a series, or a boxed <c>(quo, rem)</c> tuple for <see cref="PyBinOp.DivMod"/>.</returns>
        /// <exception cref="TypeError">Python's errors (see <see cref="PyBinary"/>).</exception>
        private protected static object Binary(PyBinOp op, object a, object b)
        {
            using var scope = NDScope.Open();
            return PyBinary(op, Operand(a), Operand(b));
        }

        /// <summary>
        ///     Python's binary operator <c>a OP b</c> where at least one operand is a series: the left operand's forward
        ///     method, then (when it declines) the right operand's reflected method unless both have the same type, then
        ///     CPython's sequence fallbacks, then the <c>unsupported operand type(s)</c> TypeError.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>A series, or a boxed <c>(quo, rem)</c> for <see cref="PyBinOp.DivMod"/>.</returns>
        /// <exception cref="TypeError">NumPy's / CPython's texts: <c>Polynomial types differ</c>, <c>Domains differ</c>,
        ///     <c>Windows differ</c>, <c>unsupported types for true division: ...</c>, <c>unsupported operand type(s) for +:
        ///     'Polynomial' and 'bool'</c>, <c>can only concatenate list (not "Polynomial") to list</c>, <c>can't multiply
        ///     sequence by non-int of type 'Polynomial'</c>, <c>operand 'Polynomial' does not support ufuncs
        ///     (__array_ufunc__=None)</c>, <c>not all arguments converted during string formatting</c>, ...</exception>
        /// <exception cref="ValueError"><c>Polynomial symbols differ</c>; the constructor's errors for a result.</exception>
        /// <exception cref="DivideByZeroException">A zero divisor in <c>/ // % divmod</c> (NumPy's bare ZeroDivisionError).</exception>
        /// <exception cref="NotSupportedException">An operand NumPy computes with as Python objects (see the file header).</exception>
        internal static object PyBinary(PyBinOp op, in PyOperand a, in PyOperand b)
        {
            if (a.Kind == PyOperandKind.Series)
            {
                object r = a.Series.Forward(op, b);
                if (r is not null)
                    return r;
                // CPython tries the right operand's reflected slot only when its type differs from the left's. A series of
                // another class never gets here: _get_coefficients raised "Polynomial types differ" first.
                if (b.Kind == PyOperandKind.Series)
                    throw Unsupported(op, a, b);
                throw RightDeclined(op, a, b);
            }
            // The left operand is not a series, so the right one is. The left's forward slot: a Python number declines, a
            // NumPy scalar or ndarray defers to a series (__array_ufunc__ = None), lists and tuples have none — only a str's
            // `%` is printf formatting, which runs (and raises) before the series is asked.
            if (op == PyBinOp.Mod && a.Kind == PyOperandKind.Str)
                throw StrFormatting(a);
            object rr = b.Series.Reflected(op, a);
            if (rr is not null)
                return rr;
            throw LeftDeclined(op, a, b);
        }

        /// <summary>
        ///     The series' forward method (NumPy's <c>__add__</c>, <c>__sub__</c>, <c>__mul__</c>, <c>__truediv__</c>,
        ///     <c>__floordiv__</c>, <c>__mod__</c>, <c>__divmod__</c>), returning null for NumPy's NotImplemented.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="other">The right operand.</param>
        /// <returns>The result, or null (NotImplemented).</returns>
        /// <exception cref="TypeError"><c>_get_coefficients</c>'s texts; <c>/</c> by a non-number.</exception>
        /// <exception cref="ValueError"><c>Polynomial symbols differ</c>.</exception>
        /// <exception cref="DivideByZeroException">A zero divisor (re-raised, as NumPy's __divmod__ does).</exception>
        private object Forward(PyBinOp op, in PyOperand other)
        {
            switch (op)
            {
                case PyBinOp.Add:
                case PyBinOp.Sub:
                case PyBinOp.Mul:
                {
                    object oc = GetCoefficients(other);
                    NDArray c;
                    try
                    {
                        c = Engine(op, Coef, oc);
                    }
                    catch (Exception e) when (IsNotImplemented(e, other))
                    {
                        return null;
                    }
                    return Derive(c, fresh: true);
                }
                case PyBinOp.TrueDiv:
                    // "There is no true divide if the rhs is not a Number": raised, not declined.
                    if (!other.IsNumber)
                        throw new TypeError($"unsupported types for true division: '{Kind.ClassRepr}', '{other.ClassRepr}'");
                    return Quotient(DivModForward(other));
                case PyBinOp.FloorDiv:
                    return Quotient(DivModForward(other));
                case PyBinOp.Mod:
                    return Remainder(DivModForward(other));
                default:
                    return DivModForward(other) is { } t ? (object)(t.quo, t.rem) : null;
            }
        }

        /// <summary>
        ///     NumPy's <c>res[0]</c> of a <c>__divmod__</c> result (<c>//</c> and <c>/</c>): the quotient, with the remainder —
        ///     a series NumPy drops to the garbage collector — disposed here, because its arrays belong to no scope.
        /// </summary>
        /// <param name="t">The divmod result, or null (NotImplemented).</param>
        /// <returns>The quotient, or null.</returns>
        private static NDPolyBase Quotient((NDPolyBase quo, NDPolyBase rem)? t)
        {
            if (t is not { } r)
                return null;
            r.rem.Dispose();
            return r.quo;
        }

        /// <summary>NumPy's <c>res[1]</c> of a <c>__divmod__</c> result (<c>%</c>): the remainder, the quotient disposed.</summary>
        /// <param name="t">The divmod result, or null (NotImplemented).</param>
        /// <returns>The remainder, or null.</returns>
        private static NDPolyBase Remainder((NDPolyBase quo, NDPolyBase rem)? t)
        {
            if (t is not { } r)
                return null;
            r.quo.Dispose();
            return r.rem;
        }

        /// <summary>
        ///     The series' reflected method (NumPy's <c>__radd__</c>, ..., <c>__rdivmod__</c>) for <c>other OP self</c>:
        ///     <c>_get_coefficients</c> is NOT run; <c>__rtruediv__</c> always declines.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="other">The left operand.</param>
        /// <returns>The result, or null (NotImplemented).</returns>
        /// <exception cref="DivideByZeroException">A zero divisor (re-raised by NumPy's __rdivmod__).</exception>
        private object Reflected(PyBinOp op, in PyOperand other)
        {
            switch (op)
            {
                case PyBinOp.Add:
                case PyBinOp.Sub:
                case PyBinOp.Mul:
                {
                    NDArray c;
                    try
                    {
                        c = Engine(op, OperandArgument(other), Coef);
                    }
                    catch (Exception e) when (IsNotImplemented(e, other))
                    {
                        return null;
                    }
                    return Derive(c, fresh: true);
                }
                case PyBinOp.TrueDiv:
                    return null;   // "An instance of ABCPolyBase is not considered a Number."
                case PyBinOp.FloorDiv:
                    return Quotient(DivModReflected(other));
                case PyBinOp.Mod:
                    return Remainder(DivModReflected(other));
                default:
                    return DivModReflected(other) is { } t ? (object)(t.quo, t.rem) : null;
            }
        }

        /// <summary>NumPy's <c>__divmod__</c>: <c>_div(self.coef, othercoef)</c>, both parts as new series.</summary>
        /// <param name="other">The divisor.</param>
        /// <returns>(quo, rem), or null (NotImplemented).</returns>
        /// <exception cref="DivideByZeroException">A zero divisor.</exception>
        private (NDPolyBase quo, NDPolyBase rem)? DivModForward(in PyOperand other)
        {
            object oc = GetCoefficients(other);
            NDArray q, r;
            try
            {
                (q, r) = NDPolyAlgebra.Div(Basis, Coef, oc);
            }
            catch (DivideByZeroException)
            {
                throw;
            }
            catch (Exception e) when (IsNotImplemented(e, other))
            {
                return null;
            }
            // `quo = self.__class__(quo, ...)` before `rem = self.__class__(rem, ...)`.
            var quo = Derive(q, fresh: true);
            return (quo, Derive(r, fresh: true));
        }

        /// <summary>NumPy's <c>__rdivmod__</c>: <c>_div(other, self.coef)</c>, both parts as new series.</summary>
        /// <param name="other">The dividend.</param>
        /// <returns>(quo, rem), or null (NotImplemented).</returns>
        /// <exception cref="DivideByZeroException">A zero divisor (this series is zero).</exception>
        private (NDPolyBase quo, NDPolyBase rem)? DivModReflected(in PyOperand other)
        {
            NDArray q, r;
            try
            {
                (q, r) = NDPolyAlgebra.Div(Basis, OperandArgument(other), Coef);
            }
            catch (DivideByZeroException)
            {
                throw;
            }
            catch (Exception e) when (IsNotImplemented(e, other))
            {
                return null;
            }
            var quo = Derive(q, fresh: true);
            return (quo, Derive(r, fresh: true));
        }

        /// <summary><c>_add</c> / <c>_sub</c> / <c>_mul</c> of the class's basis.</summary>
        /// <param name="op">Add, Sub or Mul.</param>
        /// <param name="c1">The first series argument.</param><param name="c2">The second.</param>
        /// <returns>The module function's result.</returns>
        private NDArray Engine(PyBinOp op, object c1, object c2) => op switch
        {
            PyBinOp.Add => NDPolySeries.AddSub(c1, c2, subtract: false),
            PyBinOp.Sub => NDPolySeries.AddSub(c1, c2, subtract: true),
            _ => NDPolyAlgebra.Mul(Basis, c1, c2),
        };

        /// <summary>
        ///     NumPy's <c>_get_coefficients(other)</c>: a series must be of this class, with an equal domain and window
        ///     (<c>np.all(==)</c>: a NaN never matches, so a NaN domain differs from itself) and the same symbol — then its
        ///     coefficients; anything else is passed on as it is.
        /// </summary>
        /// <param name="other">The operand.</param>
        /// <returns>The module-function argument.</returns>
        /// <exception cref="TypeError"><c>Polynomial types differ</c>, <c>Domains differ</c>, <c>Windows differ</c>.</exception>
        /// <exception cref="ValueError"><c>Polynomial symbols differ</c>.</exception>
        private object GetCoefficients(in PyOperand other)
        {
            if (other.Kind != PyOperandKind.Series)
                return other.Arg;
            var o = other.Series;
            if (o.Kind != Kind)
                throw new TypeError("Polynomial types differ");
            if (!AllEqual(domain, o.domain))
                throw new TypeError("Domains differ");
            if (!AllEqual(window, o.window))
                throw new TypeError("Windows differ");
            if (symbol != o.symbol)
                throw new ValueError("Polynomial symbols differ");
            return o.Coef;
        }

        /// <summary>The module-function argument of a reflected operand (a series would be its coefficients).</summary>
        /// <param name="other">The operand.</param>
        /// <returns>The argument.</returns>
        private static object OperandArgument(in PyOperand other)
            => other.Kind == PyOperandKind.Series ? other.Series.Coef : other.Arg;

        /// <summary>
        ///     NumPy's <c>except Exception: return NotImplemented</c> around a module function: every failure declines,
        ///     except a NumSharp refusal of an operand NumPy computes with as Python objects (it may succeed in NumPy, so the
        ///     refusal stands) and failures no Python program sees (a disposed series, an exhausted machine).
        /// </summary>
        /// <param name="e">The exception.</param>
        /// <param name="other">The other operand.</param>
        /// <returns>Whether NumPy's method returns NotImplemented here.</returns>
        private static bool IsNotImplemented(Exception e, in PyOperand other)
        {
            switch (e)
            {
                case OutOfMemoryException:
                case ObjectDisposedException:
                case InsufficientExecutionStackException:
                    return false;
                case NotSupportedException:
                    // None and a non-numeric object: NumPy's object arithmetic with them always fails (None supports no
                    // operator, and _div's zero test, the only statement before it, is modelled by the module function).
                    return other.Kind is PyOperandKind.None or PyOperandKind.Other;
                default:
                    return true;
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  CPython's fallbacks and texts
        // ---------------------------------------------------------------------------------------------

        /// <summary>The operator's text in CPython's <c>unsupported operand type(s) for X</c>.</summary>
        /// <param name="op">The operator.</param>
        /// <returns><c>+</c>, <c>-</c>, <c>*</c>, <c>/</c>, <c>//</c>, <c>%</c> or <c>divmod()</c>.</returns>
        private static string Symbol(PyBinOp op) => op switch
        {
            PyBinOp.Add => "+",
            PyBinOp.Sub => "-",
            PyBinOp.Mul => "*",
            PyBinOp.TrueDiv => "/",
            PyBinOp.FloorDiv => "//",
            PyBinOp.Mod => "%",
            _ => "divmod()",
        };

        /// <summary>CPython's last word: <c>unsupported operand type(s) for +: 'A' and 'B'</c>.</summary>
        /// <param name="op">The operator.</param><param name="a">The left operand.</param><param name="b">The right operand.</param>
        /// <returns>The exception.</returns>
        private static TypeError Unsupported(PyBinOp op, in PyOperand a, in PyOperand b)
            => new TypeError($"unsupported operand type(s) for {Symbol(op)}: '{a.TypeName}' and '{b.TypeName}'");

        /// <summary>
        ///     What CPython does after the series' forward method declined: a NumPy scalar's or ndarray's reflected slot runs
        ///     the ufunc, which refuses the series; a list / tuple / str repeats under <c>*</c> but needs an index; otherwise
        ///     nothing is left.
        /// </summary>
        /// <param name="op">The operator.</param><param name="a">The series.</param><param name="b">The right operand.</param>
        /// <returns>The exception.</returns>
        private static TypeError RightDeclined(PyBinOp op, in PyOperand a, in PyOperand b)
        {
            switch (b.Kind)
            {
                case PyOperandKind.NumPyNumber:
                case PyOperandKind.NumPyBool:
                case PyOperandKind.NdArray:
                    return new TypeError($"operand '{a.TypeName}' does not support ufuncs (__array_ufunc__=None)");
                case PyOperandKind.List:
                case PyOperandKind.Tuple:
                case PyOperandKind.Str:
                    if (op == PyBinOp.Mul)
                        return new TypeError($"can't multiply sequence by non-int of type '{a.TypeName}'");
                    break;
            }
            return Unsupported(op, a, b);
        }

        /// <summary>
        ///     What CPython does after the series' reflected method declined: <c>+</c> concatenates a list, tuple or str
        ///     (refusing a series) and an ndarray's concatenation slot refuses everything; <c>*</c> repeats a list, tuple or
        ///     str (refusing a non-index count); otherwise nothing is left.
        /// </summary>
        /// <param name="op">The operator.</param><param name="a">The left operand.</param><param name="b">The series.</param>
        /// <returns>The exception.</returns>
        private static TypeError LeftDeclined(PyBinOp op, in PyOperand a, in PyOperand b)
        {
            if (op == PyBinOp.Add)
            {
                switch (a.Kind)
                {
                    case PyOperandKind.List:
                        return new TypeError($"can only concatenate list (not \"{b.TypeName}\") to list");
                    case PyOperandKind.Tuple:
                        return new TypeError($"can only concatenate tuple (not \"{b.TypeName}\") to tuple");
                    case PyOperandKind.Str:
                        return a.TypeName == "bytes"
                            ? new TypeError($"can't concat {b.TypeName} to bytes")
                            : new TypeError($"can only concatenate str (not \"{b.TypeName}\") to str");
                    case PyOperandKind.NdArray:
                        return new TypeError("Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please do not rely on this error; it may not be given on all Python implementations.");
                }
            }
            else if (op == PyBinOp.Mul && a.Kind is PyOperandKind.List or PyOperandKind.Tuple or PyOperandKind.Str)
                return new TypeError($"can't multiply sequence by non-int of type '{b.TypeName}'");
            return Unsupported(op, a, b);
        }

        /// <summary>
        ///     <c>str % series</c>: printf formatting runs first, with the series as the single argument. A format with no
        ///     conversion leaves the argument unconsumed — CPython's TypeError; one with a conversion would make a str (or
        ///     fail per conversion), which an operator returning a series cannot give.
        /// </summary>
        /// <param name="a">The str operand.</param>
        /// <returns>The exception.</returns>
        private static Exception StrFormatting(in PyOperand a)
        {
            string fmt = a.Arg as string ?? (a.Arg as NDArray)?.Storage.GetAtIndex(0) as string ?? string.Empty;
            for (int i = 0; i < fmt.Length; i++)
            {
                if (fmt[i] != '%')
                    continue;
                if (i + 1 < fmt.Length && fmt[i + 1] == '%')
                {
                    i++;
                    continue;
                }
                return new NotSupportedException("str % series formats the series into a str (printf-style), which the % operator of a series cannot return");
            }
            return new TypeError("not all arguments converted during string formatting");
        }

        // ---------------------------------------------------------------------------------------------
        //  Unary operators and the power
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>__neg__</c>: <c>self.__class__(-self.coef, self.domain, self.window, self.symbol)</c>.</summary>
        /// <returns>The new series.</returns>
        /// <exception cref="ObjectDisposedException">The series was disposed.</exception>
        private protected NDPolyBase Negative()
        {
            using var scope = NDScope.Open();
            return Derive(np.negative(Coef), fresh: true);
        }

        /// <summary>
        ///     NumPy's <c>p ** n</c> (<c>__pow__</c>): <c>self._pow(self.coef, n, maxpower=self.maxpower)</c> — no
        ///     try/except, so every error of the power argument propagates — in a new series.
        /// </summary>
        /// <param name="power">The power: a non-negative integer (Python's <c>int(pow)</c> and <c>power != pow</c> decide:
        ///     2.0 and True work, 2.5 and '2' do not).</param>
        /// <returns>The new series.</returns>
        /// <exception cref="ValueError"><c>Power must be a non-negative integer.</c>, <c>Power is too large</c> (above
        ///     <see cref="maxpower"/>), int()'s ValueErrors.</exception>
        /// <exception cref="TypeError">int() refuses the power: <c>int() argument must be a string, a bytes-like object or a
        ///     real number, not 'complex'</c> (a series: its class name), <c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>.</exception>
        /// <exception cref="OverflowException">An infinite power.</exception>
        public virtual NDPolyBase __pow__(object power)
        {
            using var scope = NDScope.Open();
            var op = Operand(power);
            object arg = power;
            switch (op.Kind)
            {
                case PyOperandKind.Series:
                    throw new TypeError($"int() argument must be a string, a bytes-like object or a real number, not '{op.TypeName}'");
                case PyOperandKind.NumPyNumber when op.Number.Dtype == NPTypeCode.Complex:
                {
                    // A NumPy complex SCALAR converts in int() — np.complexfloating.__int__ drops the imaginary part (NumPy's
                    // ComplexWarning, which NumSharp does not emit) — so the power is int(real), and `power != pow` then
                    // compares that int with the complex: unequal exactly when the imaginary part is not zero.
                    var z = (System.Numerics.Complex)op.Number.ToObject();
                    var power0 = PolyPowerArgument.Int(z.Real);
                    if (z.Imaginary != 0 || PolyPowerArgument.DiffersFromInt(power0, z.Real))
                        throw new ValueError("Power must be a non-negative integer.");
                    arg = z.Real;
                    break;
                }
                case PyOperandKind.NumPyNumber:
                case PyOperandKind.NumPyBool:
                    // Any other NumPy scalar: int() and `!=` read its value as the C# number of its dtype does.
                    arg = op.Number.ToObject();
                    break;
            }
            NDArray c = NDPolyAlgebra.Pow(Basis, Coef, arg, maxpower);
            return Derive(c, fresh: true);
        }

        /// <summary>NumPy's <c>p // other</c> (<c>__floordiv__</c>, then CPython's fallbacks): the quotient of
        ///     <c>divmod(p, other)</c>.</summary>
        /// <param name="other">The divisor: a series of this class, a number, an array or a sequence of coefficients.</param>
        /// <returns>The quotient series.</returns>
        /// <exception cref="TypeError">See <see cref="PyBinary"/>.</exception>
        /// <exception cref="DivideByZeroException">A zero divisor.</exception>
        public virtual NDPolyBase __floordiv__(object other) => (NDPolyBase)Binary(PyBinOp.FloorDiv, this, other);

        /// <summary>NumPy's <c>other // p</c> (<c>__rfloordiv__</c>): the quotient of <c>divmod(other, p)</c>.</summary>
        /// <param name="other">The dividend.</param>
        /// <returns>The quotient series.</returns>
        /// <exception cref="TypeError">See <see cref="PyBinary"/>.</exception>
        /// <exception cref="DivideByZeroException">This series is zero.</exception>
        public virtual NDPolyBase __rfloordiv__(object other) => (NDPolyBase)Binary(PyBinOp.FloorDiv, other, this);

        /// <summary>NumPy's <c>divmod(p, other)</c> (<c>__divmod__</c>): the quotient and the remainder.</summary>
        /// <param name="other">The divisor.</param>
        /// <returns>(quo, rem), two new series of this class.</returns>
        /// <exception cref="TypeError">See <see cref="PyBinary"/>.</exception>
        /// <exception cref="DivideByZeroException">A zero divisor.</exception>
        public (NDPolyBase quo, NDPolyBase rem) __divmod__(object other)
            => ((NDPolyBase, NDPolyBase))Binary(PyBinOp.DivMod, this, other);

        /// <summary>NumPy's <c>divmod(other, p)</c> (<c>__rdivmod__</c>).</summary>
        /// <param name="other">The dividend.</param>
        /// <returns>(quo, rem), two new series of this class.</returns>
        /// <exception cref="TypeError">See <see cref="PyBinary"/>.</exception>
        /// <exception cref="DivideByZeroException">This series is zero.</exception>
        public (NDPolyBase quo, NDPolyBase rem) __rdivmod__(object other)
            => ((NDPolyBase, NDPolyBase))Binary(PyBinOp.DivMod, other, this);

        // ---------------------------------------------------------------------------------------------
        //  The base type's operators (a concrete class declares the same set typed with itself)
        // ---------------------------------------------------------------------------------------------

        /// <summary>NumPy's <c>-p</c>.</summary><param name="a">The series.</param><returns>The negated series.</returns>
        public static NDPolyBase operator -(NDPolyBase a) => a.Negative();

        /// <summary>NumPy's <c>+p</c>: the series itself (<c>__pos__</c> returns self).</summary>
        /// <param name="a">The series.</param><returns><paramref name="a"/>.</returns>
        public static NDPolyBase operator +(NDPolyBase a) => a;

        /// <summary>NumPy's <c>p + other</c> (see <see cref="PyBinary"/>).</summary>
        /// <param name="a">The series.</param><param name="b">A series of the same class, a number, an array or a
        ///     sequence of coefficients.</param><returns>The sum.</returns>
        public static NDPolyBase operator +(NDPolyBase a, object b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDPolyBase, object)"/>
        public static NDPolyBase operator +(NDPolyBase a, NDArray b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDPolyBase, object)"/>
        public static NDPolyBase operator +(NDPolyBase a, object[] b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <summary>
        ///     NumPy's <c>p + seq</c> for a C# array (<c>p + new[] { 1.0, 2.0 }</c>, NumPy's <c>p + [1, 2]</c>): an
        ///     overload of its own because C# would otherwise convert the array to an <see cref="NDArray"/> at the call
        ///     site and bind <see cref="operator +(NDPolyBase, NDArray)"/>, leaving that temporary to the garbage
        ///     collector. The array is read as NumPy reads an ndarray operand.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The coefficients.</param><returns>The sum.</returns>
        public static NDPolyBase operator +(NDPolyBase a, Array b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <summary>
        ///     NumPy's <c>p + "text"</c>, a TypeError as in NumPy: an overload of its own so the C# string reaches the
        ///     protocol as Python's str, not through NumSharp's string-to-char-array conversion (whose temporary would be
        ///     left to the garbage collector). Footgun: <c>p + null</c> is ambiguous between this overload and the
        ///     <c>object[]</c> one; NumPy's <c>p + None</c> is <c>p + (object)null</c>.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The text.</param><returns>Never: NumPy's TypeError.</returns>
        public static NDPolyBase operator +(NDPolyBase a, string b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <summary>NumPy's <c>other + p</c>: the other operand declines, then the series' <c>__radd__</c>.</summary>
        /// <param name="a">The other operand.</param><param name="b">The series.</param><returns>The sum.</returns>
        public static NDPolyBase operator +(NDArray a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator +(Array a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator +(object[] a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator +(IList a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator +(ITuple a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);
        /// <inheritdoc cref="operator +(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator +(string a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Add, a, b);

        /// <summary>NumPy's <c>p - other</c> (see <see cref="PyBinary"/>).</summary>
        /// <param name="a">The series.</param><param name="b">A series of the same class, a number, an array or a
        ///     sequence of coefficients.</param><returns>The difference.</returns>
        public static NDPolyBase operator -(NDPolyBase a, object b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDPolyBase, object)"/>
        public static NDPolyBase operator -(NDPolyBase a, NDArray b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDPolyBase, object)"/>
        public static NDPolyBase operator -(NDPolyBase a, object[] b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <summary>
        ///     NumPy's <c>p - seq</c> for a C# array (<c>p - new[] { 1.0, 2.0 }</c>, NumPy's <c>p - [1, 2]</c>): an
        ///     overload of its own because C# would otherwise convert the array to an <see cref="NDArray"/> at the call
        ///     site and bind <see cref="operator -(NDPolyBase, NDArray)"/>, leaving that temporary to the garbage
        ///     collector. The array is read as NumPy reads an ndarray operand.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The coefficients.</param><returns>The difference.</returns>
        public static NDPolyBase operator -(NDPolyBase a, Array b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <summary>
        ///     NumPy's <c>p - "text"</c>, a TypeError as in NumPy: an overload of its own so the C# string reaches the
        ///     protocol as Python's str, not through NumSharp's string-to-char-array conversion (whose temporary would be
        ///     left to the garbage collector). Footgun: <c>p - null</c> is ambiguous between this overload and the
        ///     <c>object[]</c> one; NumPy's <c>p - None</c> is <c>p - (object)null</c>.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The text.</param><returns>Never: NumPy's TypeError.</returns>
        public static NDPolyBase operator -(NDPolyBase a, string b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <summary>NumPy's <c>other - p</c>: the series' <c>__rsub__</c>.</summary>
        /// <param name="a">The other operand.</param><param name="b">The series.</param><returns>The difference.</returns>
        public static NDPolyBase operator -(NDArray a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator -(Array a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator -(object[] a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator -(IList a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator -(ITuple a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);
        /// <inheritdoc cref="operator -(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator -(string a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Sub, a, b);

        /// <summary>NumPy's <c>p * other</c> (see <see cref="PyBinary"/>).</summary>
        /// <param name="a">The series.</param><param name="b">A series of the same class, a number, an array or a
        ///     sequence of coefficients.</param><returns>The product.</returns>
        public static NDPolyBase operator *(NDPolyBase a, object b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDPolyBase, object)"/>
        public static NDPolyBase operator *(NDPolyBase a, NDArray b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDPolyBase, object)"/>
        public static NDPolyBase operator *(NDPolyBase a, object[] b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <summary>
        ///     NumPy's <c>p * seq</c> for a C# array (<c>p * new[] { 1.0, 2.0 }</c>, NumPy's <c>p * [1, 2]</c>): an
        ///     overload of its own because C# would otherwise convert the array to an <see cref="NDArray"/> at the call
        ///     site and bind <see cref="operator *(NDPolyBase, NDArray)"/>, leaving that temporary to the garbage
        ///     collector. The array is read as NumPy reads an ndarray operand.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The coefficients.</param><returns>The product.</returns>
        public static NDPolyBase operator *(NDPolyBase a, Array b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <summary>
        ///     NumPy's <c>p * "text"</c>, a TypeError as in NumPy: an overload of its own so the C# string reaches the
        ///     protocol as Python's str, not through NumSharp's string-to-char-array conversion (whose temporary would be
        ///     left to the garbage collector). Footgun: <c>p * null</c> is ambiguous between this overload and the
        ///     <c>object[]</c> one; NumPy's <c>p * None</c> is <c>p * (object)null</c>.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The text.</param><returns>Never: NumPy's TypeError.</returns>
        public static NDPolyBase operator *(NDPolyBase a, string b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <summary>NumPy's <c>other * p</c>: the series' <c>__rmul__</c>.</summary>
        /// <param name="a">The other operand.</param><param name="b">The series.</param><returns>The product.</returns>
        public static NDPolyBase operator *(NDArray a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator *(Array a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator *(object[] a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator *(IList a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator *(ITuple a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);
        /// <inheritdoc cref="operator *(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator *(string a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mul, a, b);

        /// <summary>NumPy's <c>p / other</c>: only a number divides (<c>p // number</c>).</summary>
        /// <param name="a">The series.</param><param name="b">A Python or NumPy number (not a bool).</param>
        /// <returns>The quotient.</returns>
        public static NDPolyBase operator /(NDPolyBase a, object b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDPolyBase, object)"/>
        public static NDPolyBase operator /(NDPolyBase a, NDArray b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDPolyBase, object)"/>
        public static NDPolyBase operator /(NDPolyBase a, object[] b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <summary>
        ///     NumPy's <c>p / seq</c> for a C# array (<c>p / new[] { 1.0, 2.0 }</c>, NumPy's <c>p / [1, 2]</c>): an
        ///     overload of its own because C# would otherwise convert the array to an <see cref="NDArray"/> at the call
        ///     site and bind <see cref="operator /(NDPolyBase, NDArray)"/>, leaving that temporary to the garbage
        ///     collector. The array is read as NumPy reads an ndarray operand.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The coefficients.</param><returns>The quotient.</returns>
        public static NDPolyBase operator /(NDPolyBase a, Array b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <summary>
        ///     NumPy's <c>p / "text"</c>, a TypeError as in NumPy: an overload of its own so the C# string reaches the
        ///     protocol as Python's str, not through NumSharp's string-to-char-array conversion (whose temporary would be
        ///     left to the garbage collector). Footgun: <c>p / null</c> is ambiguous between this overload and the
        ///     <c>object[]</c> one; NumPy's <c>p / None</c> is <c>p / (object)null</c>.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The text.</param><returns>Never: NumPy's TypeError.</returns>
        public static NDPolyBase operator /(NDPolyBase a, string b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <summary>NumPy's <c>other / p</c>: always CPython's TypeError (<c>__rtruediv__</c> declines every operand).</summary>
        /// <param name="a">The other operand.</param><param name="b">The series.</param><returns>Never returns.</returns>
        public static NDPolyBase operator /(NDArray a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator /(Array a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator /(object[] a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator /(IList a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator /(ITuple a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);
        /// <inheritdoc cref="operator /(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator /(string a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.TrueDiv, a, b);

        /// <summary>NumPy's <c>p % other</c>: the remainder of <c>divmod(p, other)</c>.</summary>
        /// <param name="a">The series.</param><param name="b">The divisor.</param><returns>The remainder.</returns>
        public static NDPolyBase operator %(NDPolyBase a, object b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDPolyBase, object)"/>
        public static NDPolyBase operator %(NDPolyBase a, NDArray b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDPolyBase, object)"/>
        public static NDPolyBase operator %(NDPolyBase a, object[] b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <summary>
        ///     NumPy's <c>p % seq</c> for a C# array (<c>p % new[] { 1.0, 2.0 }</c>, NumPy's <c>p % [1, 2]</c>): an
        ///     overload of its own because C# would otherwise convert the array to an <see cref="NDArray"/> at the call
        ///     site and bind <see cref="operator %(NDPolyBase, NDArray)"/>, leaving that temporary to the garbage
        ///     collector. The array is read as NumPy reads an ndarray operand.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The coefficients.</param><returns>The remainder.</returns>
        public static NDPolyBase operator %(NDPolyBase a, Array b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <summary>
        ///     NumPy's <c>p % "text"</c>, a TypeError as in NumPy: an overload of its own so the C# string reaches the
        ///     protocol as Python's str, not through NumSharp's string-to-char-array conversion (whose temporary would be
        ///     left to the garbage collector). Footgun: <c>p % null</c> is ambiguous between this overload and the
        ///     <c>object[]</c> one; NumPy's <c>p % None</c> is <c>p % (object)null</c>.
        /// </summary>
        /// <param name="a">The series.</param><param name="b">The text.</param><returns>Never: NumPy's TypeError.</returns>
        public static NDPolyBase operator %(NDPolyBase a, string b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <summary>NumPy's <c>other % p</c>: the series' <c>__rmod__</c> (a str's <c>%</c> formats first).</summary>
        /// <param name="a">The dividend.</param><param name="b">The series.</param><returns>The remainder.</returns>
        public static NDPolyBase operator %(NDArray a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator %(Array a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator %(object[] a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator %(IList a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator %(ITuple a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
        /// <inheritdoc cref="operator %(NDArray, NDPolyBase)"/>
        public static NDPolyBase operator %(string a, NDPolyBase b) => (NDPolyBase)Binary(PyBinOp.Mod, a, b);
    }
}
