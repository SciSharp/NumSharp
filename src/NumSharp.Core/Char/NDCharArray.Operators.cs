using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Strings;
using NumSharp.Generic;

// =============================================================================
// NDCharArray.Operators.cs — chararray.__eq__ / __ne__ / __lt__ / __le__ / __gt__ / __ge__ / __add__ / __radd__ /
// __mul__ / __rmul__ / __mod__ / __rmod__ (numpy/_core/defchararray.py) with Python's binary-operator dispatch
// =============================================================================
//
// NumPy's methods, one line each:
//     __eq__(self, other)  -> equal(self, other)          (np.char.equal: compare_chararrays(..., '==', True), STRIPS)
//     ... and so on for != < <= > >=
//     __add__(self, other) -> add(self, other)            (np.add: a ufunc; __array_wrap__ makes the text a chararray view)
//     __radd__(self, other)-> add(other, self)
//     __mul__(self, i)     -> asarray(multiply(self, i))  (np.char.multiply, ValueError for a non-integer count)
//     __rmul__(self, i)    -> asarray(multiply(self, i))
//     __mod__(self, i)     -> asarray(mod(self, i))       (np.char.mod: CPython's % per element, over the STORED text)
//     __rmod__(self, other)-> NotImplemented              (so `x % c` is answered by x alone: a str / bytes formats c as
//                                                          its one argument, an ndarray runs np.remainder — no text loop,
//                                                          NumPy's TypeError —, anything else is CPython's TypeError)
//
// PYTHON'S DISPATCH decides which method runs, and NumSharp replays it:
//   * a chararray on the LEFT: its method first;
//   * a chararray on the RIGHT: its REFLECTED method first when the left operand is a plain ndarray (a subclass's
//     reflected method wins), else after the left operand's own method gave up (str, bytes, a list and an int return
//     NotImplemented against an array). Either way the chararray's reflected method computes: `u == c` strips;
//   * np.char.equal returns NotImplemented for an S / U mix. Python then asks the OTHER operand: a plain ndarray answers
//     with its own (unstripped) comparison — `c == s` is an all-False array, `c < s` NumPy's ufunc no-loop TypeError —
//     and when nothing answers, `==` is False, `!=` True and an ordering raises
//     TypeError("'<' not supported between instances of 'chararray' and 'bytes'");
//   * an element stand-in (c[0], NumPy's Python str / bytes) is not a chararray operand: an expression with no real
//     chararray in it computes as the plain arrays do.
//
// THE STATIC TYPE DECIDES (the NDMatrix rule). These operators bind when an operand is TYPED NDCharArray; a chararray
// object behind an NDArray reference uses NDArray's operators (no stripping). Seven forms per operator — (chararray,
// chararray), (chararray, array), (array, chararray), (chararray, object), (object, chararray), (chararray, string),
// (string, chararray) — so an expression with a chararray on either side, against any operand C# can spell, binds
// here without an ambiguity against NDArray's own forms.
//
// THE C# null LITERAL keeps NDArray's meaning for == and != (a reference test, 0-d result); an ordering against null is
// NumPy's TypeError for None. For + it is None too: np.add's object loop copies a 0-d or short 1-d chararray into object
// as a chararray (the finalize error), and a larger one reaches CPython's concatenation error on its first element; for *
// it is a count that is not an integer (np.char.multiply's ValueError); for % it is a value ('%s' % None is 'None') on the
// right of a chararray, and None.__mod__ does not exist on the left (CPython's TypeError).
//
// % WITH A str / bytes ON THE LEFT is Python's own string formatting, not the chararray's: str.__mod__ formats the
// chararray as ONE argument object (a mapping: '%s' is str(c) — "['a' 'b']" —, '%r' its repr, '%d' int(c), which only a
// 0-d array has, '%(k)s' NumPy's IndexError), and the result is a Python str / bytes, here a read-only 0-d stand-in (the
// form + gives a Python str result too).
//
// A RESULT THE CLASS CANNOT HOLD: chararray + StringDType is a PLAIN StringDType array in NumPy (__array_wrap__ keeps a
// non-text result an ndarray) once the trivial loop does not copy the chararray; the operator's NDCharArray type cannot
// return it, so it raises NotSupportedException naming np.add (the string oracle's one excused refusal).
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>The six comparisons, in the order <see cref="ReflectedComparison"/> pairs them.</summary>
        private enum ComparisonKind
        {
            /// <summary><c>==</c> (<c>np.char.equal</c>).</summary>
            Equal,
            /// <summary><c>!=</c> (<c>np.char.not_equal</c>).</summary>
            NotEqual,
            /// <summary><c>&lt;</c> (<c>np.char.less</c>).</summary>
            Less,
            /// <summary><c>&lt;=</c> (<c>np.char.less_equal</c>).</summary>
            LessEqual,
            /// <summary><c>&gt;</c> (<c>np.char.greater</c>).</summary>
            Greater,
            /// <summary><c>&gt;=</c> (<c>np.char.greater_equal</c>).</summary>
            GreaterEqual,
        }

        #region Comparison core

        /// <summary>The ufunc name <c>compare_chararrays</c> runs for a comparison.</summary>
        /// <param name="kind">The comparison.</param>
        /// <returns><c>equal</c>, <c>not_equal</c>, <c>less</c>, <c>less_equal</c>, <c>greater</c> or <c>greater_equal</c>.</returns>
        private static string UfuncOf(ComparisonKind kind) => kind switch
        {
            ComparisonKind.Equal => "equal",
            ComparisonKind.NotEqual => "not_equal",
            ComparisonKind.Less => "less",
            ComparisonKind.LessEqual => "less_equal",
            ComparisonKind.Greater => "greater",
            _ => "greater_equal",
        };

        /// <summary>The Python operator symbol of a comparison, as its <c>TypeError</c> names it.</summary>
        /// <param name="kind">The comparison.</param>
        /// <returns>The symbol.</returns>
        private static string SymbolOf(ComparisonKind kind) => kind switch
        {
            ComparisonKind.Equal => "==",
            ComparisonKind.NotEqual => "!=",
            ComparisonKind.Less => "<",
            ComparisonKind.LessEqual => "<=",
            ComparisonKind.Greater => ">",
            _ => ">=",
        };

        /// <summary>
        ///     The reflected comparison Python calls on the RIGHT operand (<c>a &lt; b</c> asks <c>b.__gt__(a)</c>):
        ///     <c>==</c> and <c>!=</c> reflect to themselves, the orderings to their mirror.
        /// </summary>
        /// <param name="kind">The comparison as written.</param>
        /// <returns>The comparison the right operand's method makes.</returns>
        private static ComparisonKind ReflectedComparison(ComparisonKind kind) => kind switch
        {
            ComparisonKind.Less => ComparisonKind.Greater,
            ComparisonKind.LessEqual => ComparisonKind.GreaterEqual,
            ComparisonKind.Greater => ComparisonKind.Less,
            ComparisonKind.GreaterEqual => ComparisonKind.LessEqual,
            _ => kind,
        };

        /// <summary>Whether <paramref name="value"/> is a chararray OPERAND (not an element's scalar stand-in).</summary>
        /// <param name="value">The operand as the operator received it.</param>
        /// <param name="chars">The chararray when the answer is true.</param>
        /// <returns>True for a chararray object of any rank that is not a stand-in.</returns>
        private static bool IsRealOperand(object value, out NDCharArray chars)
        {
            chars = value as NDCharArray;
            return chars is not null && !chars.IsScalarStandIn;
        }

        /// <summary>
        ///     One comparison operator with a chararray on at least one side, replaying Python's dispatch (see the file
        ///     header): the chararray's stripping method, then the other operand's answer for an <c>S</c> / <c>U</c> mix, then
        ///     Python's fallback.
        /// </summary>
        /// <param name="kind">The comparison as written.</param>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool result: an array, or a 0-d array for Python's <c>False</c> / <c>True</c> fallback and for a
        ///     comparison with the C# <c>null</c> literal.</returns>
        /// <exception cref="TypeError">A non-text operand (<c>comparison of non-string arrays</c>), an ordering with
        ///     nothing to answer it (<c>'&lt;' not supported between instances of 'chararray' and 'bytes'</c>), or the
        ///     ufunc no-loop text of a plain array's ordering against the other string class.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        private static NDArray<bool> CompareOperator(ComparisonKind kind, object left, object right)
        {
            bool leftReal = IsRealOperand(left, out NDCharArray leftChars);
            bool rightReal = IsRealOperand(right, out NDCharArray rightChars);
            if (!leftReal && !rightReal)
                return PlainComparison(kind, left, right);

            // The C# null literal: NDArray's reference test for == / != (an ordering is NumPy's TypeError for None).
            if (left is null || right is null)
            {
                if (kind == ComparisonKind.Equal || kind == ComparisonKind.NotEqual)
                    return BoolScalar((kind == ComparisonKind.Equal) == ReferenceEquals(left, right));
                throw new TypeError("comparison of non-string arrays");
            }

            NDArray result;
            if (leftReal)
            {
                // self.__op__(other) -> np.char.<op>(self, other).
                result = StrippingComparison(UfuncOf(kind), leftChars, right);
                if (result is not null)
                    return AsBool(result);
                // NotImplemented (S against U): Python asks the right operand's reflected method next. A plain ndarray
                // compares without stripping; another chararray answers NotImplemented too, and so does a Python str /
                // bytes (an element stand-in): str.__eq__ knows no array.
                if (IsPlainArrayOperand(right, out NDArray plainRight))
                    return PlainComparison(ReflectedComparison(kind), plainRight, (NDArray)left);
                return Fallback(kind, left, right);
            }

            // The chararray is on the right: its reflected method computes (before a plain ndarray's own method, and after
            // a str / bytes / list / int on the left gave up).
            result = StrippingComparison(UfuncOf(ReflectedComparison(kind)), rightChars, left);
            if (result is not null)
                return AsBool(result);
            if (IsPlainArrayOperand(left, out NDArray plainLeft))
                return PlainComparison(kind, plainLeft, (NDArray)right);
            return Fallback(kind, left, right);
        }

        /// <summary>
        ///     Whether an operand is a PLAIN ndarray to Python's dispatch: an <see cref="NDArray"/> that is neither a
        ///     chararray nor an element stand-in (the stand-in is NumPy's Python <c>str</c> / <c>bytes</c>, whose rich
        ///     comparison answers <c>NotImplemented</c> against an array).
        /// </summary>
        /// <param name="value">The operand.</param>
        /// <param name="array">The array when the answer is true.</param>
        /// <returns>True for a plain array.</returns>
        private static bool IsPlainArrayOperand(object value, out NDArray array)
        {
            array = value as NDArray;
            return array is not null && array is not NDCharArray;
        }

        /// <summary>
        ///     <c>np.char.&lt;op&gt;(chars, other)</c>: <c>compare_chararrays(chars, other, op, rstrip=True)</c> with the
        ///     other operand converted as NumPy's <c>np.asarray</c> converts it (a string is <c>str</c>, a <c>byte[]</c>
        ///     <c>bytes</c>, a list a text array).
        /// </summary>
        /// <param name="ufunc">The comparison's ufunc name.</param>
        /// <param name="chars">The chararray (the method's <c>self</c>, always the comparison's first operand).</param>
        /// <param name="other">The other operand (not null).</param>
        /// <returns>The bool result, or null for NumPy's <c>NotImplemented</c> (an <c>S</c> / <c>U</c> mix).</returns>
        /// <exception cref="TypeError">The other operand is not <c>S</c> / <c>U</c> text.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        private static NDArray StrippingComparison(string ufunc, NDCharArray chars, object other)
        {
            if (other is NDArray array)
                return NDStringComparison.CompareChararrays(ufunc, chars, array, rstrip: true);
            // A converted operand is this call's temporary: released once the comparison is computed.
            using var converted = CharModule.TextOperand(other, "other");
            return NDStringComparison.CompareChararrays(ufunc, chars, converted, rstrip: true);
        }

        /// <summary>
        ///     The plain (unstripped) comparison of the NDArray operators, with <paramref name="left"/> on the left: what a
        ///     plain ndarray answers, and what an expression whose only chararray operands are element stand-ins computes.
        /// </summary>
        /// <param name="kind">The comparison.</param>
        /// <param name="left">The left operand (an array, or a value the NDArray operators convert).</param>
        /// <param name="right">The right operand.</param>
        /// <returns>NDArray's comparison result.</returns>
        /// <exception cref="Exception">Whatever the NDArray operator raises (a no-loop TypeError, a broadcast error).</exception>
        private static NDArray<bool> PlainComparison(ComparisonKind kind, object left, object right)
        {
            if (left is NDArray l)
            {
                return kind switch
                {
                    ComparisonKind.Equal => l == right,
                    ComparisonKind.NotEqual => l != right,
                    ComparisonKind.Less => l < right,
                    ComparisonKind.LessEqual => l <= right,
                    ComparisonKind.Greater => l > right,
                    _ => l >= right,
                };
            }

            // One operand is an NDArray: the operator's own chararray-typed operand (a stand-in here).
            var r = (NDArray)right;
            return kind switch
            {
                ComparisonKind.Equal => left == r,
                ComparisonKind.NotEqual => left != r,
                ComparisonKind.Less => left < r,
                ComparisonKind.LessEqual => left <= r,
                ComparisonKind.Greater => left > r,
                _ => left >= r,
            };
        }

        /// <summary>
        ///     A plain ndarray's own comparison with the chararray (Python's other method once the chararray's answered
        ///     <c>NotImplemented</c>): NDArray's operator of two arrays, so the chararray is compared as the plain string array
        ///     it is (no stripping) and no public function sees it at the boundary.
        /// </summary>
        /// <param name="kind">The comparison, as the plain array's method makes it (<paramref name="left"/> first).</param>
        /// <param name="left">The plain array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>NDArray's comparison result.</returns>
        /// <exception cref="Exception">Whatever the NDArray operator raises (a no-loop TypeError, a broadcast error).</exception>
        private static NDArray<bool> PlainComparison(ComparisonKind kind, NDArray left, NDArray right) => kind switch
        {
            ComparisonKind.Equal => left == right,
            ComparisonKind.NotEqual => left != right,
            ComparisonKind.Less => left < right,
            ComparisonKind.LessEqual => left <= right,
            ComparisonKind.Greater => left > right,
            _ => left >= right,
        };

        /// <summary>
        ///     Python's answer when neither operand's method computes a comparison: <c>==</c> is <c>False</c>, <c>!=</c>
        ///     <c>True</c> (the identity test of two different objects), an ordering a <c>TypeError</c> naming both types.
        /// </summary>
        /// <param name="kind">The comparison as written.</param>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>A 0-d bool for <c>==</c> / <c>!=</c>.</returns>
        /// <exception cref="TypeError">An ordering: <c>'&lt;' not supported between instances of 'X' and 'Y'</c>.</exception>
        private static NDArray<bool> Fallback(ComparisonKind kind, object left, object right)
        {
            if (kind == ComparisonKind.Equal)
                return BoolScalar(ReferenceEquals(left, right));
            if (kind == ComparisonKind.NotEqual)
                return BoolScalar(!ReferenceEquals(left, right));
            throw new TypeError($"'{SymbolOf(kind)}' not supported between instances of '{PythonTypeName(left)}' and " +
                                $"'{PythonTypeName(right)}'");
        }

        /// <summary>
        ///     The Python type name of an operator operand, as CPython's binary-operator <c>TypeError</c> spells it: a
        ///     chararray is <c>chararray</c>, an element stand-in the <c>str</c> / <c>bytes</c> it stands for, a C# string
        ///     <c>str</c>, a <c>byte[]</c> <c>bytes</c>, a <c>decimal</c> <c>float</c>, a <see cref="System.Numerics.Complex"/>
        ///     <c>complex</c>, a tuple <c>tuple</c>, a dictionary <c>dict</c>, a sequence <c>list</c>.
        /// </summary>
        /// <param name="value">The operand.</param>
        /// <returns>The name.</returns>
        private static string PythonTypeName(object value) => value switch
        {
            null => "NoneType",
            NDCharArray s when s.IsScalarStandIn => s.dtype.@char == 'S' ? "bytes" : "str",
            NDCharArray => "chararray",
            NDArray => "numpy.ndarray",
            string => "str",
            byte[] => "bytes",
            bool => "bool",
            ITuple => "tuple",
            sbyte or byte or short or ushort or int or uint or long or ulong or System.Numerics.BigInteger => "int",
            // NumSharp's decimal stands for a Python float wherever a C# value is a Python object (NDPyObject.TypeName).
            float or double or decimal => "float",
            System.Numerics.Complex => "complex",
            // A dict is enumerable too: tested before the sequences.
            System.Collections.IDictionary => "dict",
            System.Collections.IEnumerable => "list",
            _ => value.GetType().Name,
        };

        /// <summary>A comparison result as the operators' static type (the driver returns a fresh bool array).</summary>
        /// <param name="result">The bool array, which this call owns: an untyped one is released here.</param>
        /// <returns>The typed array: <paramref name="result"/> itself when it already is one, else a typed alias of it.</returns>
        /// <exception cref="InvalidOperationException">The result is not bool (a driver change; never in practice).</exception>
        /// <remarks>The typed array is a second object with its own counted reference on the buffer, so the driver's untyped
        ///     wrapper is disposed rather than left to the finalizer (the leak sweep measured one stranded buffer per
        ///     comparison when it was not). <c>AsGeneric</c>, not <c>MakeGeneric</c>: the typed array shares the result's
        ///     storage object, so it OWNS the data as NumPy's comparison result does (a <c>MakeGeneric</c> alias reports
        ///     OWNDATA False); disposing the untyped wrapper releases only that wrapper's reference
        ///     (<see cref="NDArray.Dispose"/> touches nothing else of the storage). The untyped case is tested first because
        ///     <c>AsGeneric</c> returns its receiver itself when that already is typed, which the <c>using</c> would
        ///     dispose.</remarks>
        private static NDArray<bool> AsBool(NDArray result)
        {
            if (result is NDArray<bool> typed)
                return typed;
            using (result)
                return result.AsGeneric<bool>() ?? throw new InvalidOperationException("a chararray comparison produced a non-bool result");
        }

        /// <summary>A 0-d bool for Python's <c>True</c> / <c>False</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>A new 0-d bool array.</returns>
        private static NDArray<bool> BoolScalar(bool value)
        {
            // The untyped scalar is this method's temporary: the typed alias keeps its own reference on the buffer.
            using var scalar = Scalar<bool>(value);
            return scalar.MakeGeneric<bool>();
        }

        #endregion

        #region Concatenation and repetition core

        /// <summary>
        ///     <c>left + right</c> with a chararray on at least one side: <c>chararray.__add__</c> / <c>__radd__</c>, i.e.
        ///     <c>np.add</c> in the written operand order, whose text result is a chararray VIEW of the ufunc's output.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The concatenation as a chararray. With no real chararray operand (element stand-ins only), the plain sum:
        ///     a 0-d result is a new stand-in (NumPy's Python <c>str</c>), an array a chararray over it.</returns>
        /// <exception cref="TypeError">A bytes / str mix or a number operand (NumPy's ufunc texts); the C# <c>null</c>
        ///     (NumPy's <c>None</c>) beside a chararray NumPy's trivial loop does not copy (larger than 1-d, or 1-d with more
        ///     than <c>np.getbufsize()</c> elements): CPython's error on the first pair (<c>can only concatenate str (not
        ///     "NoneType") to str</c>, <c>can't concat NoneType to bytes</c>, or for <c>None + c</c> <c>unsupported operand
        ///     type(s) for +: 'NoneType' and 'str'</c>).</exception>
        /// <exception cref="ValueError">The finalize error (<c>Can only create a chararray from string data.</c>): a
        ///     <c>StringDType</c> or <c>null</c> operand beside a 0-d or short 1-d chararray, which NumPy's trivial loop copies
        ///     into the loop's dtype as a chararray.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand beside a chararray NumPy does not copy:
        ///     NumPy then returns a PLAIN StringDType array, which this operator's <see cref="NDCharArray"/> type cannot hold
        ///     (<c>np.add</c> returns it); or <c>null</c> beside an EMPTY chararray NumPy does not copy (NumPy's empty object
        ///     array: NumSharp has no object dtype).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        private static NDCharArray AddOperator(object left, object right)
        {
            bool leftReal = IsRealOperand(left, out NDCharArray leftChars);
            bool rightReal = IsRealOperand(right, out NDCharArray rightChars);
            if (!leftReal && !rightReal)
                return FromPlainOperator(PlainAdd(left, right));
            // A null operand is NumPy's None: np.add resolves the object loop, whose trivial-loop copy of a short chararray
            // into object meets the finalize rule; a longer one reaches CPython's own concatenation error on its first
            // element (NumSharp has no object dtype to compute anything else).
            if (left is null || right is null)
                throw NoneConcatenation(leftReal ? leftChars : rightChars, noneOnLeft: left is null);
            // __add__(self, other) = add(self, other); __radd__(self, other) = add(other, self): both are np.char.add in
            // the written order. np.add's chararray rule hands the text back as a chararray view; inside a boundary region
            // (an operator used by a member that computes plain) the result is plain, and __array_wrap__'s view of it is
            // the same answer.
            NDArray sum = np.@char.add(left, right);
            if (sum is NDCharArray chars)
                return chars;
            // A result that is not text: a StringDType operand makes np.add's loop StringDType, and NumPy's chararray.__add__
            // returns that plain StringDType array (__array_wrap__ keeps a non-text result an ndarray). The operator's static
            // NDCharArray type cannot hold it; np.add returns it.
            if (!AcceptsDType(sum.dtype) || sum.dtype.@char == 'V')
                throw new NotSupportedException(
                    $"chararray + {(left is NDCharArray ? DescribeOperand(right) : DescribeOperand(left))}: NumPy returns a plain " +
                    $"{sum.dtype} array, which the NDCharArray result of this operator cannot hold; call np.add for it.");
            return ViewOf(sum);
        }

        /// <summary>An operand as the refusal of <see cref="AddOperator"/> names it: its dtype, or its C# type.</summary>
        /// <param name="value">The operand.</param>
        /// <returns>The description.</returns>
        private static string DescribeOperand(object value) => value is NDArray a ? $"a {a.dtype} array" : value?.GetType().Name ?? "None";

        /// <summary>
        ///     <c>left * right</c> with a chararray on at least one side: <c>chararray.__mul__</c> / <c>__rmul__</c>, i.e.
        ///     <c>np.char.asarray(np.char.multiply(chars, count))</c> with the chararray first whichever side it is on.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The repetition as a chararray: an owning copy for <c>U</c> data (<c>asarray</c> always copies <c>U</c>),
        ///     a view of the computed array for <c>S</c>. With no real chararray operand, the plain product as
        ///     <see cref="AddOperator"/> returns it.</returns>
        /// <exception cref="ValueError">A count that is not an integer, the C# <c>null</c> (NumPy's <c>None</c>) included
        ///     (<c>Can only multiply by integers</c>), or an empty result (<c>zero-size array to reduction operation maximum
        ///     which has no identity</c>).</exception>
        /// <exception cref="OverflowException">A repetition longer than <c>sys.maxsize</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        private static NDCharArray MultiplyOperator(object left, object right)
        {
            bool leftReal = IsRealOperand(left, out NDCharArray leftChars);
            bool rightReal = IsRealOperand(right, out NDCharArray rightChars);
            if (!leftReal && !rightReal)
                return FromPlainOperator(PlainMultiply(left, right));
            NDCharArray chars = leftReal ? leftChars : rightChars;
            object count = leftReal ? right : left;
            // A null count is NumPy's None: np.strings.multiply's np.asanyarray(None) is an object array, which is not an
            // integer type — its TypeError, re-raised by np.char.multiply.
            if (count is null)
                throw new ValueError("Can only multiply by integers");
            // np.char.multiply's text result is a chararray view (np.strings.multiply's rule); asarray then copies a U
            // result and views an S one, as defchararray's __mul__ does. The scope reclaims the intermediate when asarray
            // made a new object.
            return np.@char.asarray(np.@char.multiply(chars, count));
        }

        /// <summary>
        ///     NumPy's answer for <c>c + None</c> / <c>None + c</c>: <c>np.add</c> runs its object loop, and its trivial loop
        ///     first copies a 0-d or short 1-d chararray into object as a chararray — the finalize error. A larger chararray
        ///     is not copied; the loop then concatenates element by element and CPython refuses the first pair.
        /// </summary>
        /// <param name="chars">The chararray operand.</param>
        /// <param name="noneOnLeft">True for <c>None + c</c>.</param>
        /// <returns>The exception to throw.</returns>
        /// <remarks>An empty large chararray would give NumPy an empty object array: NumSharp has no object dtype, so that
        ///     one case is refused as unsupported.</remarks>
        private static Exception NoneConcatenation(NDCharArray chars, bool noneOnLeft)
        {
            if (chars.ndim == 0 || chars.ndim == 1 && chars.size <= np.getbufsize())
                return new ValueError(FinalizeText);
            if (chars.size == 0)
                return new NotSupportedException("NumPy concatenates None into an empty object array; NumSharp has no object dtype.");
            bool bytes = chars.dtype.@char == 'S';
            if (noneOnLeft)
                return new TypeError($"unsupported operand type(s) for +: 'NoneType' and '{(bytes ? "bytes" : "str")}'");
            return new TypeError(bytes ? "can't concat NoneType to bytes" : "can only concatenate str (not \"NoneType\") to str");
        }

        /// <summary>The NDArray <c>+</c> of two operands, one of which is an NDArray (the stand-in path).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>NDArray's sum.</returns>
        /// <exception cref="Exception">Whatever the NDArray operator raises.</exception>
        private static NDArray PlainAdd(object left, object right)
            => left is NDArray l ? l + right : left + (NDArray)right;

        /// <summary>The NDArray <c>*</c> of two operands, one of which is an NDArray (the stand-in path).</summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>NDArray's product.</returns>
        /// <exception cref="Exception">Whatever the NDArray operator raises.</exception>
        private static NDArray PlainMultiply(object left, object right)
            => left is NDArray l ? l * right : left * (NDArray)right;

        /// <summary>
        ///     The operator's result for an expression whose only chararray operands are element stand-ins: NumPy computes
        ///     with Python <c>str</c> / <c>bytes</c> values there, so a 0-d result is a new stand-in (the Python value) and an
        ///     array result a chararray over the plain result (NumPy returns an ndarray there: the static return type of the
        ///     operator cannot carry it — a documented divergence).
        /// </summary>
        /// <param name="result">The plain result (its wrapper is reclaimed by the caller's scope).</param>
        /// <returns>The chararray.</returns>
        private static NDCharArray FromPlainOperator(NDArray result)
        {
            if (result.ndim == 0 && result.Storage.IsFixedString)
            {
                var scalar = new NDCharArray(result.Storage.Alias()) { IsScalarStandIn = true, TensorEngine = result.TensorEngine };
                scalar.MarkReductionScalar();   // read-only, as the immutable Python value it stands for
                return scalar;
            }
            return new NDCharArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };
        }

        #endregion

        #region Formatting core

        /// <summary>
        ///     <c>left % right</c> with an <see cref="NDCharArray"/> on at least one side, replaying Python's dispatch:
        ///     <c>chararray.__mod__(self, i)</c> is <c>asarray(mod(self, i))</c> and <c>chararray.__rmod__</c> returns
        ///     <c>NotImplemented</c>, so a chararray on the LEFT formats its own elements, and with the chararray on the RIGHT
        ///     only the left operand's own <c>%</c> can answer.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        ///     A chararray on the left: the formatted chararray (<see cref="FormatElements"/>: an owning copy for <c>U</c>, a
        ///     view for <c>S</c>). A Python str / bytes on the left (a C# string or char, a <c>byte[]</c>, an element
        ///     stand-in): the str / bytes CPython's <c>str.__mod__</c> / <c>bytes.__mod__</c> returns, as a new read-only 0-d
        ///     stand-in (<see cref="FormatPythonText"/>).
        /// </returns>
        /// <exception cref="TypeError">
        ///     A chararray on the left: <see cref="np.strings.mod(NDArray, object)"/>'s errors (the argument count, a
        ///     conversion the value refuses). A str / bytes on the left: CPython's formatting errors with the right operand
        ///     as the one argument (<c>%d format: a real number is required, not chararray</c>, <c>only 0-dimensional arrays
        ///     can be converted to Python scalars</c>, <c>not enough arguments for format string</c>). An ndarray or a NumPy
        ///     scalar on the left: <c>np.remainder</c>'s no-loop texts (<c>ufunc 'remainder' not supported for the input
        ///     types, ...</c>). Any other left operand: <c>unsupported operand type(s) for %: 'int' and 'chararray'</c>.
        /// </exception>
        /// <exception cref="ValueError">A malformed format; <c>int()</c> / <c>float()</c> of text that spells no number.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> argument out of range.</exception>
        /// <exception cref="IndexError">A <c>%(key)</c> lookup on an ndarray or a NumPy scalar.</exception>
        /// <exception cref="KeyError">A <c>%(key)</c> missing from a dict.</exception>
        /// <exception cref="IncorrectShapeException">A chararray on the left whose values do not broadcast.</exception>
        /// <remarks>
        ///     The order is Python's. A chararray on the left answers first (with a chararray on the right too: Python asks a
        ///     reflected method first only for a right operand whose class SUBCLASSES the left's, which another chararray does
        ///     not). Then a str or bytes on the left: <c>str</c>'s number slot is asked before the chararray's (the chararray
        ///     is no str subclass), and it formats ANY right operand. Then nothing: <c>__rmod__</c> is NotImplemented, so an
        ///     ndarray on the left runs its own <c>np.remainder</c>, and every other Python object ends in CPython's
        ///     <c>TypeError</c>.
        /// </remarks>
        [NDScoped]
        private static NDCharArray ModOperator(object left, object right)
        {
            if (IsRealOperand(left, out NDCharArray chars))
                return FormatElements(chars, right);
            if (TryPythonText(left, out NDArray format))
                return FormatPythonText(format, right);
            // A C# number or bool literal reaches the (NDArray, chararray) form through NDArray's implicit conversion (C#
            // prefers it to boxing): the 0-d array is marked as the Python int / float / complex / bool it was, whose own %
            // gives up against an array — and so does __rmod__.
            if (left is NDArray { WasPythonLiteral: true } or NDArray { WasPythonBool: true })
            {
                var literal = (NDArray)left;
                object python = literal.WasPythonBool ? true : literal.Nep50Operand;
                throw new TypeError($"unsupported operand type(s) for %: '{PythonTypeName(python)}' and '{PythonTypeName(right)}'");
            }
            // __rmod__ is NotImplemented: an ndarray (or a NumPy scalar: np.float16, a C# char's np.uint16) answers with
            // ndarray.__mod__, which is np.remainder — no loop takes a text operand, so it raises NumPy's ufunc TypeError.
            if (left is NDArray or Half)
                throw RemainderOfArray(left, right);
            throw new TypeError($"unsupported operand type(s) for %: '{PythonTypeName(left)}' and '{PythonTypeName(right)}'");
        }

        /// <summary>
        ///     NumPy's <c>chararray.__mod__(self, i)</c>: <c>asarray(mod(self, i))</c> — every stored element formatted with
        ///     CPython's <c>%</c> against the broadcast element of <paramref name="values"/>, then <c>np.char.asarray</c>'s
        ///     form of the plain result.
        /// </summary>
        /// <param name="chars">The chararray (a real one, not an element stand-in).</param>
        /// <param name="values">The values, as <see cref="np.strings.mod(NDArray, object)"/> converts them.</param>
        /// <returns>An owning chararray for <c>U</c> (<c>asarray</c> copies <c>U</c>: the fresh result is taken over, no copy
        ///     made), a chararray view of the computed array for <c>S</c>.</returns>
        /// <exception cref="Exception"><see cref="np.strings.mod(NDArray, object)"/>'s errors.</exception>
        /// <remarks>The formats are the STORED text: trailing whitespace stays in the format (<c>'%s  ' % 'x'</c> is
        ///     <c>'x  '</c>), as NumPy's <c>np.asarray(self)</c> reads it; only element access strips.</remarks>
        [NDScoped]
        private static NDCharArray FormatElements(NDCharArray chars, object values)
        {
            NDArray fresh;
            // Inside a boundary region the module function computes the plain array (no chararray view of it is made just
            // to be dropped), and asarray then gives the result its final form.
            using (NDCharArrayBoundary.Inside())
                fresh = np.@char.mod(chars, values);
            return AsArrayOfFresh(fresh);
        }

        /// <summary>
        ///     The left operand as the Python str / bytes whose <c>__mod__</c> formats: a C# string or a boxed char (a str), a
        ///     <c>byte[]</c> (bytes), an element stand-in (the str / bytes it stands for), or the 0-d array a C# string /
        ///     Python bytes became through a conversion (marked <see cref="NDArray.WasPythonStr"/> /
        ///     <see cref="NDArray.WasPythonBytes"/>).
        /// </summary>
        /// <param name="value">The left operand.</param>
        /// <param name="format">Receives the 0-d <c>U</c> / <c>S</c> array holding the format (the operand itself, or a new
        ///     array the calling operator's scope reclaims).</param>
        /// <returns>True for a str or bytes.</returns>
        /// <remarks>An UNBOXED C# char binds the (NDArray, chararray) form through NDArray's implicit conversion, which makes
        ///     it NumPy's <c>np.uint16</c> scalar (the house map of a char as an array); only a char reaching the
        ///     <c>object</c> form (boxed) is the one-character str of the string module's object map.</remarks>
        private static bool TryPythonText(object value, out NDArray format)
        {
            switch (value)
            {
                case string s:
                    format = NDArray.FromPythonOperand(s);
                    return true;
                case char c:
                    format = NDArray.FromPythonOperand(c.ToString());
                    return true;
                case byte[] b:
                    format = NDArray.PythonBytesLiteral(b);
                    return true;
                case NDCharArray standIn when standIn.IsScalarStandIn:
                    format = standIn;
                    return true;
                case NDArray marked when marked.WasPythonStr || marked.WasPythonBytes:
                    format = marked;
                    return true;
                default:
                    format = null;
                    return false;
            }
        }

        /// <summary>
        ///     CPython's <c>str.__mod__</c> / <c>bytes.__mod__</c> with <paramref name="right"/> as the ONE argument object:
        ///     Python's answer for <c>'%s' % chararray</c> and for an element stand-in on the left of any operand.
        /// </summary>
        /// <param name="format">The format (a 0-d <c>U</c> / <c>S</c> array: a Python str / bytes).</param>
        /// <param name="right">The argument: a chararray or another ndarray (a mapping: <c>%s</c> is its <c>str()</c>, a
        ///     <c>%(key)</c> lookup NumPy's IndexError, the numbers <c>int(a)</c> / <c>float(a)</c> of a 0-d array), an element
        ///     stand-in (the str / bytes it stands for), a tuple (its items, one per spec), a C# number, string, <c>byte[]</c>,
        ///     dict or null as the Python object it stands for.</param>
        /// <returns>The formatted str / bytes as a new read-only 0-d stand-in.</returns>
        /// <exception cref="Exception">CPython's formatting errors (see <see cref="NDCharObjectPath.ModObject"/>).</exception>
        /// <remarks>A Python str / bytes result keeps every character the formatting writes, trailing NULs included (bytes
        ///     formatting's <c>%s</c> of an <c>S</c> chararray ends with its NUL padding); the 0-d stand-in, like any <c>S</c> /
        ///     <c>U</c> element, reads back without the trailing ones.</remarks>
        [NDScoped]
        private static NDCharArray FormatPythonText(NDArray format, object right)
            => FromPlainOperator(NDCharObjectPath.ModObject(format, PythonObjectOf(right)));

        /// <summary>
        ///     The Python object a right operand stands for when it is formatted as ONE argument object: a 0-d array that a C#
        ///     literal became through NDArray's implicit conversion (C# prefers that conversion to boxing, so
        ///     <c>c[0] % 5</c> binds the (chararray, NDArray) form) is the literal itself again — a Python int / float /
        ///     complex, bool, str or bytes, or a <see cref="Half"/> (np.float16) or char (a one-character str) — and anything
        ///     else is itself.
        /// </summary>
        /// <param name="value">The right operand.</param>
        /// <returns>The object the formatting reads.</returns>
        /// <remarks>The difference is visible: a Python int has no buffer, so <c>b'%s' % 5</c> raises <c>%b requires a
        ///     bytes-like object, or an object that implements __bytes__, not 'int'</c>, where a 0-d int64 array's buffer
        ///     would be written; and <c>'%r' % 5</c> is <c>5</c>, not <c>array(5)</c>.</remarks>
        private static object PythonObjectOf(object value) => value switch
        {
            NDArray { WasPythonLiteral: true } literal => literal.Nep50Operand,
            NDArray { WasPythonBool: true } flag => flag.item(),
            NDArray { WasPythonStr: true } text => text.PythonStrText,
            NDArray { WasPythonBytes: true } bytes => bytes.tolist(),
            NDArray { WasNumPyScalar: true } scalar => scalar.item(),
            _ => value,
        };

        /// <summary>
        ///     <c>ndarray.__mod__(left, right)</c>, i.e. <c>np.remainder(left, right)</c>, for an ndarray (or NumPy scalar) on
        ///     the left of a chararray: NumPy has no remainder loop for text, so this always raises the ufunc's TypeError.
        /// </summary>
        /// <param name="left">The ndarray, or a <see cref="Half"/> (NumPy's np.float16 scalar).</param>
        /// <param name="right">The right operand (a chararray or an element stand-in).</param>
        /// <returns>Never returns the exception for a text operand: <c>np.remainder</c> raises during the call. The
        ///     exception returned (for the caller to throw) reports a result that NumPy's rule forbids.</returns>
        /// <exception cref="TypeError">NumPy's no-loop texts (<c>ufunc 'remainder' not supported for the input types, and
        ///     the inputs could not be safely coerced to any supported types according to the casting rule ''safe''</c>, or
        ///     the <c>StringDType</c> loop's).</exception>
        /// <remarks>Scoped by hand (an <see cref="Exception"/> result is no carrier the scope weaver handles): the 0-d array
        ///     a NumPy scalar becomes is this call's own, released on every path.</remarks>
        private static Exception RemainderOfArray(object left, object right)
        {
            NDArray converted = left is NDArray ? null : NDArray.FromPythonOperand(left);
            try
            {
                // NDArray's own (NDArray, NDArray) operator: np.remainder's engine over the two operands, which raises for
                // the text one. The right operand is a chararray or an element stand-in here (the forms that reach this
                // call all have one on the right), already an array, so no np.asanyarray runs on it.
                NDArray l = converted ?? (NDArray)left;
                using NDArray none = right is NDArray r ? l % r : l % right;
                return new InvalidOperationException("np.remainder computed a result for a chararray operand.");
            }
            finally
            {
                converted?.Dispose();
            }
        }

        #endregion

        #region == and !=

        /// <summary>
        ///     NumPy's <c>chararray.__eq__</c>: <c>np.char.equal(left, right)</c>, element-wise equality once each element
        ///     has lost its trailing NULs and ASCII whitespace (<c>'a '</c> equals <c>'a'</c>), after Python's dispatch
        ///     (see the class's operator notes: a plain array on the other side still strips).
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array of the broadcast shape; a 0-d <c>False</c> when an <c>S</c> chararray meets <c>U</c> text
        ///     with nothing to compare them (Python's fallback).</returns>
        /// <exception cref="TypeError">A non-text operand: <c>comparison of non-string arrays</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator ==(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(NDCharArray left, object right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(object left, NDCharArray right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(NDCharArray left, string right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <inheritdoc cref="operator ==(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator ==(string left, NDCharArray right) => CompareOperator(ComparisonKind.Equal, left, right);

        /// <summary>
        ///     NumPy's <c>chararray.__ne__</c>: <c>np.char.not_equal(left, right)</c> after stripping, with Python's dispatch.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array; a 0-d <c>True</c> for Python's fallback (an <c>S</c> / <c>U</c> mix).</returns>
        /// <exception cref="TypeError">A non-text operand: <c>comparison of non-string arrays</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator !=(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(NDCharArray left, object right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(object left, NDCharArray right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(NDCharArray left, string right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        /// <inheritdoc cref="operator !=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator !=(string left, NDCharArray right) => CompareOperator(ComparisonKind.NotEqual, left, right);

        #endregion

        #region < <= > >=

        /// <summary>
        ///     NumPy's <c>chararray.__lt__</c>: <c>np.char.less(left, right)</c> after stripping, with Python's dispatch (a
        ///     chararray on the right computes its reflected <c>__gt__</c>).
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array of the broadcast shape.</returns>
        /// <exception cref="TypeError">A non-text operand (<c>comparison of non-string arrays</c>); an <c>S</c> / <c>U</c>
        ///     mix (<c>'&lt;' not supported between instances of ...</c>, or a plain array's ufunc no-loop text).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator <(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(NDCharArray left, object right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(object left, NDCharArray right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(NDCharArray left, string right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <inheritdoc cref="operator &lt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <(string left, NDCharArray right) => CompareOperator(ComparisonKind.Less, left, right);

        /// <summary>
        ///     NumPy's <c>chararray.__gt__</c>: <c>np.char.greater(left, right)</c> after stripping, with Python's dispatch.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array of the broadcast shape.</returns>
        /// <exception cref="TypeError">A non-text operand, or an <c>S</c> / <c>U</c> mix nothing compares.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator >(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(NDCharArray left, object right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(object left, NDCharArray right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(NDCharArray left, string right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <inheritdoc cref="operator &gt;(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >(string left, NDCharArray right) => CompareOperator(ComparisonKind.Greater, left, right);

        /// <summary>
        ///     NumPy's <c>chararray.__le__</c>: <c>np.char.less_equal(left, right)</c> after stripping, with Python's dispatch.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array of the broadcast shape.</returns>
        /// <exception cref="TypeError">A non-text operand, or an <c>S</c> / <c>U</c> mix nothing compares.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator <=(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(NDCharArray left, object right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(object left, NDCharArray right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(NDCharArray left, string right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <inheritdoc cref="operator &lt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator <=(string left, NDCharArray right) => CompareOperator(ComparisonKind.LessEqual, left, right);

        /// <summary>
        ///     NumPy's <c>chararray.__ge__</c>: <c>np.char.greater_equal(left, right)</c> after stripping, with Python's
        ///     dispatch.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The bool array of the broadcast shape.</returns>
        /// <exception cref="TypeError">A non-text operand, or an <c>S</c> / <c>U</c> mix nothing compares.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDArray<bool> operator >=(NDCharArray left, NDCharArray right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(NDCharArray left, NDArray right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(NDArray left, NDCharArray right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(NDCharArray left, object right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(object left, NDCharArray right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(NDCharArray left, string right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        /// <inheritdoc cref="operator &gt;=(NDCharArray, NDCharArray)"/>
        public static NDArray<bool> operator >=(string left, NDCharArray right) => CompareOperator(ComparisonKind.GreaterEqual, left, right);

        #endregion

        #region + and *

        /// <summary>
        ///     NumPy's <c>chararray.__add__</c> / <c>__radd__</c>: element-wise concatenation in the written order
        ///     (<c>np.add</c>), the result a chararray view of the computed array (<c>c + "x"</c> keeps trailing whitespace
        ///     inside the stored text: only element access strips).
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The concatenation as a chararray.</returns>
        /// <exception cref="TypeError">A bytes / str mix or a number operand (NumPy's ufunc texts); the C# <c>null</c>
        ///     (NumPy's <c>None</c>) beside a chararray NumPy does not copy: CPython's concatenation error.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> or <c>null</c> operand beside a 0-d or short 1-d chararray (the
        ///     finalize error of NumPy's trivial-loop copy).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand beside a larger chararray: NumPy returns a
        ///     plain StringDType array, which an <see cref="NDCharArray"/> cannot hold (call <c>np.add</c>).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <remarks>Python's dispatch: the chararray's <c>__add__</c> / <c>__radd__</c> answers whichever side it is on, an
        ///     element stand-in (NumPy's <c>str</c> / <c>bytes</c>) is no chararray, and the static rule applies — an
        ///     <see cref="NDArray"/>-typed reference to a chararray binds <see cref="NDArray"/>'s <c>+</c>, plain.</remarks>
        public static NDCharArray operator +(NDCharArray left, NDCharArray right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(NDCharArray left, NDArray right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(NDArray left, NDCharArray right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(NDCharArray left, object right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(object left, NDCharArray right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(NDCharArray left, string right) => AddOperator(left, right);

        /// <inheritdoc cref="operator +(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator +(string left, NDCharArray right) => AddOperator(left, right);

        /// <summary>
        ///     NumPy's <c>chararray.__mul__</c> / <c>__rmul__</c>: every string repeated by the integer count on the other
        ///     side (<c>np.char.asarray(np.char.multiply(chars, count))</c>), whichever side the chararray is on.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>The repetition as a chararray (an owning copy for <c>U</c>, a view for <c>S</c>).</returns>
        /// <exception cref="ValueError">A count that is not an integer, the C# <c>null</c> (NumPy's <c>None</c>) included
        ///     (<c>Can only multiply by integers</c>), or an empty result (<c>zero-size array to reduction operation maximum
        ///     which has no identity</c>).</exception>
        /// <exception cref="OverflowException">A repetition longer than <c>sys.maxsize</c>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        public static NDCharArray operator *(NDCharArray left, NDCharArray right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(NDCharArray left, NDArray right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(NDArray left, NDCharArray right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(NDCharArray left, object right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(object left, NDCharArray right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(NDCharArray left, string right) => MultiplyOperator(left, right);

        /// <inheritdoc cref="operator *(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator *(string left, NDCharArray right) => MultiplyOperator(left, right);

        #endregion

        #region %

        /// <summary>
        ///     NumPy's <c>chararray.__mod__</c>: every element of the left chararray formatted with CPython's <c>%</c> against
        ///     the broadcast element of the right operand (<c>np.char.asarray(np.char.mod(self, values))</c>); with the
        ///     chararray on the RIGHT, the left operand's own <c>%</c> (<c>chararray.__rmod__</c> is NotImplemented).
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        ///     A chararray on the left: the formatted chararray — an owning copy for <c>U</c>, a view of the computed array for
        ///     <c>S</c> — over the broadcast shape (<c>np.@char.array(new[] { "%d", "%x" }) % 255</c> is <c>['255', 'ff']</c>).
        ///     A C# string, a <c>byte[]</c> or an element stand-in on the left: Python's <c>str</c> / <c>bytes</c> formatting
        ///     with the right operand as its ONE argument, the result a read-only 0-d stand-in (<c>"%s" % c</c> is
        ///     <c>str(c)</c>, <c>"['a' 'b']"</c>; <c>c[0] % 5</c> formats the element's text).
        /// </returns>
        /// <exception cref="TypeError">The formatting's errors (the argument count, a conversion the value refuses:
        ///     <c>%d format: a real number is required, not numpy.str_</c>); an ndarray or a NumPy scalar on the left of a
        ///     chararray: <c>np.remainder</c>'s no-loop text; any other left operand: <c>unsupported operand type(s) for %:
        ///     'int' and 'chararray'</c>.</exception>
        /// <exception cref="ValueError">A malformed format (<c>incomplete format</c>); a NaN under an integer conversion;
        ///     <c>int()</c> / <c>float()</c> of text that spells no number.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> argument out of
        ///     range.</exception>
        /// <exception cref="IndexError">A <c>%(key)</c> lookup on a NumPy scalar or an ndarray.</exception>
        /// <exception cref="KeyError">A <c>%(key)</c> missing from a dict.</exception>
        /// <exception cref="IncorrectShapeException">The values do not broadcast against the chararray.</exception>
        /// <remarks>Python's dispatch: the chararray's <c>__mod__</c> answers when it is on the left; on the right only a str /
        ///     bytes (which formats any object) or an ndarray (<c>np.remainder</c>, which raises) has a <c>%</c> for it. The
        ///     static rule applies, as for <c>+</c>: an <see cref="NDArray"/>-typed reference to a chararray binds
        ///     <see cref="NDArray"/>'s <c>%</c> (<c>np.remainder</c>).</remarks>
        public static NDCharArray operator %(NDCharArray left, NDCharArray right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(NDCharArray left, NDArray right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(NDArray left, NDCharArray right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(NDCharArray left, object right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(object left, NDCharArray right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(NDCharArray left, string right) => ModOperator(left, right);

        /// <inheritdoc cref="operator %(NDCharArray, NDCharArray)"/>
        public static NDCharArray operator %(string left, NDCharArray right) => ModOperator(left, right);

        #endregion

        #region Equality members (declared because this class declares ==)

        /// <summary>Value equality, as for <see cref="NDArray"/>: true when every element compares equal (unstripped) after
        ///     broadcasting. Declared because this class declares <c>==</c>.</summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><c>true</c> when all elements are equal; <c>false</c> for <c>null</c>.</returns>
        public override bool Equals(object obj) => base.Equals(obj);

        /// <summary>Always throws: an array is mutable and therefore unhashable, as in NumPy.</summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override int GetHashCode() => base.GetHashCode();

        #endregion
    }
}
