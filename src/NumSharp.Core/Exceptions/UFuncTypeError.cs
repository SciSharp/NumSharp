using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>numpy._core._exceptions.UFuncTypeError</c>: the base of the errors a ufunc raises when it cannot run
    ///     for its operands' dtypes — no loop for them (<see cref="UFuncNoLoopError"/>), two operands it cannot combine
    ///     (<see cref="UFuncBinaryResolutionError"/>), or a cast its <c>casting=</c> rule forbids
    ///     (<see cref="UFuncInputCastingError"/>, <see cref="UFuncOutputCastingError"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A subclass of <see cref="TypeError"/>, as NumPy's is, so <c>catch (TypeError)</c> catches every ufunc refusal
    ///     (NumPy advises catching <c>TypeError</c> or <c>UFuncTypeError</c>: its subclasses are private and its traceback
    ///     shows each as <c>UFuncTypeError</c>). NumSharp names the subclasses without NumPy's leading underscore, since
    ///     a C# class cannot hide behind <c>_display_as_base</c>; footgun: an MSTest <c>Assert.ThrowsException&lt;TypeError&gt;</c>
    ///     (an exact-type check) no longer matches a ufunc refusal, which now throws the subclass.
    ///     </para>
    ///     <para>
    ///     <see cref="ufunc"/> is the ufunc's NAME (<c>add</c>): NumSharp has no ufunc objects, and NumPy's errors only ever
    ///     print the name. A dtype-like value (<see cref="UFuncNoLoopError.dtypes"/>, <see cref="UFuncCastingError.from_"/>,
    ///     <see cref="UFuncCastingError.to"/>) is a <see cref="DType"/> instance (printed <c>dtype('float64')</c>), a
    ///     <see cref="DTypeMeta"/> class (printed <c>&lt;class 'numpy.dtypes.StrDType'&gt;</c>) or null (<c>None</c>), as
    ///     NumPy's resolver hands them over; where NumPy names its object dtype, which NumSharp has no descriptor for, it is
    ///     a placeholder whose <see cref="object.ToString"/> is NumPy's text.
    ///     </para>
    ///     <para>
    ///     Every class of the family also takes .NET's standard <c>(message, innerException)</c>, which keeps the message
    ///     verbatim and leaves the attributes empty (null, no dtypes, index 0): for code that wraps a failure in NumPy's
    ///     class without a ufunc at hand. NumSharp's own raise sites always use NumPy's arguments.
    ///     </para>
    /// </remarks>
    public class UFuncTypeError : TypeError
    {
        /// <summary>
        ///     NumPy's <c>UFuncTypeError(ufunc)</c>: the message is the ufunc's <c>str()</c>, <c>&lt;ufunc 'add'&gt;</c>
        ///     (NumPy's base class prints its lone argument).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        /// <remarks>Footgun: the lone string is the ufunc's NAME, not a message (<c>new UFuncTypeError("add")</c> reads
        ///     <c>&lt;ufunc 'add'&gt;</c>), while the two-argument <see cref="UFuncTypeError(string, Exception)"/> takes a
        ///     message.</remarks>
        public UFuncTypeError(string ufunc) : this(ufunc, $"<ufunc '{ufunc}'>") { }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim and <see cref="ufunc"/> is null.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>UFuncTypeError</c> takes the ufunc alone). Footgun: here the
        ///     string is the MESSAGE, while the one-argument <see cref="UFuncTypeError(string)"/> takes the ufunc's name;
        ///     <c>new UFuncTypeError("add", null)</c> reads <c>add</c>.</remarks>
        public UFuncTypeError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>
        ///     The subclasses' form: the ufunc and the subclass's own text (NumPy's subclasses override <c>__str__</c>).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="message">The text, already in NumPy's form.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        protected UFuncTypeError(string ufunc, string message)
            : base(message)
        {
            this.ufunc = ufunc ?? throw new ArgumentNullException(nameof(ufunc));
        }

        /// <summary>
        ///     The subclasses' form with a cause: Python's <c>raise ... from cause</c> (NumPy chains a promotion error into the
        ///     no-loop refusal that replaces it), the cause kept in <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="message">The text, already in NumPy's form.</param>
        /// <param name="innerException">The exception the refusal replaces, or null for none.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        private protected UFuncTypeError(string ufunc, string message, Exception innerException)
            : base(message, innerException)
        {
            this.ufunc = ufunc ?? throw new ArgumentNullException(nameof(ufunc));
        }

        /// <summary>
        ///     The ufunc's name (NumPy's <c>ufunc</c> attribute holds the ufunc object; NumSharp, which has none, keeps its
        ///     <c>__name__</c>: <c>add</c>, <c>str_len</c>, <c>modf</c>). Null when the error was built from a message alone
        ///     (<see cref="UFuncTypeError(string, Exception)"/> and its subclasses' twins).
        /// </summary>
        public string ufunc { get; }

        /// <summary>
        ///     Python's <c>repr()</c> of a dtype-like value as NumPy's texts print it: <c>dtype('float64')</c> for a
        ///     <see cref="DType"/>, <c>&lt;class '…'&gt;</c> for a <see cref="DTypeMeta"/>, <c>None</c> for null, a
        ///     placeholder's own text, any other value's Python <c>repr()</c>.
        /// </summary>
        /// <param name="descr">The value.</param>
        /// <returns>The text.</returns>
        internal static string DescrRepr(object descr) => descr switch
        {
            null => "None",
            DType d => d.ToString(true),
            DTypeMeta m => m.ToString(),
            PyDescrPlaceholder p => p.ToString(),
            _ => NDPyObject.Repr(descr),
        };

        /// <summary>Python's <c>repr()</c> of a ufunc's name: <c>'add'</c>, CPython's quoting and escapes.</summary>
        /// <param name="ufunc">The name.</param>
        /// <returns>The quoted name.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        internal static string NameRepr(string ufunc)
            => PyRepr.Str(ufunc ?? throw new ArgumentNullException(nameof(ufunc)));
    }

    /// <summary>
    ///     NumPy's <c>_UFuncNoLoopError</c>: a ufunc has no loop for its operands' dtypes. Its text is NumPy's
    ///     <c>ufunc 'add' did not contain a loop with signature matching types (dtype('&lt;U1'), dtype('int64')) -&gt;
    ///     None</c>: the input dtypes, then the output ones, each a lone value or a tuple.
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="UFuncTypeError"/> (and so of <see cref="TypeError"/>); its own subclass is
    ///     <see cref="UFuncBinaryResolutionError"/>. NumSharp raises it wherever NumPy does (a numeric ufunc over text, a
    ///     text ufunc over numbers, a <c>dtype=</c> the ufunc has no loop for), with NumPy's dtype classes or instances as
    ///     its resolver reported them.
    /// </remarks>
    public class UFuncNoLoopError : UFuncTypeError
    {
        /// <summary>The operand dtypes, inputs then outputs, as given.</summary>
        private readonly object[] _dtypes;

        /// <summary>
        ///     NumPy's <c>_UFuncNoLoopError(ufunc, dtypes)</c>: the message is NumPy's <c>__str__</c>, splitting
        ///     <paramref name="dtypes"/> after the ufunc's <paramref name="nin"/> inputs.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="nin">The ufunc's input count (NumPy reads <c>ufunc.nin</c>): the first <paramref name="nin"/>
        ///     dtypes are the inputs, the rest the outputs.</param>
        /// <param name="dtypes">Every operand's dtype-like value, inputs then outputs (null where NumPy has <c>None</c>,
        ///     an output nobody requested).</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="nin"/> is negative.</exception>
        /// <exception cref="TypeError"><paramref name="dtypes"/> is null (Python's <c>tuple(None)</c>: <c>'NoneType' object
        ///     is not iterable</c>).</exception>
        /// <remarks>
        ///     One side prints as its lone value when it has one (<c>dtype('&lt;U1') -&gt; None</c>) and as a tuple otherwise
        ///     (<c>(dtype('&lt;U1'), dtype('int64')) -&gt; (None, None)</c>, <c>()</c> when empty), as NumPy's
        ///     <c>_unpack_tuple</c> does. Footgun: a single literal <c>null</c> argument binds as the whole
        ///     <paramref name="dtypes"/> array (C#'s <c>params</c> rule), not as one None; write
        ///     <c>new object[] { null }</c>.
        /// </remarks>
        public UFuncNoLoopError(string ufunc, int nin, params object[] dtypes)
            : this(ufunc, Copy(dtypes), FormatNoLoop(ufunc, nin, dtypes)) { }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim, <see cref="UFuncTypeError.ufunc"/> is null and
        ///     <see cref="dtypes"/> is empty.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>_UFuncNoLoopError</c> takes the ufunc and its dtypes); for code
        ///     that holds a text only.</remarks>
        public UFuncNoLoopError(string message, Exception innerException) : base(message, innerException)
        {
            // No operands to report: an empty tuple, so dtypes reads as NumPy's () rather than failing on a null array.
            _dtypes = Array.Empty<object>();
        }

        /// <summary>
        ///     NumSharp's raise sites' form with a cause (Python's <c>raise _UFuncNoLoopError(...) from cause</c>): the
        ///     promotion error the no-loop refusal replaces stays as <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="innerException">The exception the refusal replaces (first, so it never binds as a dtype).</param>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="nin">The ufunc's input count.</param>
        /// <param name="dtypes">Every operand's dtype-like value, inputs then outputs.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="nin"/> is negative.</exception>
        /// <exception cref="TypeError"><paramref name="dtypes"/> is null.</exception>
        internal UFuncNoLoopError(Exception innerException, string ufunc, int nin, object[] dtypes)
            : base(ufunc, FormatNoLoop(ufunc, nin, dtypes), innerException)
        {
            _dtypes = Copy(dtypes);
        }

        /// <summary>
        ///     The subclass form: the ufunc, its dtypes and the subclass's own text (NumPy's
        ///     <c>_UFuncBinaryResolutionError</c> overrides <c>__str__</c>).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="dtypes">The dtypes, already copied.</param>
        /// <param name="message">The text, already in NumPy's form.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        protected UFuncNoLoopError(string ufunc, object[] dtypes, string message) : base(ufunc, message)
        {
            _dtypes = dtypes;
        }

        /// <summary>
        ///     Every operand's dtype-like value, inputs then outputs (NumPy's <c>dtypes</c> tuple): a <see cref="DType"/>, a
        ///     <see cref="DTypeMeta"/>, or null for <c>None</c>. Read-only, as NumPy's tuple is.
        /// </summary>
        public ReadOnlyCollection<object> dtypes => Array.AsReadOnly(_dtypes);

        /// <summary>A private copy of the caller's dtypes (Python's <c>tuple(dtypes)</c>).</summary>
        /// <param name="dtypes">The caller's array.</param>
        /// <returns>The copy.</returns>
        /// <exception cref="TypeError">The array is null.</exception>
        private protected static object[] Copy(object[] dtypes)
            => (object[])(dtypes ?? throw new TypeError("'NoneType' object is not iterable")).Clone();

        /// <summary>NumPy's <c>_UFuncNoLoopError.__str__</c>.</summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="nin">The input count.</param>
        /// <param name="dtypes">The dtypes.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">A null name.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A negative input count.</exception>
        /// <exception cref="TypeError">A null dtype array.</exception>
        private static string FormatNoLoop(string ufunc, int nin, object[] dtypes)
        {
            string name = NameRepr(ufunc);
            if (dtypes is null)
                throw new TypeError("'NoneType' object is not iterable");
            if (nin < 0)
                throw new ArgumentOutOfRangeException(nameof(nin), nin.ToString(CultureInfo.InvariantCulture),
                    "A ufunc's input count is not negative.");
            // Python's slices clamp: dtypes[:nin] is everything when nin is past the end, dtypes[nin:] then empty.
            int split = Math.Min(nin, dtypes.Length);
            return $"ufunc {name} did not contain a loop with signature matching types {Unpack(dtypes, 0, split)} -> " +
                   Unpack(dtypes, split, dtypes.Length - split);
        }

        /// <summary>
        ///     NumPy's <c>repr(_unpack_tuple(t))</c>: the lone value's repr for a one-element slice, the tuple's repr
        ///     otherwise (<c>()</c> when empty, no trailing comma since a one-element tuple never reaches here).
        /// </summary>
        /// <param name="dtypes">The dtypes.</param>
        /// <param name="start">The slice's start.</param>
        /// <param name="count">Its length.</param>
        /// <returns>The text.</returns>
        private static string Unpack(object[] dtypes, int start, int count)
        {
            if (count == 1)
                return DescrRepr(dtypes[start]);
            var sb = new StringBuilder("(");
            for (int i = 0; i < count; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(DescrRepr(dtypes[start + i]));
            }
            return sb.Append(')').ToString();
        }
    }

    /// <summary>
    ///     NumPy's <c>_UFuncBinaryResolutionError</c>: a binary ufunc whose two operand dtypes cannot be combined (<c>ufunc
    ///     'add' cannot use operands with types dtype('S1') and dtype('&lt;U1')</c>).
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="UFuncNoLoopError"/>, as NumPy's is, holding exactly two <see cref="UFuncNoLoopError.dtypes"/>.
    /// </remarks>
    public class UFuncBinaryResolutionError : UFuncNoLoopError
    {
        /// <summary>
        ///     NumPy's <c>_UFuncBinaryResolutionError(ufunc, dtypes)</c>: the message names both operand dtypes.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="dtypes">The two operands' dtype-like values.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtypes"/> is null.</exception>
        /// <exception cref="AssertionError"><paramref name="dtypes"/> does not hold exactly two values (NumPy's <c>assert
        ///     len(self.dtypes) == 2</c>, an empty text).</exception>
        public UFuncBinaryResolutionError(string ufunc, params object[] dtypes)
            : base(ufunc, CheckTwo(dtypes), FormatBinary(ufunc, dtypes)) { }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim, <see cref="UFuncTypeError.ufunc"/> is null and
        ///     <see cref="UFuncNoLoopError.dtypes"/> is empty (NumPy's assertion of two dtypes does not apply).
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>_UFuncBinaryResolutionError</c> takes the ufunc and the two
        ///     dtypes); for code that holds a text only. An <see cref="Exception"/>-typed second argument binds here, not as
        ///     a lone dtype of the <c>params</c> form.</remarks>
        public UFuncBinaryResolutionError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>NumPy's assertion: exactly two dtypes.</summary>
        /// <param name="dtypes">The caller's array.</param>
        /// <returns>A private copy.</returns>
        /// <exception cref="TypeError">The array is null.</exception>
        /// <exception cref="AssertionError">It does not hold two values.</exception>
        private static object[] CheckTwo(object[] dtypes)
        {
            object[] copy = Copy(dtypes);
            if (copy.Length != 2)
                throw new AssertionError();
            return copy;
        }

        /// <summary>NumPy's <c>_UFuncBinaryResolutionError.__str__</c>.</summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="dtypes">The two dtypes (checked by <see cref="CheckTwo"/> first).</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">A null name.</exception>
        private static string FormatBinary(string ufunc, object[] dtypes)
            => $"ufunc {NameRepr(ufunc)} cannot use operands with types {DescrRepr(dtypes[0])} and {DescrRepr(dtypes[1])}";
    }

    /// <summary>
    ///     NumPy's <c>_UFuncCastingError</c>: the base of the casting refusals, a cast between an operand and the loop's
    ///     dtype that the <c>casting=</c> rule forbids. NumPy raises only its two subclasses,
    ///     <see cref="UFuncInputCastingError"/> and <see cref="UFuncOutputCastingError"/>.
    /// </summary>
    /// <remarks>
    ///     It carries <see cref="casting"/>, <see cref="from_"/> and <see cref="to"/>. Created directly, its text is Python's
    ///     <c>str()</c> of its arguments, <c>(&lt;ufunc 'add'&gt;, 'safe', dtype('float64'), dtype('int64'))</c>, as NumPy
    ///     gives the class no <c>__str__</c>.
    /// </remarks>
    public class UFuncCastingError : UFuncTypeError
    {
        /// <summary>
        ///     NumPy's <c>_UFuncCastingError(ufunc, casting, from_, to)</c>.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="casting">The rule (<c>no</c>, <c>equiv</c>, <c>safe</c>, <c>same_kind</c>, <c>unsafe</c>).</param>
        /// <param name="from_">The dtype the cast starts from.</param>
        /// <param name="to">The dtype it would reach.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        public UFuncCastingError(string ufunc, string casting, object from_, object to)
            : this(ufunc, casting, from_, to,
                   $"(<ufunc '{ufunc}'>, {(casting is null ? "None" : PyRepr.Str(casting))}, {DescrRepr(from_)}, {DescrRepr(to)})") { }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim and <see cref="UFuncTypeError.ufunc"/>, <see cref="casting"/>,
        ///     <see cref="from_"/> and <see cref="to"/> are null.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>_UFuncCastingError</c> takes the ufunc, the rule and the two
        ///     dtypes); for code that holds a text only.</remarks>
        public UFuncCastingError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>The subclasses' form: the four values and the subclass's own text.</summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="casting">The rule.</param>
        /// <param name="from_">The dtype the cast starts from.</param>
        /// <param name="to">The dtype it would reach.</param>
        /// <param name="message">The text, already in NumPy's form.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        protected UFuncCastingError(string ufunc, string casting, object from_, object to, string message)
            : base(ufunc, message)
        {
            this.casting = casting;
            this.from_ = from_;
            this.to = to;
        }

        /// <summary>The casting rule the cast broke (NumPy's <c>casting</c>: <c>same_kind</c>, <c>safe</c>, …).</summary>
        public string casting { get; }

        /// <summary>The dtype the cast starts from (NumPy's <c>from_</c>): a <see cref="DType"/>, or the placeholder
        ///     for NumPy's object dtype (see <see cref="UFuncTypeError"/>).</summary>
        public object from_ { get; }

        /// <summary>The dtype the cast would reach (NumPy's <c>to</c>): a <see cref="DType"/>, or the placeholder for
        ///     NumPy's object dtype.</summary>
        public object to { get; }

        /// <summary>
        ///     NumPy's casting texts: <c>Cannot cast ufunc 'add' {role} {i }from dtype('float64') to dtype('int64') with
        ///     casting rule 'same_kind'</c>, the index printed only when the ufunc has more than one operand on that side.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="role"><c>input</c> or <c>output</c>.</param>
        /// <param name="count">The ufunc's input (or output) count.</param>
        /// <param name="i">The operand index NumPy reports.</param>
        /// <param name="casting">The rule.</param>
        /// <param name="from_">The source dtype.</param>
        /// <param name="to">The target dtype.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">A null name.</exception>
        private protected static string FormatCast(string ufunc, string role, int count, int i, string casting, object from_, object to)
        {
            string index = count != 1 ? i.ToString(CultureInfo.InvariantCulture) + " " : string.Empty;
            return $"Cannot cast ufunc {NameRepr(ufunc)} {role} {index}from {DescrRepr(from_)} to {DescrRepr(to)} " +
                   $"with casting rule {(casting is null ? "None" : PyRepr.Str(casting))}";
        }
    }

    /// <summary>
    ///     NumPy's <c>_UFuncInputCastingError</c>: an input cannot be cast to the loop's dtype under the <c>casting=</c>
    ///     rule (<c>Cannot cast ufunc 'add' input 0 from dtype('float64') to dtype('int64') with casting rule 'safe'</c>).
    /// </summary>
    /// <remarks>A subclass of <see cref="UFuncCastingError"/>; it adds <see cref="in_i"/>, the input's index.</remarks>
    public class UFuncInputCastingError : UFuncCastingError
    {
        /// <summary>
        ///     NumPy's <c>_UFuncInputCastingError(ufunc, casting, from_, to, i)</c>, with the ufunc's input count NumPy
        ///     reads off the ufunc object.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="nin">The ufunc's input count: the text names the input's index only when it is not 1.</param>
        /// <param name="casting">The rule (<c>safe</c>, <c>same_kind</c>, …).</param>
        /// <param name="from_">The input's dtype.</param>
        /// <param name="to">The loop's dtype for it.</param>
        /// <param name="i">The input's index.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        public UFuncInputCastingError(string ufunc, int nin, string casting, object from_, object to, int i)
            : base(ufunc, casting, from_, to, FormatCast(ufunc, "input", nin, i, casting, from_, to))
        {
            in_i = i;
        }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim, <see cref="UFuncTypeError.ufunc"/>,
        ///     <see cref="UFuncCastingError.casting"/>, <see cref="UFuncCastingError.from_"/> and
        ///     <see cref="UFuncCastingError.to"/> are null and <see cref="in_i"/> is 0.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>_UFuncInputCastingError</c> takes the ufunc, the rule, the two
        ///     dtypes and the index); for code that holds a text only. Footgun: <see cref="in_i"/> reads 0, a valid index, as
        ///     the Unicode errors' <c>start</c> does when built from a message.</remarks>
        public UFuncInputCastingError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>The index of the input that cannot be cast (NumPy's <c>in_i</c>); 0 when the error was built from a
        ///     message alone.</summary>
        public int in_i { get; }
    }

    /// <summary>
    ///     NumPy's <c>_UFuncOutputCastingError</c>: the loop's result cannot be cast to an output (<c>out=</c>) under the
    ///     <c>casting=</c> rule (<c>Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule
    ///     'same_kind'</c>).
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="UFuncCastingError"/>; it adds <see cref="out_i"/>. Footgun, NumPy's own: <see cref="out_i"/>
    ///     is the operand's index among ALL operands (inputs first: <c>np.add</c>'s output is 2), and that is the number
    ///     the text prints for a ufunc with several outputs (<c>np.modf</c>'s first output reads <c>output 1</c>).
    /// </remarks>
    public class UFuncOutputCastingError : UFuncCastingError
    {
        /// <summary>
        ///     NumPy's <c>_UFuncOutputCastingError(ufunc, casting, from_, to, i)</c>, with the ufunc's output count NumPy
        ///     reads off the ufunc object.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="nout">The ufunc's output count: the text names the operand's index only when it is not 1.</param>
        /// <param name="casting">The rule (<c>same_kind</c>, …).</param>
        /// <param name="from_">The loop's result dtype.</param>
        /// <param name="to">The output array's dtype.</param>
        /// <param name="i">The output's index among all operands (inputs first).</param>
        /// <exception cref="ArgumentNullException"><paramref name="ufunc"/> is null.</exception>
        public UFuncOutputCastingError(string ufunc, int nout, string casting, object from_, object to, int i)
            : base(ufunc, casting, from_, to, FormatCast(ufunc, "output", nout, i, casting, from_, to))
        {
            out_i = i;
        }

        /// <summary>
        ///     Creates the error from a message and the exception that caused it, .NET's standard constructor every NumSharp
        ///     exception has: the message is kept verbatim, <see cref="UFuncTypeError.ufunc"/>,
        ///     <see cref="UFuncCastingError.casting"/>, <see cref="UFuncCastingError.from_"/> and
        ///     <see cref="UFuncCastingError.to"/> are null and <see cref="out_i"/> is 0.
        /// </summary>
        /// <param name="message">The verbatim message.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        /// <remarks>Python has no such constructor (NumPy's <c>_UFuncOutputCastingError</c> takes the ufunc, the rule, the two
        ///     dtypes and the index); for code that holds a text only. Footgun: <see cref="out_i"/> reads 0, a valid operand
        ///     index, as the Unicode errors' <c>start</c> does when built from a message.</remarks>
        public UFuncOutputCastingError(string message, Exception innerException) : base(message, innerException) { }

        /// <summary>The index of the output that cannot be cast, among all operands (NumPy's <c>out_i</c>); 0 when the error
        ///     was built from a message alone.</summary>
        public int out_i { get; }
    }

    /// <summary>
    ///     Stands in, inside a ufunc error, for a NumPy dtype NumSharp has no descriptor for (the object dtype and its
    ///     class): its <see cref="ToString"/> is NumPy's <c>repr()</c> of the missing value, so the error's text and its
    ///     dtype attributes still read as NumPy's.
    /// </summary>
    internal sealed class PyDescrPlaceholder
    {
        /// <summary>NumPy's object dtype instance, <c>dtype('O')</c>.</summary>
        internal static readonly PyDescrPlaceholder ObjectDescr = new("dtype('O')");

        /// <summary>NumPy's object dtype class, <c>&lt;class 'numpy.dtypes.ObjectDType'&gt;</c>.</summary>
        internal static readonly PyDescrPlaceholder ObjectDTypeClass = new("<class 'numpy.dtypes.ObjectDType'>");

        /// <summary>NumPy's complex64 dtype, <c>dtype('complex64')</c>: the loop NEP 50 picks for a Python complex beside
        ///     a float32 array, which NumSharp computes in complex128 but names as NumPy does.</summary>
        internal static readonly PyDescrPlaceholder Complex64Descr = new("dtype('complex64')");

        /// <summary>NumPy's <c>repr()</c> of the stood-in value.</summary>
        private readonly string _repr;

        /// <summary>Creates a placeholder printing <paramref name="repr"/>.</summary>
        /// <param name="repr">NumPy's <c>repr()</c> of the value.</param>
        private PyDescrPlaceholder(string repr) => _repr = repr;

        /// <summary>NumPy's <c>repr()</c> of the stood-in value.</summary>
        /// <returns>The text (<c>dtype('O')</c>).</returns>
        public override string ToString() => _repr;
    }
}
