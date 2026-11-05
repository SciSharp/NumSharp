using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

// =============================================================================
// NDExpr.Strings.cs — numpy.strings' ufuncs and searches as np.evaluate nodes (docs/plans/evaluate-text.md, phase E2)
// =============================================================================
//
// NumPy's np.strings.str_len, the nine character predicates and the searches (find / rfind / count / index / rindex,
// startswith / endswith) read text and answer numbers, so their results feed the rest of a fused expression:
//
//     np.evaluate(NDExpr.StartsWith(names, "pa") & ((NDExpr)age > 30))
//     np.evaluate(NDExpr.Sum(NDExpr.StrLen(names)))
//
// A StringsNode never enters a fused kernel. When its program is built, the text lowering (NDExpr.Text.cs) turns it into
// a STAGE: per call, the engine's own string function runs over the node's operands (an input array as it is, a literal
// as NumPy's Python scalar, a computed numeric sub-tree into the temporary NumPy's unfused sequence would hold), with
// NumPy's checks, NA rules and errors, and its int64 / bool result is one more input of the fused remainder. A node whose
// text operand is not text at all (StrLen over an int array) reaches the engine's own no-loop refusal the same way.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        // ===================================================================
        // np.strings: the length and the nine character predicates (x -> int64 / bool)
        // ===================================================================

        /// <summary>
        ///     NumPy's <c>np.strings.str_len(x)</c> as an expression node: the length of every element in characters (code
        ///     points for <c>U</c> and <c>StringDType</c>, bytes for <c>S</c>), trailing NULs of a fixed-width element not
        ///     counted; an int64 result.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c>, <c>U</c> or <c>StringDType</c> array leaf (or a C# <c>string</c>, which is
        ///     NumPy's <c>str</c>).</param>
        /// <returns>The node; evaluating it runs the engine's own <c>str_len</c> over <paramref name="x"/> first (a stage),
        ///     so the rest of the expression reads its int64 result.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>
        ///     Raised at evaluation, as NumPy raises them: <see cref="TypeError"/> when <paramref name="x"/> is not text
        ///     (NumPy's no-loop text naming its class); <see cref="ValueError"/> for a <c>StringDType</c> NA whose NA
        ///     object is not a string (<c>The length of a null string is undefined</c>).
        /// </remarks>
        public static NDExpr StrLen(NDExpr x) => new StringsNode(NDExprStringsFunction.StrLen, x);

        /// <summary>NumPy's <c>np.strings.isalpha(x)</c> as an expression node: True where every character is alphabetic and
        ///     there is at least one (CPython 3.12's Unicode tables for <c>U</c> / <c>StringDType</c>, ASCII for <c>S</c>).</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/> (see <see cref="StrLen"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string (NumPy's texts).</remarks>
        public static NDExpr IsAlpha(NDExpr x) => new StringsNode(NDExprStringsFunction.IsAlpha, x);

        /// <summary>NumPy's <c>np.strings.isdigit(x)</c> as an expression node: True where every character is a digit and
        ///     there is at least one.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsDigit(NDExpr x) => new StringsNode(NDExprStringsFunction.IsDigit, x);

        /// <summary>NumPy's <c>np.strings.isspace(x)</c> as an expression node: True where every character is whitespace and
        ///     there is at least one.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsSpace(NDExpr x) => new StringsNode(NDExprStringsFunction.IsSpace, x);

        /// <summary>NumPy's <c>np.strings.isalnum(x)</c> as an expression node: True where every character is alphanumeric
        ///     and there is at least one.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsAlnum(NDExpr x) => new StringsNode(NDExprStringsFunction.IsAlnum, x);

        /// <summary>NumPy's <c>np.strings.islower(x)</c> as an expression node: True where every cased character is
        ///     lowercase and there is at least one cased character.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsLower(NDExpr x) => new StringsNode(NDExprStringsFunction.IsLower, x);

        /// <summary>NumPy's <c>np.strings.isupper(x)</c> as an expression node: True where every cased character is
        ///     uppercase and there is at least one cased character.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsUpper(NDExpr x) => new StringsNode(NDExprStringsFunction.IsUpper, x);

        /// <summary>NumPy's <c>np.strings.istitle(x)</c> as an expression node: True where the string is titlecased
        ///     (uppercase characters only after uncased ones, lowercase only after cased ones) and has a cased character.</summary>
        /// <param name="x">The strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for a non-text <paramref name="x"/>; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsTitle(NDExpr x) => new StringsNode(NDExprStringsFunction.IsTitle, x);

        /// <summary>NumPy's <c>np.strings.isdecimal(x)</c> as an expression node: True where every character is a decimal
        ///     character and there is at least one. <c>U</c> and <c>StringDType</c> only.</summary>
        /// <param name="x">The strings (a <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for an <c>S</c> or non-text <paramref name="x"/> (NumPy has no
        ///     bytes loop: <c>ufunc 'isdecimal' did not contain a loop ... BytesDType</c>); <see cref="ValueError"/> for a
        ///     <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsDecimal(NDExpr x) => new StringsNode(NDExprStringsFunction.IsDecimal, x);

        /// <summary>NumPy's <c>np.strings.isnumeric(x)</c> as an expression node: True where every character is numeric and
        ///     there is at least one. <c>U</c> and <c>StringDType</c> only.</summary>
        /// <param name="x">The strings (a <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <returns>The node, a bool stage over <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <remarks>At evaluation: <see cref="TypeError"/> for an <c>S</c> or non-text <paramref name="x"/> (no bytes loop);
        ///     <see cref="ValueError"/> for a <c>StringDType</c> NA that is not NaN-like or a string.</remarks>
        public static NDExpr IsNumeric(NDExpr x) => new StringsNode(NDExprStringsFunction.IsNumeric, x);

        // ===================================================================
        // np.strings: the searches (a, sub, start=0, end=None)
        // ===================================================================

        /// <summary>
        ///     NumPy's <c>np.strings.find(a, sub, start=0, end=None)</c> as an expression node: for each element, the lowest
        ///     index where <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1; an int64 result. Indices are
        ///     characters (code points for <c>U</c> / <c>StringDType</c>).
        /// </summary>
        /// <param name="a">The searched strings (an <c>S</c>, <c>U</c> or <c>StringDType</c> leaf, or a C# <c>string</c>).</param>
        /// <param name="sub">The substrings, of <paramref name="a"/>'s class (a C# <c>string</c> is <c>str</c>, so an <c>S</c>
        ///     <paramref name="a"/> needs a 0-d <c>S</c> array).</param>
        /// <param name="start">The range start in slice notation (negative counts from the end): an integer literal, array or
        ///     sub-expression; null is NumPy's default <c>0</c>.</param>
        /// <param name="end">The range end in slice notation; null is NumPy's <c>None</c> (the whole string).</param>
        /// <returns>The node; evaluating it runs the engine's own <c>find</c> first (a stage), after evaluating any computed
        ///     <paramref name="start"/> / <paramref name="end"/> as NumPy's unfused sequence would (into a temporary).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <remarks>
        ///     Raised at evaluation, as NumPy raises them: <see cref="TypeError"/> when no loop takes the operand classes
        ///     (an <c>S</c> array with a <c>str</c> substring, a number as <paramref name="a"/>) or when two
        ///     <c>StringDType</c> operands carry different NA objects; <see cref="ArgumentException"/> for a
        ///     <paramref name="start"/> / <paramref name="end"/> that cannot reach int64 by a same_kind cast (a float);
        ///     <see cref="IncorrectShapeException"/> when the four operands do not broadcast; <see cref="ValueError"/> for
        ///     a <c>StringDType</c> NA whose NA object is not a string.
        /// </remarks>
        public static NDExpr Find(NDExpr a, NDExpr sub, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.Find, a, sub, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.rfind(a, sub, start=0, end=None)</c> as an expression node: for each element, the
        ///     highest index where <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1 (an empty substring is
        ///     found at the end); an int64 result.
        /// </summary>
        /// <param name="a">The searched strings.</param>
        /// <param name="sub">The substrings (see <see cref="Find"/>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, an int64 stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/>.</remarks>
        public static NDExpr RFind(NDExpr a, NDExpr sub, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.RFind, a, sub, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.count(a, sub, start=0, end=None)</c> as an expression node: for each element, the number
        ///     of non-overlapping occurrences of <paramref name="sub"/> inside <c>a[start:end]</c>; an int64 result.
        /// </summary>
        /// <param name="a">The searched strings.</param>
        /// <param name="sub">The substrings (see <see cref="Find"/>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, an int64 stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/>.</remarks>
        public static NDExpr Count(NDExpr a, NDExpr sub, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.Count, a, sub, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.index(a, sub, start=0, end=None)</c> as an expression node: <see cref="Find"/>, but an
        ///     element without a match raises <see cref="ValueError"/> (<c>substring not found</c>) for the whole evaluation.
        /// </summary>
        /// <param name="a">The searched strings.</param>
        /// <param name="sub">The substrings (see <see cref="Find"/>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, an int64 stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/> and <see cref="ValueError"/> when any element has no
        ///     match.</remarks>
        public static NDExpr Index(NDExpr a, NDExpr sub, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.Index, a, sub, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.rindex(a, sub, start=0, end=None)</c> as an expression node: <see cref="RFind"/>, but an
        ///     element without a match raises <see cref="ValueError"/> (<c>substring not found</c>).
        /// </summary>
        /// <param name="a">The searched strings.</param>
        /// <param name="sub">The substrings (see <see cref="Find"/>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, an int64 stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/> and <see cref="ValueError"/> when any element has no
        ///     match.</remarks>
        public static NDExpr RIndex(NDExpr a, NDExpr sub, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.RIndex, a, sub, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.startswith(a, prefix, start=0, end=None)</c> as an expression node: True where
        ///     <c>a[start:end]</c> starts with <paramref name="prefix"/>; a bool result.
        /// </summary>
        /// <param name="a">The strings.</param>
        /// <param name="prefix">The prefixes, of <paramref name="a"/>'s class (see <see cref="Find"/>'s <c>sub</c>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, a bool stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="prefix"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/> (a <c>StringDType</c> NA that is neither NaN-like nor a
        ///     string raises <see cref="ValueError"/>).</remarks>
        public static NDExpr StartsWith(NDExpr a, NDExpr prefix, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.StartsWith, a, prefix, start, end);

        /// <summary>
        ///     NumPy's <c>np.strings.endswith(a, suffix, start=0, end=None)</c> as an expression node: True where
        ///     <c>a[start:end]</c> ends with <paramref name="suffix"/>; a bool result.
        /// </summary>
        /// <param name="a">The strings.</param>
        /// <param name="suffix">The suffixes, of <paramref name="a"/>'s class (see <see cref="Find"/>'s <c>sub</c>).</param>
        /// <param name="start">The range start; null is <c>0</c>.</param>
        /// <param name="end">The range end; null is <c>None</c>.</param>
        /// <returns>The node, a bool stage (see <see cref="Find"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="suffix"/> is null.</exception>
        /// <remarks>At evaluation, the errors of <see cref="Find"/> (a <c>StringDType</c> NA that is neither NaN-like nor a
        ///     string raises <see cref="ValueError"/>).</remarks>
        public static NDExpr EndsWith(NDExpr a, NDExpr suffix, NDExpr start = null, NDExpr end = null)
            => new StringsNode(NDExprStringsFunction.EndsWith, a, suffix, start, end);
    }

    /// <summary>
    ///     The numpy.strings function a <see cref="StringsNode"/> runs. The member names are the factories' (PascalCase),
    ///     NumPy's names are <see cref="StringsNode.NumPyName"/>.
    /// </summary>
    internal enum NDExprStringsFunction : byte
    {
        /// <summary><c>str_len(x)</c> → int64.</summary>
        StrLen,

        /// <summary><c>isalpha(x)</c> → bool.</summary>
        IsAlpha,

        /// <summary><c>isdigit(x)</c> → bool.</summary>
        IsDigit,

        /// <summary><c>isspace(x)</c> → bool.</summary>
        IsSpace,

        /// <summary><c>isalnum(x)</c> → bool.</summary>
        IsAlnum,

        /// <summary><c>islower(x)</c> → bool.</summary>
        IsLower,

        /// <summary><c>isupper(x)</c> → bool.</summary>
        IsUpper,

        /// <summary><c>istitle(x)</c> → bool.</summary>
        IsTitle,

        /// <summary><c>isdecimal(x)</c> → bool (no bytes loop).</summary>
        IsDecimal,

        /// <summary><c>isnumeric(x)</c> → bool (no bytes loop).</summary>
        IsNumeric,

        /// <summary><c>find(a, sub, start, end)</c> → int64.</summary>
        Find,

        /// <summary><c>rfind(a, sub, start, end)</c> → int64.</summary>
        RFind,

        /// <summary><c>count(a, sub, start, end)</c> → int64.</summary>
        Count,

        /// <summary><c>index(a, sub, start, end)</c> → int64, raising for a missing match.</summary>
        Index,

        /// <summary><c>rindex(a, sub, start, end)</c> → int64, raising for a missing match.</summary>
        RIndex,

        /// <summary><c>startswith(a, prefix, start, end)</c> → bool.</summary>
        StartsWith,

        /// <summary><c>endswith(a, suffix, start, end)</c> → bool.</summary>
        EndsWith,
    }

    /// <summary>
    ///     A numpy.strings function in an expression (see the file header). Never compiled into a kernel: the text lowering
    ///     always makes it a stage that runs the engine's own function per call, so its IL members refuse and only its
    ///     structure (binding, the structural cache's hash and equality, the lowering) is implemented.
    /// </summary>
    /// <remarks>
    ///     Operands: <c>A</c> (the strings), then for the searches <c>Sub</c> (the substring, prefix or suffix) and the
    ///     optional <c>Start</c> / <c>End</c> (null = NumPy's defaults <c>0</c> / <c>None</c>). Every operand may be any
    ///     expression: a text leaf, a literal, a numeric sub-tree; the engine's function decides what computes and raises
    ///     NumPy's error for what does not.
    /// </remarks>
    internal sealed partial class StringsNode : NDExpr
    {
        /// <summary>The function.</summary>
        private readonly NDExprStringsFunction _fn;

        /// <summary>The strings operand (NumPy's <c>x</c> / <c>a</c>).</summary>
        private readonly NDExpr _a;

        /// <summary>The substring / prefix / suffix operand of a search; null for the length and the predicates.</summary>
        private readonly NDExpr _sub;

        /// <summary>The range start of a search; null for NumPy's default <c>0</c>.</summary>
        private readonly NDExpr _start;

        /// <summary>The range end of a search; null for NumPy's default <c>None</c>.</summary>
        private readonly NDExpr _end;

        /// <summary>Creates the node of a one-operand function (the length or a predicate).</summary>
        /// <param name="fn">The function.</param>
        /// <param name="x">The strings operand.</param>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        internal StringsNode(NDExprStringsFunction fn, NDExpr x)
        {
            _fn = fn;
            _a = x ?? throw new ArgumentNullException(nameof(x));
        }

        /// <summary>Creates the node of a search.</summary>
        /// <param name="fn">The search.</param>
        /// <param name="a">The strings operand.</param>
        /// <param name="sub">The substring / prefix / suffix operand.</param>
        /// <param name="start">The range start, or null for <c>0</c>.</param>
        /// <param name="end">The range end, or null for <c>None</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null (named as NumPy
        ///     names the parameter: <c>sub</c>, <c>prefix</c> or <c>suffix</c>).</exception>
        internal StringsNode(NDExprStringsFunction fn, NDExpr a, NDExpr sub, NDExpr start, NDExpr end)
        {
            _fn = fn;
            _a = a ?? throw new ArgumentNullException(nameof(a));
            _sub = sub ?? throw new ArgumentNullException(fn switch
            {
                NDExprStringsFunction.StartsWith => "prefix",
                NDExprStringsFunction.EndsWith => "suffix",
                _ => nameof(sub),
            });
            _start = start;
            _end = end;
        }

        /// <summary>The function this node runs.</summary>
        internal NDExprStringsFunction Function => _fn;

        /// <summary>NumPy's name of the function (<c>str_len</c>, <c>isalpha</c>, ..., <c>find</c>, <c>startswith</c>).</summary>
        internal string NumPyName => NumPyNameOf(_fn);

        /// <summary>
        ///     The dtype the function always returns: int64 for the length and the index searches, bool for the
        ///     predicates and <c>startswith</c> / <c>endswith</c> (NumPy's loops have no other output).
        /// </summary>
        internal NPTypeCode ResultType => ResultTypeOf(_fn);

        /// <summary>NumPy's name of a function.</summary>
        /// <param name="fn">The function.</param>
        /// <returns>The name NumPy's ufunc / function carries (its errors name it).</returns>
        internal static string NumPyNameOf(NDExprStringsFunction fn) => fn switch
        {
            NDExprStringsFunction.StrLen => "str_len",
            NDExprStringsFunction.IsAlpha => "isalpha",
            NDExprStringsFunction.IsDigit => "isdigit",
            NDExprStringsFunction.IsSpace => "isspace",
            NDExprStringsFunction.IsAlnum => "isalnum",
            NDExprStringsFunction.IsLower => "islower",
            NDExprStringsFunction.IsUpper => "isupper",
            NDExprStringsFunction.IsTitle => "istitle",
            NDExprStringsFunction.IsDecimal => "isdecimal",
            NDExprStringsFunction.IsNumeric => "isnumeric",
            NDExprStringsFunction.Find => "find",
            NDExprStringsFunction.RFind => "rfind",
            NDExprStringsFunction.Count => "count",
            NDExprStringsFunction.Index => "index",
            NDExprStringsFunction.RIndex => "rindex",
            NDExprStringsFunction.StartsWith => "startswith",
            _ => "endswith",
        };

        /// <summary>The result dtype of a function (see <see cref="ResultType"/>).</summary>
        /// <param name="fn">The function.</param>
        /// <returns><see cref="NPTypeCode.Int64"/> or <see cref="NPTypeCode.Boolean"/>.</returns>
        internal static NPTypeCode ResultTypeOf(NDExprStringsFunction fn) => fn switch
        {
            NDExprStringsFunction.StrLen or NDExprStringsFunction.Find or NDExprStringsFunction.RFind
                or NDExprStringsFunction.Count or NDExprStringsFunction.Index or NDExprStringsFunction.RIndex => NPTypeCode.Int64,
            _ => NPTypeCode.Boolean,
        };

        /// <summary>The <c>NDTextElements</c> predicate kind of a predicate function, or -1 for any other function.</summary>
        /// <param name="fn">The function.</param>
        /// <returns>The kind <see cref="NDStringMethods.Predicate"/> takes, or -1.</returns>
        internal static int PredicateKindOf(NDExprStringsFunction fn) => fn switch
        {
            NDExprStringsFunction.IsAlpha => NDTextElements.IsAlphaKind,
            NDExprStringsFunction.IsDigit => NDTextElements.IsDigitKind,
            NDExprStringsFunction.IsSpace => NDTextElements.IsSpaceKind,
            NDExprStringsFunction.IsAlnum => NDTextElements.IsAlnumKind,
            NDExprStringsFunction.IsLower => NDTextElements.IsLowerKind,
            NDExprStringsFunction.IsUpper => NDTextElements.IsUpperKind,
            NDExprStringsFunction.IsTitle => NDTextElements.IsTitleKind,
            NDExprStringsFunction.IsDecimal => NDTextElements.IsDecimalKind,
            NDExprStringsFunction.IsNumeric => NDTextElements.IsNumericKind,
            _ => -1,
        };

        // ---------------------------------------------------------------------
        // The kernel members: a StringsNode is always a stage, never IL
        // ---------------------------------------------------------------------

        /// <inheritdoc/>
        public override bool SupportsSimd => false;

        /// <inheritdoc/>
        /// <exception cref="InvalidOperationException">Always: the text lowering makes every StringsNode a stage before any
        ///     kernel is compiled, so reaching IL emission is a program-build bug.</exception>
        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                $"NDExpr.{_fn} runs as a stage (np.strings.{NumPyName}) and has no kernel: a program-build bug.");

        /// <inheritdoc/>
        /// <exception cref="InvalidOperationException">Always (see <see cref="EmitScalar"/>).</exception>
        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                $"NDExpr.{_fn} runs as a stage (np.strings.{NumPyName}) and has no kernel: a program-build bug.");

        /// <inheritdoc/>
        /// <remarks>Distinct per function and per present optional operand, so no two StringsNode structures share a key
        ///     (no kernel is ever keyed by it; the text plan keys its own remainder).</remarks>
        public override void AppendSignature(StringBuilder sb)
        {
            sb.Append("strings.").Append(NumPyName).Append('(');
            _a.AppendSignature(sb);
            AppendOptional(sb, _sub);
            AppendOptional(sb, _start);
            AppendOptional(sb, _end);
            sb.Append(')');
        }

        /// <summary>Appends <c>,</c> and an optional operand's signature, or <c>,-</c> for an absent one.</summary>
        /// <param name="sb">The signature builder.</param>
        /// <param name="operand">The operand, or null.</param>
        private static void AppendOptional(StringBuilder sb, NDExpr operand)
        {
            sb.Append(',');
            if (operand is null)
                sb.Append('-');
            else
                operand.AppendSignature(sb);
        }

        /// <inheritdoc/>
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;

        /// <inheritdoc/>
        /// <remarks>
        ///     The function's fixed result dtype, without typing the operands: they may be text (no NumPy type a fused
        ///     kernel knows), and the node itself is never typed inside a kernel — the remainder sees only the stage's
        ///     result leaf. The node is recorded so a caller's map holds an entry for it like any other node.
        /// </remarks>
        internal override NDExprTypeInfo InferType(NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            nodeTypes[this] = ResultType;
            return NDExprTypeInfo.Strong(ResultType);
        }

        // ---------------------------------------------------------------------
        // Binding and structure
        // ---------------------------------------------------------------------

        /// <inheritdoc/>
        /// <remarks>Also marks the context (<see cref="NDExprBindContext.HoldsStageOnlyNode"/>): a program over such a
        ///     tree takes the text plan even without a text input, so <c>StrLen(int array)</c> reaches the engine's own
        ///     refusal per call instead of a kernel build.</remarks>
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            ctx.HoldsStageOnlyNode = true;
            var a = _a.BindArrays(ctx);
            var sub = _sub?.BindArrays(ctx);
            var start = _start?.BindArrays(ctx);
            var end = _end?.BindArrays(ctx);
            return ReferenceEquals(a, _a) && ReferenceEquals(sub, _sub) && ReferenceEquals(start, _start)
                   && ReferenceEquals(end, _end)
                ? this
                : _sub is null
                    ? new StringsNode(_fn, a)
                    : new StringsNode(_fn, a, sub, start, end);
        }

        /// <inheritdoc/>
        /// <remarks>True when an operand holds a reduction, which the program build then refuses (a reduction is
        ///     root-only, and a stage operand is never the root).</remarks>
        internal override bool ContainsReduce
            => _a.ContainsReduce || (_sub?.ContainsReduce ?? false) || (_start?.ContainsReduce ?? false)
               || (_end?.ContainsReduce ?? false);

        /// <inheritdoc/>
        internal override NDArray FirstArray()
            => _a.FirstArray() ?? _sub?.FirstArray() ?? _start?.FirstArray() ?? _end?.FirstArray();

        /// <inheritdoc/>
        /// <remarks>The function and the presence of each optional operand fold into the hash, then the present operands in
        ///     order: <c>Find(a, s)</c> and <c>Find(a, s, 0)</c> are different structures (NumPy reads a 0 start the same,
        ///     but the trees differ, and the structural cache must not match them).</remarks>
        internal override void HashStructure(ref NDExprStructureHasher h, NDExprOperandScratch operands)
        {
            long shape = (long)_fn | (_sub is null ? 0L : 1L << 8) | (_start is null ? 0L : 1L << 9) | (_end is null ? 0L : 1L << 10);
            h.Add(NDExprNodeTag.Strings, shape);
            _a.HashStructure(ref h, operands);
            _sub?.HashStructure(ref h, operands);
            _start?.HashStructure(ref h, operands);
            _end?.HashStructure(ref h, operands);
        }

        /// <inheritdoc/>
        /// <remarks>The operands are compared in binding order (a, sub, start, end): an array leaf's index is the order of
        ///     its first visit, so any other order would number the leaves differently from the hash and the binding.</remarks>
        internal override bool StructureEquals(NDExpr bound, NDExprOperandScratch operands)
            => bound is StringsNode o && o._fn == _fn
               && (_sub is null) == (o._sub is null) && (_start is null) == (o._start is null) && (_end is null) == (o._end is null)
               && _a.StructureEquals(o._a, operands) && OptionalEquals(_sub, o._sub, operands)
               && OptionalEquals(_start, o._start, operands) && OptionalEquals(_end, o._end, operands);

        /// <summary>Whether two optional operands are structurally equal: both absent, or both present and equal.</summary>
        /// <param name="mine">This node's operand, or null.</param>
        /// <param name="theirs">The bound node's operand, or null.</param>
        /// <param name="operands">The operand scratch of the walk.</param>
        /// <returns>True when they match.</returns>
        private static bool OptionalEquals(NDExpr mine, NDExpr theirs, NDExprOperandScratch operands)
            => mine is null ? theirs is null : theirs is not null && mine.StructureEquals(theirs, operands);
    }
}
