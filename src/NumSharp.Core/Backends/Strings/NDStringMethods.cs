using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDStringMethods.cs — NumPy's string ufuncs of numpy.strings, and the fused sizing of its computed-width wrappers
// =============================================================================
//
// WHAT IS HERE (strings phase 2, P2.5)
//   The ufunc layer of every numpy.strings function that computes with text: the loop resolution (which class of loop —
//   S, U or StringDType — a call reaches, and NumPy's UFuncTypeError when none), the promoters' casts of the integer
//   parameters to int64 (with NumPy's input-cast errors and the Python-int overflow), the broadcast with NumPy's ufunc
//   text, the output's descriptor and layout, and the run (NDTextLoop over the IL shell). StringDType calls go on to
//   NDVStringMethods with their operands already in StringDType.
//
// THE COMPUTED-WIDTH WRAPPERS (strings plan D15)
//   NumPy's center / ljust / rjust / zfill / replace / expandtabs / partition size their out= array with numeric ufuncs
//   (str_len, np.maximum, count, np.where, max) before the loop. Here the size is ONE fused pass (an NDTextLoop size kernel
//   folding each element's size into a maximum), and NumPy's statements are kept exactly where they decide an error or a
//   dtype: the broadcast checks in NumPy's order with each statement's own text, the zero-size reduction error, the
//   'U6.0' descriptor of a float size (a uint64 width or count), the final out= descriptor.
//
// CONVENTIONS
//   * A null parameter is the wrapper's default (start 0, end None → MAX, chars None → whitespace, count -1).
//   * A C# number literal (NDArray.WasPythonLiteral) is NumPy's Python int / float: it names _PyLongDType in a no-loop
//     error, converts to int64 with the LP64 OverflowError text past int64, and np.asanyarray makes it int64 (uint64 past
//     int64) where a wrapper calls np.asanyarray first.
//   * House exception types: TypeError for NumPy's UFuncTypeError / TypeError, ArgumentException for an input- or
//     output-cast refusal, ValueError, IncorrectShapeException for a broadcast failure, OverflowException.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The ufunc layer of NumPy's <c>numpy.strings</c> computations over <c>S</c> / <c>U</c> text, and the dispatch of
    ///     <c>StringDType</c> calls to <see cref="NDVStringMethods"/> (see the file header).
    /// </summary>
    internal static unsafe class NDStringMethods
    {
        /// <summary><c>np.iinfo(np.int64).max</c>: the wrappers' <c>end=None</c> and <c>replace</c>'s count search bound.</summary>
        internal const long Max = long.MaxValue;

        /// <summary>The int64 descriptor every integer parameter is cast to (NumPy's promoters' <c>Int64DType</c>).</summary>
        private static DType Int64Descr => DType.From(NPTypeCode.Int64);

        // =============================================================================================================
        // Operand classes and conversions
        // =============================================================================================================

        /// <summary>The class of a string loop: fixed-width bytes (<c>S</c>), fixed-width str (<c>U</c>), or <c>StringDType</c>.</summary>
        internal enum TextLoop : byte
        {
            /// <summary>No loop exists for the operands.</summary>
            None,

            /// <summary>NumPy's <c>ENCODING::ASCII</c> loop over <c>S</c> operands.</summary>
            Bytes,

            /// <summary>NumPy's <c>ENCODING::UTF32</c> loop over <c>U</c> operands.</summary>
            Str,

            /// <summary>NumPy's <c>StringDType</c> loop (a <c>U</c> operand is cast to <c>StringDType()</c> first).</summary>
            VString,
        }

        /// <summary>True for a fixed-width <c>S</c> operand.</summary>
        /// <param name="x">The operand, or null.</param>
        /// <returns>True for <c>S</c>.</returns>
        internal static bool IsBytes(NDArray x) => x is not null && x.typecode == NPTypeCode.String && !x.Storage.IsVString && x.dtype.kind == 'S';

        /// <summary>True for a fixed-width <c>U</c> operand.</summary>
        /// <param name="x">The operand, or null.</param>
        /// <returns>True for <c>U</c>.</returns>
        internal static bool IsStr(NDArray x) => x is not null && x.typecode == NPTypeCode.String && !x.Storage.IsVString && x.dtype.kind == 'U';

        /// <summary>
        ///     The loop of a two-text ufunc (find family, startswith / endswith, the chars strip loops): two <c>S</c>, two
        ///     <c>U</c>, or a <c>StringDType</c> operand with a <c>StringDType</c> or <c>U</c> one (NumPy's StringDType
        ///     promoters); anything else has no loop.
        /// </summary>
        /// <param name="a">The first text operand.</param>
        /// <param name="b">The second text operand.</param>
        /// <returns>The loop class.</returns>
        internal static TextLoop ResolvePair(NDArray a, NDArray b)
        {
            bool aT = NDStringUfuncs.IsVString(a), bT = NDStringUfuncs.IsVString(b);
            if (IsBytes(a) && IsBytes(b))
                return TextLoop.Bytes;
            if (IsStr(a) && IsStr(b))
                return TextLoop.Str;
            if ((aT || IsStr(a)) && (bT || IsStr(b)) && (aT || bT))
                return TextLoop.VString;
            return TextLoop.None;
        }

        /// <summary>The loop of a one-text ufunc (str_len, the whitespace strip loops, _slice, …).</summary>
        /// <param name="a">The text operand.</param>
        /// <returns>The loop class.</returns>
        internal static TextLoop ResolveOne(NDArray a)
            => IsBytes(a) ? TextLoop.Bytes : IsStr(a) ? TextLoop.Str : NDStringUfuncs.IsVString(a) ? TextLoop.VString : TextLoop.None;

        /// <summary>
        ///     NumPy's UFuncTypeError of a call no loop accepts: <c>ufunc 'find' did not contain a loop with signature
        ///     matching types (&lt;class …&gt;, …) -&gt; None</c>, naming each operand's DType class (a C# literal's abstract
        ///     <c>_PyLongDType</c> / <c>_PyFloatDType</c>; a null default parameter is the Python int NumPy's wrapper passes).
        ///     A one-input ufunc names a C# literal by its default DType instead (<c>Int64DType</c>), as NumPy marks no
        ///     operand of such a ufunc weak (<see cref="NDStringUfuncs.UnaryDispatchClass"/>).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="operands">The inputs, in order (null = a Python-int default).</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError NoLoop(string ufunc, params NDArray[] operands)
        {
            var classes = new string[operands.Length];
            // NumPy marks a Python scalar weak only for a ufunc of two or more inputs (convert_ufunc_arguments stops at the
            // first input when nin == 1): a one-input ufunc (the whitespace strips) names a C# literal by its default DType.
            for (int i = 0; i < classes.Length; i++)
                classes[i] = operands[i] is null ? DTypeRegistry.PyLong.ToString()
                    : classes.Length == 1 ? NDStringUfuncs.UnaryDispatchClass(operands[i])
                    : NDStringUfuncs.DispatchClass(operands[i]);
            string types = classes.Length == 1 ? classes[0] : "(" + string.Join(", ", classes) + ")";
            return new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {types} -> None");
        }

        /// <summary>
        ///     NumPy's <c>np.asanyarray</c> of a wrapper parameter: a C# literal becomes the strong array NumPy makes of the
        ///     Python scalar (a Python int is int64, or uint64 past int64; a float float64; a complex complex128); any other
        ///     operand is itself.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>The array (a new one only for a literal; the caller disposes it then).</returns>
        internal static NDArray AsAnyArray(NDArray x)
        {
            if (x is null || !x.WasPythonLiteral)
                return x;
            return x.Storage.GetAtIndex(0) switch
            {
                double or float => NDArray.Scalar(Convert.ToDouble(x.Storage.GetAtIndex(0))),
                System.Numerics.Complex c => NDArray.Scalar(c),
                ulong u when u > long.MaxValue => NDArray.Scalar(u),
                var v => NDArray.Scalar(Convert.ToInt64(v)),
            };
        }

        /// <summary>
        ///     The int64 operand a promoter casts an integer parameter to (NumPy's <c>Int64DType</c> slot): an int64 array as
        ///     it is, another dtype cast if the same_kind rule allows it, a C# literal (Python int) converted with NumPy's
        ///     range check, a null parameter the wrapper's default.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name (for the error).</param>
        /// <param name="index">The input position NumPy names.</param>
        /// <param name="x">The operand, or null.</param>
        /// <param name="fallback">The default value of a null operand.</param>
        /// <returns>An int64 array (new unless <paramref name="x"/> already is one; the caller disposes it when new).</returns>
        /// <exception cref="ArgumentException">The operand cannot reach int64 by a same_kind cast: <c>Cannot cast ufunc
        ///     'find' input 2 from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c> (a C# double literal
        ///     is NumPy's Python float, named float64).</exception>
        /// <exception cref="OverflowException">A C# literal past int64: <c>Python int too large to convert to C long</c>
        ///     (NumPy's LP64 text).</exception>
        internal static NDArray LoopInt64(string ufunc, int index, NDArray x, long fallback)
        {
            if (x is null)
                return NDArray.Scalar(fallback);
            if (x.WasPythonLiteral)
            {
                switch (x.Storage.GetAtIndex(0))
                {
                    case double or float:
                        throw InputCast(ufunc, index, "dtype('float64')");
                    case System.Numerics.Complex:
                        throw InputCast(ufunc, index, "dtype('complex128')");
                    case ulong u when u > long.MaxValue:
                        throw new OverflowException("Python int too large to convert to C long");
                    case var v:
                        return NDArray.Scalar(Convert.ToInt64(v));
                }
            }
            if (x.typecode == NPTypeCode.Int64)
                return x;
            if (!DTypeCasting.CanCastTypeTo(x.dtype, Int64Descr, NPY_CASTING.NPY_SAME_KIND_CASTING))
                throw InputCast(ufunc, index, x.dtype.ToString(true));
            return x.astype(np.int64);
        }

        /// <summary>NumPy's input-cast refusal text into an int64 slot.</summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="index">The input position.</param>
        /// <param name="from">The source descriptor's repr.</param>
        /// <returns>The exception (not thrown).</returns>
        private static ArgumentException InputCast(string ufunc, int index, string from)
            => new ArgumentException($"Cannot cast ufunc '{ufunc}' input {index} from {from} to dtype('int64') with casting rule 'same_kind'");

        /// <summary>Disposes a temporary the call made (never the caller's operand).</summary>
        /// <param name="temp">The possibly-temporary array.</param>
        /// <param name="original">The caller's operand it may be.</param>
        internal static void DisposeIfTemp(NDArray temp, NDArray original)
        {
            if (temp is not null && !ReferenceEquals(temp, original))
                temp.Dispose();
        }

        /// <summary>
        ///     The layout of a fresh ufunc output: NumPy's NpyIter allocation over the inputs (each votes with the axes it
        ///     really strides; C order wins ties), 0-d for 0-d dimensions.
        /// </summary>
        /// <param name="dims">The broadcast dimensions (adopted).</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns>The shape.</returns>
        internal static Shape UfuncLayout(long[] dims, params NDArray[] inputs)
        {
            if (dims.Length == 0)
                return Shape.NewScalar();
            var shapes = new Shape[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                shapes[i] = inputs[i].Shape;
            return Shape.NpyIterOutputShape(dims, shapes);
        }

        /// <summary>The kernel identity's code-unit size of a fixed-width loop.</summary>
        /// <param name="loop">The loop class (<see cref="TextLoop.Bytes"/> or <see cref="TextLoop.Str"/>).</param>
        /// <returns>1 or 4.</returns>
        private static int UnitOf(TextLoop loop) => loop == TextLoop.Str ? 4 : 1;

        // =============================================================================================================
        // str_len (a public ufunc: out= / where= / dtype=)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>str_len</c> ufunc: the length of every element (trailing NULs never count; code points for
        ///     <c>StringDType</c>) as int64, with the ufunc's <c>out=</c> / <c>where=</c> / <c>dtype=</c>.
        /// </summary>
        /// <param name="x">The text operand.</param>
        /// <param name="out">The output, or null (a fresh int64 array in NumPy's layout).</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null (only int64 has a loop).</param>
        /// <returns><paramref name="out"/>, or the fresh result.</returns>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a <c>StringDType</c> NA that is not a string
        ///     (<c>The length of a null string is undefined</c>).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>; an <paramref name="out"/> int64 cannot
        ///     reach by a same_kind cast; operands that do not broadcast with <paramref name="out"/> / <paramref name="where"/>.</exception>
        /// <exception cref="TypeError">No loop: a non-text operand, or a <paramref name="dtype"/> other than int64; a
        ///     <paramref name="dtype"/> with details beyond its class (<c>'U1'</c>, <c>'&gt;i8'</c>) or a <c>StringDType</c>
        ///     instance (NumPy's <c>_get_dtype</c>, checked first).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (no out= / where=).</exception>
        internal static NDArray StrLen(NDArray x, NDArray @out, NDArray where, DType dtype)
        {
            // _get_dtype runs before the arguments are converted: a dtype= with details beyond its class ('U1', '>i8') or a
            // StringDType instance is refused before a read-only out or a non-bool where.
            if (dtype is not null)
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
            var parse = NDStringUfuncs.OutWhereError(@out, where);
            if (parse is not null)
                throw parse;
            var loop = ResolveOne(x);
            // A one-input ufunc marks no Python scalar weak: a C# literal is named by its default DType (Int64DType).
            if (dtype is not null && dtype.Meta != Int64Descr.Meta)
                throw new TypeError($"ufunc 'str_len' did not contain a loop with signature matching types {NDStringUfuncs.UnaryDispatchClass(x)} -> {dtype.Meta}");
            if (loop == TextLoop.None)
                throw new TypeError($"ufunc 'str_len' did not contain a loop with signature matching types {NDStringUfuncs.UnaryDispatchClass(x)} -> " +
                                    (dtype is null ? "None" : dtype.Meta.ToString()));
            if (@out is not null && !DTypeCasting.CanCastTypeTo(Int64Descr, @out.dtype, NPY_CASTING.NPY_SAME_KIND_CASTING))
                throw new ArgumentException(
                    $"Cannot cast ufunc 'str_len' output from {Int64Descr.ToString(true)} to {@out.dtype.ToString(true)} with casting rule 'same_kind'");
            var shapeError = NDStringUfuncs.BroadcastError(x, null, @out, where);
            if (shapeError is not null)
                throw shapeError;

            if (loop == TextLoop.VString)
                return NDVStringMethods.StrLen(x, @out, where);

            // An out= of another dtype receives the int64 loop's result through NumPy's output cast (computed here into a
            // temporary, then copied under the mask).
            if (@out is not null && @out.typecode != NPTypeCode.Int64)
                return NDStringArithmetic.CastInto(@out, where, () => StrLen(x, null, null, null));

            var dims = NDTextLoop.BroadcastDims(x, @out, where);
            NDArray result = @out ?? (where is null
                ? NDArray.ForOverwrite(Int64Descr, UfuncLayout(dims, x))
                : new NDArray(Int64Descr, UfuncLayout(dims, x, where), true));
            try
            {
                var key = new TextMethodKey(TextMethodOp.Length, UnitOf(loop), x.dtype.itemsize);
                NDTextLoop.Execute(key, new[] { x, result }, 1, where, null, callerOut: @out is not null, maskedBlend: true);
                return result;
            }
            catch
            {
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        // =============================================================================================================
        // The predicates isalpha ... isnumeric (public ufuncs: out= / where= / dtype=)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's predicate ufuncs <c>isalpha</c>, <c>isdigit</c>, <c>isspace</c>, <c>isalnum</c>, <c>islower</c>,
        ///     <c>isupper</c>, <c>istitle</c>, <c>isdecimal</c>, <c>isnumeric</c>: a bool per element, with the ufunc's
        ///     <c>out=</c> / <c>where=</c> / <c>dtype=</c>. The loops: <c>S</c> (ASCII rules; none for <c>isdecimal</c> /
        ///     <c>isnumeric</c>), <c>U</c> and <c>StringDType</c> (CPython's Unicode rules, <see cref="PyUnicodeData"/>).
        /// </summary>
        /// <param name="kind">The predicate (<see cref="NDTextElements.IsAlphaKind"/> ...).</param>
        /// <param name="x">The operand.</param>
        /// <param name="out">The output, or null (a fresh bool array in NumPy's layout).</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null (only bool has a loop).</param>
        /// <returns><paramref name="out"/>, or the fresh result (0-d for a 0-d operand).</returns>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a <c>StringDType</c> NA its descriptor refuses
        ///     (<c>Cannot use the isalpha function with a null that is not a nan-like value</c>).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>; an <paramref name="out"/> bool cannot reach
        ///     by a same_kind cast; operands that do not broadcast with <paramref name="out"/> / <paramref name="where"/>.</exception>
        /// <exception cref="TypeError">No loop: a non-text operand, an <c>S</c> operand of <c>isdecimal</c> /
        ///     <c>isnumeric</c>, or a <paramref name="dtype"/> other than bool (NumPy's UFuncTypeError text); a
        ///     <paramref name="dtype"/> with details beyond its class (<c>'U1'</c>) or a <c>StringDType</c> instance (NumPy's
        ///     <c>_get_dtype</c>, checked first).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (no out= / where=).</exception>
        internal static NDArray Predicate(int kind, NDArray x, NDArray @out, NDArray where, DType dtype)
        {
            string ufunc = NDTextElements.PredicateName(kind);
            // _get_dtype runs before the arguments are converted: a dtype= with details beyond its class ('U1') or a
            // StringDType instance is refused before a read-only out or a non-bool where.
            if (dtype is not null)
                DefaultEngine.ThrowIfNotGeneralDType(dtype);
            var parse = NDStringUfuncs.OutWhereError(@out, where);
            if (parse is not null)
                throw parse;
            var loop = ResolveOne(x);
            // The S loop (ENCODING::ASCII) is registered for the first seven predicates only.
            if (loop == TextLoop.Bytes && kind is NDTextElements.IsDecimalKind or NDTextElements.IsNumericKind)
                loop = TextLoop.None;
            // A one-input ufunc marks no Python scalar weak: a C# literal is named by its default DType (Int64DType).
            if (dtype is not null && dtype.Meta != DType.Boolean.Meta)
                throw new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {NDStringUfuncs.UnaryDispatchClass(x)} -> {dtype.Meta}");
            if (loop == TextLoop.None)
                throw new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types {NDStringUfuncs.UnaryDispatchClass(x)} -> " +
                                    (dtype is null ? "None" : dtype.Meta.ToString()));
            if (@out is not null && !DTypeCasting.CanCastTypeTo(DType.Boolean, @out.dtype, NPY_CASTING.NPY_SAME_KIND_CASTING))
                throw new ArgumentException(
                    $"Cannot cast ufunc '{ufunc}' output from {DType.Boolean.ToString(true)} to {@out.dtype.ToString(true)} with casting rule 'same_kind'");
            var shapeError = NDStringUfuncs.BroadcastError(x, null, @out, where);
            if (shapeError is not null)
                throw shapeError;

            if (loop == TextLoop.VString)
                return NDVStringMethods.Predicate(kind, x, @out, where);

            // An out= of another dtype receives the bool loop's result through NumPy's output cast (computed here into a
            // temporary, then copied under the mask: the fixed-width loops cannot raise, so computing every element is
            // safe).
            if (@out is not null && @out.typecode != NPTypeCode.Boolean)
                return NDStringArithmetic.CastInto(@out, where, () => Predicate(kind, x, null, null, null));

            var dims = NDTextLoop.BroadcastDims(x, @out, where);
            NDArray result = @out ?? (where is null
                ? NDArray.ForOverwrite(DType.Boolean, UfuncLayout(dims, x))
                : new NDArray(DType.Boolean, UfuncLayout(dims, x, where), true));
            try
            {
                // The call's state: plan D14's escalation word (0 while every U element so far was ASCII, the kernel's
                // compare path; 1 once one was not, the gather path for the rest of the call) and the BMP flags table,
                // read here once. One block for the whole call, so a later NDIter chunk starts where the earlier one left
                // off.
                var state = NDTextElements.NewKindState();
                var key = new TextMethodKey(TextMethodOp.IsKind, UnitOf(loop), x.dtype.itemsize, Extra: kind);
                NDTextLoop.Execute(key, new[] { x, result }, 1, where, &state, callerOut: @out is not null, maskedBlend: true);
                return result;
            }
            catch
            {
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        // =============================================================================================================
        // upper / lower / swapcase / capitalize / title (NumPy's _vec_string over the bytes / str methods)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's case maps <c>upper</c>, <c>lower</c>, <c>swapcase</c>, <c>capitalize</c>, <c>title</c>:
        ///     <c>_vec_string(a, a.dtype, name)</c> — a fresh C-ordered array of the operand's own dtype, each element the
        ///     <c>bytes</c> (<c>S</c>) or <c>str</c> (<c>U</c>, <c>StringDType</c>) method's result, an <c>S</c> / <c>U</c>
        ///     result cut to the element width.
        /// </summary>
        /// <param name="map">The map (<see cref="NDTextElements.UpperMap"/> ...).</param>
        /// <param name="a">The operand (any layout; NumPy's <c>np.asarray</c> has been applied by the caller).</param>
        /// <returns>The fresh result (0-d for a 0-d operand).</returns>
        /// <exception cref="TypeError">A non-text operand (<c>string operation on non-string array</c>); a <c>StringDType</c>
        ///     NA whose object is not a str (<c>descriptor 'upper' for 'str' objects doesn't apply to a 'float' object</c>).</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a unit past U+10FFFF (CPython's <c>invalid maximum
        ///     character passed to PyUnicode_New</c>).</exception>
        /// <remarks>
        ///     The loop's order cannot show: every element's error is the same text. NumPy's loop visits C order and ends at
        ///     the first failing element; this one computes every element and raises after the loop, and the result is
        ///     discarded either way.
        /// </remarks>
        internal static NDArray CaseMap(int map, NDArray a)
        {
            var loop = ResolveOne(a);
            // _vec_string picks bytes.<name> for S, str.<name> for U and StringDType, and refuses anything else.
            if (loop == TextLoop.None)
                throw new TypeError(NDCharObjectPath.NonStringArrayText);
            if (loop == TextLoop.VString)
                return NDVStringMethods.CaseMap(map, a);
            // PyArray_SimpleNewFromDescr: C order whatever the operand's layout; the kernel writes every byte.
            var result = NDArray.ForOverwrite(a.dtype, new Shape(a.Shape.dimensions));
            try
            {
                // The call's state: D14's escalation word (U), the error status, the tables (read once here).
                var state = NDTextElements.NewCaseState();
                var key = new TextMethodKey(TextMethodOp.CaseMap, UnitOf(loop), a.dtype.itemsize, Extra: map);
                NDTextLoop.Execute(key, new[] { a, result }, 1, null, &state, callerOut: false);
                if (state.Status != 0)
                    throw new SystemError(NDTextElements.InvalidMaxCharText);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        // =============================================================================================================
        // find / rfind / index / rindex / count / startswith / endswith
        // =============================================================================================================

        /// <summary>
        ///     NumPy's find-like ufuncs (<c>find</c>, <c>rfind</c>, <c>index</c>, <c>rindex</c>, <c>count</c>,
        ///     <c>startswith</c>, <c>endswith</c>): the loop resolution, the int64 casts of <paramref name="start"/> /
        ///     <paramref name="end"/>, the broadcast, a fresh int64 (bool for the tail matches) result, the loop, and the
        ///     <c>ValueError: substring not found</c> of <c>index</c> / <c>rindex</c>.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="op">The kernel operation (<see cref="TextMethodOp.Find"/> serves <c>index</c>,
        ///     <see cref="TextMethodOp.RFind"/> <c>rindex</c>).</param>
        /// <param name="a">The searched text.</param>
        /// <param name="sub">The searched-for text (needle, prefix or suffix).</param>
        /// <param name="start">The start index (null = 0).</param>
        /// <param name="end">The end index (null = None, i.e. <see cref="Max"/>).</param>
        /// <returns>The fresh result (0-d for 0-d operands).</returns>
        /// <exception cref="TypeError">No loop for the operand classes; two <c>StringDType</c> operands with different NA
        ///     objects (raised before the int64 casts, NumPy's resolve_descriptors).</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The four operands do not broadcast.</exception>
        /// <exception cref="ValueError"><c>index</c> / <c>rindex</c> found no match; a <c>StringDType</c> NA its rule
        ///     refuses.</exception>
        internal static NDArray FindLike(string ufunc, TextMethodOp op, NDArray a, NDArray sub, NDArray start, NDArray end)
        {
            var loop = ResolvePair(a, sub);
            if (loop == TextLoop.None)
                throw NoLoop(ufunc, a, sub, start, end);
            // resolve_descriptors (the StringDType NA check) runs before the input casts and the broadcast.
            if (loop == TextLoop.VString)
                NDVStringMethods.CheckPairNA(a, sub);
            NDArray s64 = null, e64 = null;
            try
            {
                s64 = LoopInt64(ufunc, 2, start, 0);
                e64 = LoopInt64(ufunc, 3, end, Max);
                var dims = NDTextLoop.BroadcastDims(a, sub, s64, e64);
                if (loop == TextLoop.VString)
                    return NDVStringMethods.FindLike(ufunc, op, a, sub, s64, e64, dims);

                bool tail = op is TextMethodOp.StartsWith or TextMethodOp.EndsWith;
                var outDescr = tail ? DType.Boolean : Int64Descr;
                var result = NDArray.ForOverwrite(outDescr, UfuncLayout(dims, a, sub, s64, e64));
                try
                {
                    var key = new TextMethodKey(op, UnitOf(loop), a.dtype.itemsize, sub.dtype.itemsize);
                    NDTextLoop.Execute(key, new[] { a, sub, s64, e64, result }, 4, null, null, callerOut: false);
                    if (ufunc is "index" or "rindex" && AnyNotFound(result))
                        throw new ValueError("substring not found");
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(s64, start);
                DisposeIfTemp(e64, end);
            }
        }

        /// <summary>Whether an int64 result holds a -1 (an element <c>index</c> / <c>rindex</c> did not find).</summary>
        /// <param name="result">The fresh C/F-contiguous int64 result.</param>
        /// <returns>True when some element is -1.</returns>
        private static bool AnyNotFound(NDArray result)
        {
            if (result.size == 0)
                return false;
            var span = new ReadOnlySpan<long>((long*)result.Address + result.Shape.offset, checked((int)result.size));
            return span.Contains(-1L);
        }

        // =============================================================================================================
        // strip / lstrip / rstrip
        // =============================================================================================================

        /// <summary>
        ///     NumPy's whitespace strip ufuncs (<c>_strip_whitespace</c>, <c>_lstrip_whitespace</c>,
        ///     <c>_rstrip_whitespace</c>): a fresh array of the operand's descriptor in NumPy's layout.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="type">The strip type (<see cref="NDTextElements.LeftStrip"/> …).</param>
        /// <param name="a">The text operand.</param>
        /// <returns>The stripped strings (0-d for a 0-d operand).</returns>
        /// <exception cref="TypeError">A non-text operand (NumPy's no-loop text naming its class).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA that is neither NaN-like nor a string.</exception>
        internal static NDArray StripWhitespace(string ufunc, int type, NDArray a)
        {
            var loop = ResolveOne(a);
            if (loop == TextLoop.None)
                throw NoLoop(ufunc, a);
            if (loop == TextLoop.VString)
                return NDVStringMethods.StripWhitespace(ufunc, type, a);
            var dims = NDTextLoop.BroadcastDims(a);
            var result = NDArray.ForOverwrite(a.dtype, UfuncLayout(dims, a));
            try
            {
                var key = new TextMethodKey(TextMethodOp.StripWhitespace, UnitOf(loop), a.dtype.itemsize, Extra: type);
                NDTextLoop.Execute(key, new[] { a, result }, 1, null, null, callerOut: false);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>chars</c> strip ufuncs (<c>_strip_chars</c>, <c>_lstrip_chars</c>, <c>_rstrip_chars</c>): the loop
        ///     resolution of the pair, the broadcast, a fresh array of the text operand's descriptor.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="type">The strip type.</param>
        /// <param name="a">The text operand.</param>
        /// <param name="chars">The strip-set operand.</param>
        /// <returns>The stripped strings.</returns>
        /// <exception cref="TypeError">No loop for the pair (an <c>S</c> array with a <c>str</c> set, a number, …); two
        ///     <c>StringDType</c> operands with different NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA its rule refuses.</exception>
        internal static NDArray StripChars(string ufunc, int type, NDArray a, NDArray chars)
        {
            var loop = ResolvePair(a, chars);
            if (loop == TextLoop.None)
                throw NoLoop(ufunc, a, chars);
            if (loop == TextLoop.VString)
                NDVStringMethods.CheckPairNA(a, chars);
            var dims = NDTextLoop.BroadcastDims(a, chars);
            if (loop == TextLoop.VString)
                return NDVStringMethods.StripChars(ufunc, type, a, chars, dims);
            var result = NDArray.ForOverwrite(a.dtype, UfuncLayout(dims, a, chars));
            try
            {
                var key = new TextMethodKey(TextMethodOp.StripChars, UnitOf(loop), a.dtype.itemsize, chars.dtype.itemsize, Extra: type);
                NDTextLoop.Execute(key, new[] { a, chars, result }, 2, null, null, callerOut: false);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        // =============================================================================================================
        // _slice
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_slice</c> ufunc after the wrapper's defaults: the loop resolution (the class names of the four
        ///     operands), the intp casts of the indices, the broadcast, a fresh array of the text operand's descriptor.
        /// </summary>
        /// <param name="a">The text operand.</param>
        /// <param name="start">The start indices (the wrapper's <c>np.where</c> default already applied).</param>
        /// <param name="stop">The stop indices.</param>
        /// <param name="step">The steps (<c>np.asanyarray</c>'d; the wrapper rejected 0).</param>
        /// <returns>The sliced strings.</returns>
        /// <exception cref="TypeError">A non-text operand.</exception>
        /// <exception cref="ArgumentException">An index that cannot reach intp by a same_kind cast.</exception>
        /// <exception cref="OverflowException">An index literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> NA (<c>Cannot slice null string</c>).</exception>
        internal static NDArray Slice(NDArray a, NDArray start, NDArray stop, NDArray step)
        {
            var loop = ResolveOne(a);
            if (loop == TextLoop.None)
                throw NoLoop("_slice", a, start, stop, step);
            NDArray s0 = null, s1 = null, s2 = null;
            try
            {
                s0 = LoopInt64("_slice", 1, start, 0);
                s1 = LoopInt64("_slice", 2, stop, 0);
                s2 = LoopInt64("_slice", 3, step, 1);
                var dims = NDTextLoop.BroadcastDims(a, s0, s1, s2);
                if (loop == TextLoop.VString)
                    return NDVStringMethods.Slice(a, s0, s1, s2, dims);
                var result = NDArray.ForOverwrite(a.dtype, UfuncLayout(dims, a, s0, s1, s2));
                try
                {
                    var key = new TextMethodKey(TextMethodOp.Slice, UnitOf(loop), a.dtype.itemsize);
                    NDTextLoop.Execute(key, new[] { a, s0, s1, s2, result }, 4, null, null, callerOut: false);
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(s0, start);
                DisposeIfTemp(s1, stop);
                DisposeIfTemp(s2, step);
            }
        }

        // =============================================================================================================
        // center / ljust / rjust / zfill (S / U: the fused sizing, then the loop into empty_like's out=)
        // =============================================================================================================

        /// <summary>
        ///     The fixed-width part of NumPy's <c>center</c> / <c>ljust</c> / <c>rjust</c> after its T branch:
        ///     <c>fillchar.astype(a.dtype)</c>, the width <c>max(np.maximum(str_len(a), width))</c> with NumPy's checks, the
        ///     out= of <c>np.empty_like(a, shape=broadcast_shapes(a, width, fillchar), dtype=…)</c>, then the pad loop.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name (<c>_center</c>, <c>_ljust</c>, <c>_rjust</c>; not part of any message on
        ///     this path).</param>
        /// <param name="pos">The justification (<see cref="NDTextElements.Center"/> …).</param>
        /// <param name="a">The operand (<c>np.asanyarray</c>'d; not <c>StringDType</c>).</param>
        /// <param name="width">The widths (<c>np.asanyarray</c>'d, an integer array).</param>
        /// <param name="fillchar">The fill characters (checked to be one character long).</param>
        /// <returns>The padded strings, laid out like <paramref name="a"/>.</returns>
        /// <exception cref="UnicodeEncodeError">A non-ASCII fill character for an <c>S</c> operand (the cast).</exception>
        /// <exception cref="UnicodeDecodeError">A fill byte ≥ 0x80 for a <c>U</c> operand.</exception>
        /// <exception cref="ValueError">A numeric operand (the fill's cast: <c>invalid literal for int() …</c>); an empty
        ///     broadcast (<c>zero-size array to reduction operation maximum which has no identity</c>).</exception>
        /// <exception cref="TypeError">A non-text operand (<c>str_len</c>'s no-loop text); a float width descriptor
        ///     (<c>data type 'U6.0' not understood</c>, a uint64 width).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="a"/> and <paramref name="width"/> do not broadcast
        ///     (np.maximum's text), or the fill does not (np.broadcast_shapes' text).</exception>
        /// <remarks>NumPy's <c>center</c> builds the descriptor (<c>width.max()</c>) before <c>np.broadcast_shapes</c> and
        ///     <c>ljust</c> / <c>rjust</c> after it, so an empty broadcast with a fill that does not broadcast raises a
        ///     different error in each; both orders are kept.</remarks>
        internal static NDArray Pad(string ufunc, int pos, NDArray a, NDArray width, NDArray fillchar)
        {
            // fillchar = fillchar.astype(a.dtype, copy=False)
            var fill = fillchar.dtype == a.dtype ? fillchar : fillchar.astype(a.dtype);
            NDArray w64 = null;
            try
            {
                // width = np.maximum(str_len(a), width)
                var dims = PadDims(a, width);
                DType outDescr;
                Shape shape;
                if (pos == NDTextElements.Center)
                {
                    // out_dtype = f"{a.dtype.char}{width.max()}"; shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
                    outDescr = PadDescriptor(a, width, dims, ref w64);
                    shape = np.broadcast_shapes(a.Shape, Shape.Adopt(dims), fill.Shape);
                }
                else
                {
                    shape = np.broadcast_shapes(a.Shape, Shape.Adopt(dims), fill.Shape);
                    outDescr = PadDescriptor(a, width, dims, ref w64);
                }
                var @out = NDArray.ForOverwrite(outDescr, Shape.NewLikeLayout(a.Shape, 'K', (long[])shape));
                try
                {
                    var key = new TextMethodKey(TextMethodOp.Pad, a.dtype.kind == 'U' ? 4 : 1, a.dtype.itemsize, fill.dtype.itemsize,
                                                outDescr.itemsize, Extra: pos);
                    NDTextLoop.Execute(key, new[] { a, w64, fill, @out }, 3, null, null, callerOut: false);
                    return @out;
                }
                catch
                {
                    @out.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(fill, fillchar);
                DisposeIfTemp(w64, width);
            }
        }

        /// <summary>
        ///     The fixed-width part of NumPy's <c>zfill</c> after its T branch: the width
        ///     <c>max(np.maximum(str_len(a), width))</c>, the out= of <c>np.empty_like(a, shape=…, dtype=…)</c>, the loop.
        /// </summary>
        /// <param name="a">The operand (<c>np.asanyarray</c>'d; not <c>StringDType</c>).</param>
        /// <param name="width">The widths (an integer array).</param>
        /// <returns>The zero-padded strings, laid out like <paramref name="a"/>.</returns>
        /// <exception cref="TypeError">A non-text operand; a float width descriptor.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="a"/> and <paramref name="width"/> do not broadcast.</exception>
        /// <exception cref="ValueError">An empty broadcast.</exception>
        internal static NDArray ZFill(NDArray a, NDArray width)
        {
            NDArray w64 = null;
            try
            {
                // width = np.maximum(str_len(a), width); shape = np.broadcast_shapes(a.shape, width.shape) (never fails
                // after the maximum); out_dtype = f"{a.dtype.char}{width.max()}"
                var dims = PadDims(a, width);
                var outDescr = PadDescriptor(a, width, dims, ref w64);
                var @out = NDArray.ForOverwrite(outDescr, Shape.NewLikeLayout(a.Shape, 'K', dims));
                try
                {
                    var key = new TextMethodKey(TextMethodOp.ZFill, a.dtype.kind == 'U' ? 4 : 1, a.dtype.itemsize, outDescr.itemsize);
                    NDTextLoop.Execute(key, new[] { a, w64, @out }, 2, null, null, callerOut: false);
                    return @out;
                }
                catch
                {
                    @out.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(w64, width);
            }
        }

        /// <summary>
        ///     The first sizing statement the pad wrappers share, <c>width = np.maximum(str_len(a), width)</c>: str_len's
        ///     no-loop error, then np.maximum's broadcast error.
        /// </summary>
        /// <param name="a">The text operand.</param>
        /// <param name="width">The integer widths.</param>
        /// <returns>The broadcast dimensions of <paramref name="a"/> and <paramref name="width"/> (the maximum's shape).</returns>
        /// <exception cref="TypeError">A non-text operand (<c>str_len</c>'s no-loop text).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        private static long[] PadDims(NDArray a, NDArray width)
        {
            var loop = ResolveOne(a);
            if (loop is TextLoop.None or TextLoop.VString)
                throw new TypeError($"ufunc 'str_len' did not contain a loop with signature matching types {NDStringUfuncs.DispatchClass(a)} -> None");
            // np.maximum(str_len(a), width): the shapes of str_len(a) (= a's) and width.
            return NDTextLoop.BroadcastDims(a, width);
        }

        /// <summary>
        ///     The pad wrappers' descriptor statement, <c>f"{a.dtype.char}{width.max()}"</c> over the maximum
        ///     <see cref="PadDims"/> sized: the zero-size error, a float descriptor for a uint64 width (int64 against uint64
        ///     is float64), else the int64 maximum.
        /// </summary>
        /// <param name="a">The text operand (its loop already resolved by <see cref="PadDims"/>).</param>
        /// <param name="width">The integer widths.</param>
        /// <param name="dims">The maximum's dimensions.</param>
        /// <param name="w64">Receives the widths as int64 (a new array unless <paramref name="width"/> is one).</param>
        /// <returns>The output descriptor.</returns>
        /// <exception cref="TypeError">A float descriptor (<c>data type 'U6.0' not understood</c>).</exception>
        /// <exception cref="ValueError">An empty broadcast (the maximum's <c>max()</c>).</exception>
        /// <remarks>
        ///     <c>max(maximum(len(a), width))</c> over a non-empty broadcast is <c>max(max(len(a)), max(width))</c> — every
        ///     element of each operand meets some element of the other — so the size pass reads each operand once (no
        ///     broadcast iteration), the length scan through the size kernel over <paramref name="a"/> alone. No string is
        ///     longer than <paramref name="a"/>'s element width in code units, so when the widest requested width reaches
        ///     that element width it IS the maximum and the length scan (a read of every element) is skipped: the same
        ///     descriptor, one pass over the text fewer (a 1M-element <c>U</c> center was ~15 % of its time in it).
        /// </remarks>
        private static DType PadDescriptor(NDArray a, NDArray width, long[] dims, ref NDArray w64)
        {
            var loop = ResolveOne(a);
            if (NDTextLoop.Size(dims) == 0)
                throw ZeroSizeMaximum();
            if (width.typecode == NPTypeCode.UInt64)
            {
                // int64 against uint64 is float64 in NumPy: the descriptor is a float and never parses.
                long maxLen = MaxLength(a, loop);
                using var wmax = np.amax(width);
                double largest = Math.Max(maxLen, Convert.ToDouble(wmax.Storage.GetAtIndex(0)));
                using var l = NDArray.Scalar(largest);
                np.strings.OutputDescriptor(a.dtype, l);
                throw new InvalidOperationException("A float size descriptor parsed.");
            }
            w64 = width.typecode == NPTypeCode.Int64 ? width : width.astype(np.int64);
            using var wm = np.amax(w64);
            long maxWidth = Convert.ToInt64(wm.Storage.GetAtIndex(0));
            long elementUnits = a.dtype.itemsize / UnitOf(loop);
            long largestW = maxWidth >= elementUnits ? maxWidth : Math.Max(MaxLength(a, loop), maxWidth);
            return np.strings.OutputDescriptor(a.dtype, largestW, floatSizes: false);
        }

        /// <summary>The longest string of a fixed-width operand (0 for an empty one), through the pad size kernel.</summary>
        /// <param name="a">The operand.</param>
        /// <param name="loop">Its loop class.</param>
        /// <returns>The maximum length in units.</returns>
        private static long MaxLength(NDArray a, TextLoop loop)
        {
            if (a.size == 0)
                return 0;
            long* acc = stackalloc long[1];
            acc[0] = 0;
            using var zero = NDArray.Scalar(0L);
            var key = new TextMethodKey(TextMethodOp.PadSize, UnitOf(loop), a.dtype.itemsize);
            NDTextLoop.ExecuteInputs(key, new[] { a, zero }, acc);
            return acc[0];
        }

        /// <summary>NumPy's reduction error for the maximum of an empty size array.</summary>
        /// <returns>The exception (not thrown).</returns>
        internal static ValueError ZeroSizeMaximum() => new ValueError("zero-size array to reduction operation maximum which has no identity");

        // =============================================================================================================
        // expandtabs (S / U)
        // =============================================================================================================

        /// <summary>
        ///     The fixed-width part of NumPy's <c>expandtabs</c>: <c>buffersizes = _expandtabs_length(a, tabsize)</c> (its
        ///     loop resolution, int64 cast, broadcast and OverflowError), the descriptor from <c>buffersizes.max()</c>, the
        ///     out= of <c>np.empty_like(a, shape=buffersizes.shape, …)</c>, then <c>_expandtabs</c>.
        /// </summary>
        /// <param name="a">The operand (<c>np.asanyarray</c>'d; not <c>StringDType</c>).</param>
        /// <param name="tabsize">The tab sizes (<c>np.asanyarray</c>'d).</param>
        /// <returns>The expanded strings.</returns>
        /// <exception cref="TypeError">A non-text operand (<c>_expandtabs_length</c>'s no-loop text).</exception>
        /// <exception cref="ArgumentException">A tab size that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="OverflowException">An expansion past 2^31 - 1 (<c>new string is too long</c>).</exception>
        /// <exception cref="ValueError">An empty broadcast.</exception>
        internal static NDArray ExpandTabs(NDArray a, NDArray tabsize)
        {
            var loop = ResolveOne(a);
            if (loop is TextLoop.None or TextLoop.VString)
                throw NoLoop("_expandtabs_length", a, tabsize);
            NDArray t64 = null;
            try
            {
                t64 = LoopInt64("_expandtabs_length", 1, tabsize, 8);
                var dims = NDTextLoop.BroadcastDims(a, t64);
                long* acc = stackalloc long[2];
                acc[0] = 0;
                acc[1] = 0;
                var sizeKey = new TextMethodKey(TextMethodOp.ExpandTabsSize, UnitOf(loop), a.dtype.itemsize);
                NDTextLoop.ExecuteInputs(sizeKey, new[] { a, t64 }, acc);
                if (acc[1] != 0)
                    throw new OverflowException("new string is too long");
                if (NDTextLoop.Size(dims) == 0)
                    throw ZeroSizeMaximum();
                var outDescr = np.strings.OutputDescriptor(a.dtype, acc[0], floatSizes: false);
                var @out = NDArray.ForOverwrite(outDescr, Shape.NewLikeLayout(a.Shape, 'K', dims));
                try
                {
                    var key = new TextMethodKey(TextMethodOp.ExpandTabs, UnitOf(loop), a.dtype.itemsize, outDescr.itemsize);
                    NDTextLoop.Execute(key, new[] { a, t64, @out }, 2, null, null, callerOut: false);
                    return @out;
                }
                catch
                {
                    @out.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(t64, tabsize);
            }
        }

        // =============================================================================================================
        // replace (S / U)
        // =============================================================================================================

        /// <summary>
        ///     The fixed-width part of NumPy's <c>replace</c> after its T branch, statement for statement where a statement
        ///     can fail: <c>old.astype(old_dtype or a_dt)</c>, <c>new.astype(...)</c>, <c>_count_ufunc(arr, old, 0, MAX)</c>,
        ///     <c>np.minimum(counts, count)</c>, <c>str_len(new) - str_len(old)</c>, <c>counts * …</c>,
        ///     <c>str_len(arr) + …</c>, <c>buffersizes.max()</c>, the descriptor, then <c>_replace</c>'s loop resolution and
        ///     the loop. The sizes themselves are one fused pass.
        /// </summary>
        /// <param name="arr">The operand (<c>np.asanyarray</c>'d; not <c>StringDType</c>).</param>
        /// <param name="old">The searched strings (<c>np.asanyarray</c>'d).</param>
        /// <param name="new">The replacements (<c>np.asanyarray</c>'d).</param>
        /// <param name="oldHasDtype">False when the caller's <c>old</c> was a Python str / bytes / number (no
        ///     <c>dtype</c> attribute): NumPy then casts it to <c>arr</c>'s descriptor, truncating to its width.</param>
        /// <param name="newHasDtype">The same for <c>new</c>.</param>
        /// <param name="count">The integer counts (<c>np.asanyarray</c>'d).</param>
        /// <returns>The replaced strings, laid out like <paramref name="arr"/>.</returns>
        /// <exception cref="TypeError">No loop (<c>count</c>'s, <c>str_len</c>'s or <c>_replace</c>'s text); a float
        ///     descriptor (a uint64 count).</exception>
        /// <exception cref="ValueError">A numeric operand (the casts' parse error); an empty broadcast.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII <c>old</c> / <c>new</c> cast to <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">A byte ≥ 0x80 cast to <c>U</c>.</exception>
        /// <exception cref="IncorrectShapeException">A statement's operands do not broadcast (its own shapes listed).</exception>
        internal static NDArray Replace(NDArray arr, NDArray old, NDArray @new, bool oldHasDtype, bool newHasDtype, NDArray count)
        {
            var aDt = arr.dtype;
            NDArray o = null, n = null, c64 = null;
            try
            {
                // old = old.astype(old_dtype or a_dt, copy=False); new = new.astype(new_dtype or a_dt, copy=False)
                o = oldHasDtype || old.dtype == aDt ? old : old.astype(aDt);
                n = newHasDtype || @new.dtype == aDt ? @new : @new.astype(aDt);
                // counts = _count_ufunc(arr, old, 0, max_int64)
                var loop = ResolvePair(arr, o);
                if (loop is TextLoop.None or TextLoop.VString)
                    throw NoLoop("count", arr, o, null, null);
                using (var zero = NDArray.Scalar(0L))
                using (var max = NDArray.Scalar(Max))
                    NDTextLoop.BroadcastDims(arr, o, zero, max);
                long[] countsDims = NDTextLoop.BroadcastDims(arr, o);
                // counts = np.where(count < 0, counts, np.minimum(counts, count)): np.minimum names (counts, count).
                long[] minDims = BroadcastShapes(countsDims, count);
                bool floatSizes = count.typecode == NPTypeCode.UInt64;
                // buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
                if (ResolveOne(n) is TextLoop.None)
                    throw new TypeError($"ufunc 'str_len' did not contain a loop with signature matching types {NDStringUfuncs.DispatchClass(n)} -> None");
                long[] diffDims = BroadcastShapes(n.Shape.dimensions ?? Array.Empty<long>(), o);
                long[] prodDims = BroadcastShapes(minDims, diffDims);
                long[] dims = BroadcastShapes(arr.Shape.dimensions ?? Array.Empty<long>(), prodDims);
                if (NDTextLoop.Size(dims) == 0)
                    throw ZeroSizeMaximum();

                int unit = UnitOf(loop);
                if (ResolveOne(n) != loop)
                {
                    // A new of the other text class: NumPy sizes the out= with str_len over new's own class, builds the
                    // descriptor, and only then finds no _replace loop. The rare path runs NumPy's statements literally
                    // (no shape error is left: the checks above ran them in order).
                    using (var largest = MixedReplaceLargest(arr, o, n, count))
                        np.strings.OutputDescriptor(aDt, largest);
                    throw new TypeError(
                        $"ufunc '_replace' did not contain a loop with signature matching types ({NDStringUfuncs.DispatchClass(arr)}, " +
                        $"{NDStringUfuncs.DispatchClass(o)}, {NDStringUfuncs.DispatchClass(n)}, {DType.From(NPTypeCode.Int64).Meta}) -> None");
                }
                c64 = count.typecode == NPTypeCode.Int64 ? count : count.astype(np.int64);
                long* acc = stackalloc long[1];
                acc[0] = long.MinValue;
                var sizeKey = new TextMethodKey(TextMethodOp.ReplaceSize, unit, arr.dtype.itemsize, o.dtype.itemsize, n.dtype.itemsize);
                NDTextLoop.ExecuteInputs(sizeKey, new[] { arr, o, n, c64 }, acc);
                var outDescr = np.strings.OutputDescriptor(aDt, acc[0], floatSizes);
                var @out = NDArray.ForOverwrite(outDescr, Shape.NewLikeLayout(arr.Shape, 'K', dims));
                try
                {
                    var key = new TextMethodKey(TextMethodOp.Replace, unit, arr.dtype.itemsize, o.dtype.itemsize, n.dtype.itemsize, outDescr.itemsize);
                    NDTextLoop.Execute(key, new[] { arr, o, n, c64, @out }, 4, null, null, callerOut: false);
                    return @out;
                }
                catch
                {
                    @out.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(o, old);
                DisposeIfTemp(n, @new);
                DisposeIfTemp(c64, count);
            }
        }

        /// <summary>
        ///     NumPy's sizing statements of <c>replace</c> over a <paramref name="new"/> of the OTHER text class, literally:
        ///     <c>counts = _count_ufunc(arr, old, 0, MAX)</c>, <c>counts = np.where(count &lt; 0, counts,
        ///     np.minimum(counts, count))</c>, <c>buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))</c>,
        ///     <c>buffersizes.max()</c> — each length over its operand's own class (a cast of <paramref name="new"/> could
        ///     raise a codec error NumPy never meets).
        /// </summary>
        /// <param name="arr">The text operand.</param>
        /// <param name="old">The searched strings (cast to <paramref name="arr"/>'s class).</param>
        /// <param name="new">The replacements, of the other text class.</param>
        /// <param name="count">The integer counts.</param>
        /// <returns>The 0-d maximum (int64, or float64 after a uint64 count).</returns>
        /// <exception cref="ValueError">An empty broadcast (the checks before this call already raised it).</exception>
        /// <remarks>Only the path that ends in NumPy's <c>_replace</c> no-loop error runs this; the shape checks ran first in
        ///     NumPy's order, so no statement here can fail on shapes.</remarks>
        private static NDArray MixedReplaceLargest(NDArray arr, NDArray old, NDArray @new, NDArray count)
        {
            using var found = FindLike("count", TextMethodOp.Count, arr, old, null, null);
            using var negative = np.less(count, 0);
            using var limited = np.minimum(found, count);
            using var counts = np.where(negative, found, limited);
            using var lengthNew = StrLen(@new, null, null, null);
            using var lengthOld = StrLen(old, null, null, null);
            using var diff = np.subtract(lengthNew, lengthOld);
            using var growth = np.multiply(counts, diff);
            using var lengthArr = StrLen(arr, null, null, null);
            using var sizes = np.add(lengthArr, growth);
            return np.amax(sizes);
        }

        /// <summary>
        ///     The broadcast of a dimension list with an operand's shape, raising the ufunc text with the two shapes in that
        ///     order (a binary numeric ufunc's statement: <c>np.minimum(counts, count)</c>, <c>str_len(new) - str_len(old)</c>,
        ///     …).
        /// </summary>
        /// <param name="first">The first operand's dimensions.</param>
        /// <param name="second">The second operand.</param>
        /// <returns>The broadcast dimensions.</returns>
        /// <exception cref="IncorrectShapeException">They do not broadcast.</exception>
        private static long[] BroadcastShapes(long[] first, NDArray second) => BroadcastShapes(first, second.Shape.dimensions ?? Array.Empty<long>());

        /// <summary>The broadcast of two dimension lists, raising the ufunc text with both shapes.</summary>
        /// <param name="first">The first dimensions.</param>
        /// <param name="second">The second dimensions.</param>
        /// <returns>The broadcast dimensions.</returns>
        /// <exception cref="IncorrectShapeException">They do not broadcast.</exception>
        private static long[] BroadcastShapes(long[] first, long[] second)
        {
            int nd = Math.Max(first.Length, second.Length);
            var dims = new long[nd];
            for (int d = 0; d < nd; d++)
            {
                long a = d - (nd - first.Length) >= 0 ? first[d - (nd - first.Length)] : 1;
                long b = d - (nd - second.Length) >= 0 ? second[d - (nd - second.Length)] : 1;
                if (a != b && a != 1 && b != 1)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with shapes {Shape.Adopt((long[])first.Clone()).ToPythonTuple()} " +
                        $"{Shape.Adopt((long[])second.Clone()).ToPythonTuple()} ");
                dims[d] = a == 1 ? b : a;
            }
            return dims;
        }

        // =============================================================================================================
        // partition / rpartition (S / U)
        // =============================================================================================================

        /// <summary>
        ///     The fixed-width part of NumPy's <c>partition</c> / <c>rpartition</c> after its T branch:
        ///     <c>sep.astype(a.dtype)</c> (truncated to <paramref name="a"/>'s width), <c>_find_ufunc</c> /
        ///     <c>_rfind_ufunc</c>'s resolution and broadcast, the three widths of the structured out= (the before-part's
        ///     maximum, 1 when no separator is found anywhere else the separators' longest, the after-part's maximum), the
        ///     loop's <c>ValueError: empty separator</c>, then the three parts.
        /// </summary>
        /// <param name="a">The operand (<c>np.asanyarray</c>'d; not <c>StringDType</c>).</param>
        /// <param name="sep">The separators (<c>np.asanyarray</c>'d).</param>
        /// <param name="back">True for <c>rpartition</c>.</param>
        /// <returns>The three parts, each laid out like <paramref name="a"/>.</returns>
        /// <exception cref="TypeError">No <c>find</c> loop for the operands.</exception>
        /// <exception cref="ValueError">A numeric operand (the cast); an empty broadcast; an empty separator.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <remarks>NumPy returns three field views of one structured array; NumSharp has no structured dtype, so the parts
        ///     are three arrays (their values and descriptors NumPy's, their strides and OWNDATA not), and a part NumPy sizes
        ///     at width 0 (<c>'U0'</c>) is width 1.</remarks>
        internal static (NDArray Before, NDArray Separator, NDArray After) Partition(NDArray a, NDArray sep, bool back)
        {
            var s = sep.dtype == a.dtype ? sep : sep.astype(a.dtype);
            try
            {
                string find = back ? "rfind" : "find";
                var loop = ResolvePair(a, s);
                if (loop is TextLoop.None or TextLoop.VString)
                    throw NoLoop(find, a, s, null, null);
                long[] dims;
                using (var zero = NDArray.Scalar(0L))
                using (var max = NDArray.Scalar(Max))
                    dims = NDTextLoop.BroadcastDims(a, s, zero, max);
                long* acc = stackalloc long[5];
                acc[0] = acc[1] = acc[2] = 0;
                acc[3] = 1;
                acc[4] = 0;
                int unit = UnitOf(loop);
                var sizeKey = new TextMethodKey(TextMethodOp.PartitionSize, unit, a.dtype.itemsize, s.dtype.itemsize, Extra: back ? 1 : 0);
                NDTextLoop.ExecuteInputs(sizeKey, new[] { a, s }, acc);
                if (NDTextLoop.Size(dims) == 0)
                    throw ZeroSizeMaximum();
                var meta = (StringDTypeMeta)a.dtype.Meta;
                var d1 = meta.Descr(Math.Max(1, acc[0]));
                var d2 = meta.Descr(Math.Max(1, acc[3] != 0 ? 1 : acc[1]));
                var d3 = meta.Descr(Math.Max(1, acc[2]));
                if (acc[4] != 0)
                    throw new ValueError("empty separator");
                var layout = Shape.NewLikeLayout(a.Shape, 'K', dims);
                NDArray o1 = null, o2 = null, o3 = null;
                try
                {
                    o1 = NDArray.ForOverwrite(d1, layout);
                    o2 = NDArray.ForOverwrite(d2, Shape.NewLikeLayout(a.Shape, 'K', dims));
                    o3 = NDArray.ForOverwrite(d3, Shape.NewLikeLayout(a.Shape, 'K', dims));
                    var key = new TextMethodKey(TextMethodOp.Partition, unit, a.dtype.itemsize, s.dtype.itemsize, d1.itemsize, d2.itemsize,
                                                d3.itemsize, back ? 1 : 0);
                    NDTextLoop.Execute(key, new[] { a, s, o1, o2, o3 }, 2, null, null, callerOut: false);
                    return (o1, o2, o3);
                }
                catch
                {
                    o1?.Dispose();
                    o2?.Dispose();
                    o3?.Dispose();
                    throw;
                }
            }
            finally
            {
                DisposeIfTemp(s, sep);
            }
        }
    }
}
