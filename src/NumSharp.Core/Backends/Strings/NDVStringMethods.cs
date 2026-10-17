using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// NDVStringMethods.cs — NumPy's StringDType (T) loops of numpy.strings (strings phase 2, P2.5)
// =============================================================================
//
// WHAT THIS IS
//   The StringDType half of P2.5: the ufunc-level resolution of each T call (the promoters' casts of a U operand to
//   StringDType(), the NA compatibility check of resolve_descriptors, the output descriptor of each loop), and the
//   per-element bodies — literal ports of the Buffer<ENCODING::UTF8> routines string_buffer.h runs for a T operand,
//   together with the NA handling of each strided loop of stringdtype_ufuncs.cpp. The bodies are called by the IL shell
//   (ILKernelGenerator.Text.Methods.cs, the V* operations) with the call's VStringMethodAux.
//
// NUMPY'S UTF-8 RULES THE PORTS KEEP (probed against 2.4.2; several differ from the S / U loops)
//   * Lengths and positions are in CODE POINTS, and num_codepoints ignores TRAILING NULs (a T string keeps them:
//     'ab\0' has length 2).
//   * The searches run on the needle's FULL bytes, trailing NULs included (find(T 'xbc\0', T 'bc\0') is -1), except a
//     one-code-point ASCII needle, which find / rfind search as a single byte; count always searches the full bytes.
//   * The strip loops start the trailing scan at the LAST BYTE of the buffer (a trailing NUL) while counting code points
//     without the trailing NULs, so rstrip('a\0\0\0') is 'a\0\0'; the chars form looks a one-byte character up in the
//     first len(chars) BYTES of the set.
//   * center / ljust / rjust copy the whole buffer, then step over len(text) code points and pad from there, so a
//     trailing NUL is overwritten by the right padding ('ab\0'.center(6, '*') is '**ab**\0').
//   * _slice with step 1 runs to the buffer's END (trailing NULs kept) when the stop passes the code points; another
//     step selects code points only.
//   * replace and partition copy the buffers whole (trailing NULs kept); expandtabs writes only the code points.
//   Each NA rule is the loop's own (the first operand's descriptor decides): see each body.
//
// WHERE NUMPY READS OUT OF BOUNDS (the intended result is kept instead)
//   * startswith / endswith compare the needle's full bytes, so a needle with trailing NULs reads past the haystack's
//     text: bytes there read as zero here (a short cell's padding; NumPy reads whatever follows an arena or heap text).
//   * zfill of an empty text reads the byte after the buffer for a sign: no sign here.
//   * center / ljust / rjust leave the trailing-NUL tail of the output unwritten (newly allocated memory): zero here.
//   * replace of an old text that is all NULs sizes its buffer by bytes and writes past it: the exact size here.
//
// OUTPUTS (NumPy's load_new_string)
//   Every T output is a fresh array (all-zero cells). A result is written straight into its cell (up to 15 bytes) or
//   into the output allocator's arena / heap through NewString — the cell bytes NumPy's NpyString_newemptysize leaves
//   — and then filled in place, with no intermediate buffer.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The per-call state of a <c>StringDType</c> <c>numpy.strings</c> kernel, passed as the kernel's auxdata: the input
    ///     arenas, the deciding descriptor's NA rule and default string, the first error the loop met, a scratch buffer
    ///     (the code-point index of a stepped slice), and the output allocators.
    /// </summary>
    /// <remarks>
    ///     The arena bases are read once with every allocator locked: the inputs' allocators never allocate during the call
    ///     (the outputs are fresh arrays with allocators of their own), so the bases stay valid. The scratch buffer is
    ///     grown by <see cref="NDVStringMethods.Scratch"/> and freed by the driver.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct VStringMethodAux
    {
        /// <summary>The arena base of the first text input's allocator (null when it has none).</summary>
        internal byte* ArenaA;

        /// <summary>The arena base of the second text input's allocator.</summary>
        internal byte* ArenaB;

        /// <summary>The arena base of the third text input's allocator.</summary>
        internal byte* ArenaC;

        /// <summary>The deciding descriptor's default string (UTF-8), or null when empty.</summary>
        internal byte* DefaultText;

        /// <summary>The byte length of <see cref="DefaultText"/>.</summary>
        internal long DefaultSize;

        /// <summary>The deciding descriptor's NA rule (NumPy's <c>has_null</c> / <c>has_string_na</c> / <c>has_nan_na</c>).</summary>
        internal VStringNARule Rule;

        /// <summary>The first error the loop met (<see cref="VStringMethodError"/>), 0 for none.</summary>
        internal int Status;

        /// <summary>A scratch buffer (owned by the driver; freed after the call).</summary>
        internal byte* ScratchBuffer;

        /// <summary>The scratch buffer's capacity in bytes.</summary>
        internal long ScratchCapacity;

        /// <summary>The <see cref="GCHandle"/> of the first output's allocator (0 for a numeric output).</summary>
        internal IntPtr Out0;

        /// <summary>The <see cref="GCHandle"/> of the second output's allocator (partition).</summary>
        internal IntPtr Out1;

        /// <summary>The <see cref="GCHandle"/> of the third output's allocator (partition).</summary>
        internal IntPtr Out2;
    }

    /// <summary>The errors a <c>StringDType</c> loop records; the driver raises NumPy's exception for the first one.</summary>
    internal enum VStringMethodError
    {
        /// <summary>No error.</summary>
        None = 0,

        /// <summary><c>ValueError: The length of a null string is undefined</c> (str_len).</summary>
        LengthOfNull,

        /// <summary><c>ValueError: '{ufunc}' not supported for null values that are not strings.</c> (find family).</summary>
        FindNull,

        /// <summary><c>ValueError: '{ufunc}' not supported for null values that are not nan-like or strings.</c></summary>
        TailNull,

        /// <summary><c>ValueError: Can only strip null values that are strings or NaN-like values</c>.</summary>
        StripNull,

        /// <summary><c>ValueError: Cannot use a null string that is not a string as the {ufunc} delimiter</c>.</summary>
        StripDelimiterNull,

        /// <summary><c>ValueError: Null values are not supported as search patterns or replacement strings for replace</c>.</summary>
        ReplacePatternNull,

        /// <summary><c>ValueError: Only string or NaN-like null strings can be used as search strings for replace</c>.</summary>
        ReplaceSearchNull,

        /// <summary><c>ValueError: Null values are not supported arguments for expandtabs</c>.</summary>
        ExpandTabsNull,

        /// <summary><c>OverflowError: new string is too long</c> (expandtabs).</summary>
        ExpandTabsOverflow,

        /// <summary><c>ValueError: Cannot {ufunc} null that is not a nan-like value</c> (center / ljust / rjust).</summary>
        PadNull,

        /// <summary><c>OverflowError: Overflow encountered in {ufunc}</c> (center / ljust / rjust).</summary>
        PadOverflow,

        /// <summary><c>TypeError: Cannot zfill null string that is not a nan-like value</c>.</summary>
        ZFillNull,

        /// <summary><c>ValueError: Null values are not supported in {ufunc}</c> (partition).</summary>
        PartitionNull,

        /// <summary><c>ValueError: empty separator</c> (partition).</summary>
        EmptySeparator,

        /// <summary><c>TypeError: Cannot slice null string</c>.</summary>
        SliceNull,

        /// <summary><c>MemoryError: Failed to allocate string in {ufunc}</c>.</summary>
        Memory,

        /// <summary><c>MemoryError: Failed to load string in {ufunc}</c> (a corrupt arena cell).</summary>
        Load,

        /// <summary><c>ValueError: substring not found</c> (index / rindex).</summary>
        NotFound,
    }

    /// <summary>
    ///     NumPy's <c>StringDType</c> loops of <c>numpy.strings</c>: the ufunc-level resolution and drivers, and the UTF-8
    ///     element bodies (see the file header).
    /// </summary>
    internal static unsafe partial class NDVStringMethods
    {
        /// <summary>The flags byte of a short cell, its size nibble cleared (<c>INITIALIZED | OUTSIDE_ARENA</c>).</summary>
        private const byte ShortFlags = VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA;

        /// <summary>The longest text a cell holds in place.</summary>
        private const int ShortMax = VStringAllocator.NPY_SHORT_STRING_MAX_SIZE;

        /// <summary>The bytes of a cell (the "width" a kernel key records for a <c>StringDType</c> operand).</summary>
        private const int Cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;

        /// <summary>
        ///     The bytes one span search covers at most: 2^30, so a window plus a needle of up to 2^30 bytes stays within a
        ///     span's <see cref="int"/> length (a <c>StringDType</c> text can be longer than any span).
        /// </summary>
        private const long SearchWindow = 1L << 30;

        // =============================================================================================================
        // The drivers (ufunc level)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>str_len</c> over <c>StringDType</c> (<c>string_strlen_strided_loop</c>): code points without the
        ///     trailing NULs; an NA reads as the default string under a string NA, and is the error
        ///     <c>The length of a null string is undefined</c> otherwise.
        /// </summary>
        /// <param name="x">The <c>StringDType</c> operand.</param>
        /// <param name="out">The int64 (or castable) output, or null.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <returns><paramref name="out"/>, or a fresh int64 array.</returns>
        /// <exception cref="ValueError">An NA under a non-string NA rule (only a selected element raises).</exception>
        /// <remarks>
        ///     A mask runs NDIter's per-run calls, never the masked blend of the fixed-width loops: the blend computes every
        ///     element, and an NA under a non-string rule in a masked-off position must not raise (NumPy's masked loop never
        ///     visits it). For the same reason an <paramref name="out"/> of another dtype is filled from a temporary computed
        ///     under the mask.
        /// </remarks>
        internal static NDArray StrLen(NDArray x, NDArray @out, NDArray where)
        {
            if (@out is not null && @out.typecode != NPTypeCode.Int64)
                return NDStringArithmetic.CastInto(@out, where, () => StrLen(x, null, where));
            var dims = NDTextLoop.BroadcastDims(x, @out, where);
            var int64 = DType.From(NPTypeCode.Int64);
            NDArray result = @out ?? (where is null
                ? NDArray.ForOverwrite(int64, NDStringMethods.UfuncLayout(dims, x))
                : new NDArray(int64, NDStringMethods.UfuncLayout(dims, x, where), true));
            try
            {
                RunLocked(new[] { x }, Array.Empty<NDArray>(), x.dtype, aux =>
                {
                    var key = new TextMethodKey(TextMethodOp.VLength, 1, Cell);
                    NDTextLoop.Execute(key, new[] { x, result }, 1, where, (void*)aux, callerOut: @out is not null);
                }, "str_len");
                return result;
            }
            catch
            {
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's find-like ufuncs over <c>StringDType</c> (<c>string_findlike_strided_loop</c> /
        ///     <c>string_startswith_endswith_strided_loop</c>): the <c>U</c> operand cast to <c>StringDType()</c>, the first
        ///     operand's NA rule, the loop. The caller already ran <see cref="CheckPairNA"/> (before the int64 casts, NumPy's
        ///     order) and the broadcast.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="op">The fixed-width operation (mapped to its <c>StringDType</c> form).</param>
        /// <param name="a">The searched operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="sub">The searched-for operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="s64">The int64 starts.</param>
        /// <param name="e64">The int64 ends.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <returns>A fresh int64 (bool for the tail matches) result.</returns>
        /// <exception cref="ValueError">An NA the loop's rule refuses; <c>index</c> / <c>rindex</c> found no match.</exception>
        internal static NDArray FindLike(string ufunc, TextMethodOp op, NDArray a, NDArray sub, NDArray s64, NDArray e64, long[] dims)
        {
            NDArray t1 = null, t2 = null;
            try
            {
                t1 = Promote(a);
                t2 = Promote(sub);
                bool tail = op is TextMethodOp.StartsWith or TextMethodOp.EndsWith;
                var result = NDArray.ForOverwrite(tail ? DType.Boolean : DType.From(NPTypeCode.Int64),
                                                  NDStringMethods.UfuncLayout(dims, a, sub, s64, e64));
                try
                {
                    var vop = op switch
                    {
                        TextMethodOp.Find => TextMethodOp.VFind,
                        TextMethodOp.RFind => TextMethodOp.VRFind,
                        TextMethodOp.Count => TextMethodOp.VCount,
                        TextMethodOp.StartsWith => TextMethodOp.VStartsWith,
                        _ => TextMethodOp.VEndsWith,
                    };
                    bool index = ufunc is "index" or "rindex";
                    RunLocked(new[] { t1, t2 }, Array.Empty<NDArray>(), t1.dtype, aux =>
                    {
                        var key = new TextMethodKey(vop, 1, Cell, Cell, Extra: index ? 1 : 0);
                        NDTextLoop.Execute(key, new[] { t1, t2, s64, e64, result }, 4, null, (void*)aux, callerOut: false);
                    }, ufunc);
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
                NDStringMethods.DisposeIfTemp(t1, a);
                NDStringMethods.DisposeIfTemp(t2, sub);
            }
        }

        /// <summary>
        ///     NumPy's whitespace strip ufuncs over <c>StringDType</c> (<c>string_lrstrip_whitespace_strided_loop</c>): the
        ///     output keeps the operand's NA object and coercion.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="type">The strip type.</param>
        /// <param name="a">The <c>StringDType</c> operand.</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="ValueError">An NA that is neither NaN-like nor a string.</exception>
        internal static NDArray StripWhitespace(string ufunc, int type, NDArray a)
        {
            var dims = NDTextLoop.BroadcastDims(a);
            var result = new NDArray(SameNA(a.dtype), NDStringMethods.UfuncLayout(dims, a), true);
            try
            {
                RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
                {
                    var key = new TextMethodKey(TextMethodOp.VStripWhitespace, 1, Cell, Cell, Extra: type);
                    NDTextLoop.Execute(key, new[] { a, result }, 1, null, (void*)aux, callerOut: false);
                }, ufunc);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>chars</c> strip ufuncs over <c>StringDType</c> (<c>string_lrstrip_chars_strided_loop</c>, output by
        ///     <c>binary_resolve_descriptors</c>: the compatible NA object, the first descriptor's coercion). The caller
        ///     already ran <see cref="CheckPairNA"/> and the broadcast.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="type">The strip type.</param>
        /// <param name="a">The text operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="chars">The strip-set operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="ValueError">An NA the loop's rule refuses.</exception>
        internal static NDArray StripChars(string ufunc, int type, NDArray a, NDArray chars, long[] dims)
        {
            NDArray t1 = null, t2 = null;
            try
            {
                t1 = Promote(a);
                t2 = Promote(chars);
                var outDescr = NDVStringArithmetic.BinaryOutputDescr(t1.dtype, t2.dtype);
                var result = new NDArray(outDescr, NDStringMethods.UfuncLayout(dims, a, chars), true);
                try
                {
                    RunLocked(new[] { t1, t2 }, new[] { result }, t1.dtype, aux =>
                    {
                        var key = new TextMethodKey(TextMethodOp.VStripChars, 1, Cell, Cell, Cell, Extra: type);
                        NDTextLoop.Execute(key, new[] { t1, t2, result }, 2, null, (void*)aux, callerOut: false);
                    }, ufunc);
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
                NDStringMethods.DisposeIfTemp(t1, a);
                NDStringMethods.DisposeIfTemp(t2, chars);
            }
        }

        /// <summary>NumPy's <c>_slice</c> over <c>StringDType</c> (<c>slice_strided_loop</c>; the output keeps the NA object).</summary>
        /// <param name="a">The <c>StringDType</c> operand.</param>
        /// <param name="start">The int64 starts.</param>
        /// <param name="stop">The int64 stops.</param>
        /// <param name="step">The int64 steps.</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="TypeError">An NA element (<c>Cannot slice null string</c>).</exception>
        internal static NDArray Slice(NDArray a, NDArray start, NDArray stop, NDArray step, long[] dims)
        {
            var result = new NDArray(SameNA(a.dtype), NDStringMethods.UfuncLayout(dims, a, start, stop, step), true);
            try
            {
                RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
                {
                    var key = new TextMethodKey(TextMethodOp.VSlice, 1, Cell, Cell);
                    NDTextLoop.Execute(key, new[] { a, start, stop, step, result }, 4, null, (void*)aux, callerOut: false);
                }, "slice");
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>_center</c> / <c>_ljust</c> / <c>_rjust</c> over <c>StringDType</c>: the promoter's loop (a
        ///     <c>U</c> operand and fill cast to <c>StringDType()</c>; an <c>S</c> one has no loop), resolve_descriptors' NA
        ///     check (the output takes the compatible NA object, coercion of both), the width cast to int64, the broadcast,
        ///     the loop.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name (in the no-loop text and the loop's errors).</param>
        /// <param name="pos">The justification.</param>
        /// <param name="a">The text operand (<c>np.asanyarray</c>'d).</param>
        /// <param name="width">The widths (an integer array: the wrapper checked).</param>
        /// <param name="fillchar">The fill operand (<c>np.asanyarray</c>'d; one character, the wrapper checked).</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="TypeError">No loop (an <c>S</c> operand); two different NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An NA the loop's rule refuses.</exception>
        /// <exception cref="OverflowException">A padded size past <c>sys.maxsize</c>.</exception>
        /// <exception cref="OutOfMemoryException">A padded size no string can hold (NumPy's MemoryError).</exception>
        internal static NDArray Pad(string ufunc, int pos, NDArray a, NDArray width, NDArray fillchar)
        {
            if (!IsTextT(a) || !IsTextT(fillchar))
                throw NDStringMethods.NoLoop(ufunc, a, width, fillchar);
            CheckCompatibleNA(LoopDescr(a), LoopDescr(fillchar), out bool hasNA, out object na);
            NDArray t1 = null, t3 = null, w64 = null;
            try
            {
                w64 = NDStringMethods.LoopInt64(ufunc, 1, width, 0);
                var dims = NDTextLoop.BroadcastDims(a, w64, fillchar);
                t1 = Promote(a);
                t3 = Promote(fillchar);
                var outDescr = DTypeRegistry.VString.Descr(hasNA, na, t1.dtype.VString.coerce && t3.dtype.VString.coerce);
                var result = new NDArray(outDescr, NDStringMethods.UfuncLayout(dims, a, w64, fillchar), true);
                try
                {
                    RunLocked(new[] { t1, t3 }, new[] { result }, t1.dtype, aux =>
                    {
                        var key = new TextMethodKey(TextMethodOp.VPad, 1, Cell, Cell, Cell, Extra: pos);
                        NDTextLoop.Execute(key, new[] { t1, w64, t3, result }, 3, null, (void*)aux, callerOut: false);
                    }, ufunc);
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
                NDStringMethods.DisposeIfTemp(t1, a);
                NDStringMethods.DisposeIfTemp(t3, fillchar);
                NDStringMethods.DisposeIfTemp(w64, width);
            }
        }

        /// <summary>NumPy's <c>_zfill</c> over <c>StringDType</c> (the output keeps the operand's NA object and coercion).</summary>
        /// <param name="a">The <c>StringDType</c> operand.</param>
        /// <param name="width">The widths (an integer array).</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="TypeError">An NA that is neither NaN-like nor a string.</exception>
        /// <exception cref="OutOfMemoryException">A padded size no string can hold.</exception>
        internal static NDArray ZFill(NDArray a, NDArray width)
        {
            RequireMultiplyPromoterInteger("_zfill", a, width);
            NDArray w64 = null;
            try
            {
                w64 = NDStringMethods.LoopInt64("_zfill", 1, width, 0);
                var dims = NDTextLoop.BroadcastDims(a, w64);
                var result = new NDArray(SameNA(a.dtype), NDStringMethods.UfuncLayout(dims, a, w64), true);
                try
                {
                    RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
                    {
                        var key = new TextMethodKey(TextMethodOp.VZFill, 1, Cell, Cell);
                        NDTextLoop.Execute(key, new[] { a, w64, result }, 2, null, (void*)aux, callerOut: false);
                    }, "zfill");
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
                NDStringMethods.DisposeIfTemp(w64, width);
            }
        }

        /// <summary>
        ///     NumPy's <c>_replace</c> over <c>StringDType</c> (<c>string_replace_strided_loop</c>): the promoter's loop
        ///     (<c>U</c> operands cast; an <c>S</c> one has no loop), the NA compatibility of all three text operands (coerce of
        ///     all three), the count cast to int64, the broadcast, the loop. Only -1 means every occurrence: another negative
        ///     count replaces nothing, as NumPy's loop takes <c>min(count, found)</c>.
        /// </summary>
        /// <param name="a">The text operand.</param>
        /// <param name="old">The searched operand.</param>
        /// <param name="new">The replacement operand.</param>
        /// <param name="count">The counts (an integer array).</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="TypeError">No loop (an <c>S</c> operand); two different NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An NA the loop refuses.</exception>
        internal static NDArray Replace(NDArray a, NDArray old, NDArray @new, NDArray count)
        {
            if (!IsTextT(a) || !IsTextT(old) || !IsTextT(@new))
                throw NDStringMethods.NoLoop("_replace", a, old, @new, count);
            // resolve_descriptors: the first two descriptors' compatible NA object, then the third's against it.
            CheckCompatibleNA(LoopDescr(a), LoopDescr(old), out bool h12, out object na12);
            var p12 = new VStringParameters(h12, na12, true);
            var p3 = LoopDescr(@new).VString;
            if (!VStringParameters.CompatibleNA(p12, p3, out bool h, out object na))
                throw IncompatibleNA(p12.na_object, p3.na_object);
            NDArray t1 = null, t2 = null, t3 = null, c64 = null;
            try
            {
                c64 = NDStringMethods.LoopInt64("_replace", 3, count, -1);
                var dims = NDTextLoop.BroadcastDims(a, old, @new, c64);
                t1 = Promote(a);
                t2 = Promote(old);
                t3 = Promote(@new);
                bool coerce = t1.dtype.VString.coerce && t2.dtype.VString.coerce && t3.dtype.VString.coerce;
                var result = new NDArray(DTypeRegistry.VString.Descr(h, na, coerce),
                                         NDStringMethods.UfuncLayout(dims, a, old, @new, c64), true);
                try
                {
                    RunLocked(new[] { t1, t2, t3 }, new[] { result }, t1.dtype, aux =>
                    {
                        var key = new TextMethodKey(TextMethodOp.VReplace, 1, Cell, Cell, Cell, Cell);
                        NDTextLoop.Execute(key, new[] { t1, t2, t3, c64, result }, 4, null, (void*)aux, callerOut: false);
                    }, "replace");
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
                NDStringMethods.DisposeIfTemp(t1, a);
                NDStringMethods.DisposeIfTemp(t2, old);
                NDStringMethods.DisposeIfTemp(t3, @new);
                NDStringMethods.DisposeIfTemp(c64, count);
            }
        }

        /// <summary>NumPy's <c>_expandtabs</c> over <c>StringDType</c> (the output keeps the operand's NA object).</summary>
        /// <param name="a">The <c>StringDType</c> operand.</param>
        /// <param name="tabsize">The tab sizes (<c>np.asanyarray</c>'d).</param>
        /// <returns>A fresh <c>StringDType</c> result.</returns>
        /// <exception cref="ArgumentException">A tab size that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A tab-size literal past int64; an expansion past 2^31 - 1.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An NA under a non-string rule.</exception>
        internal static NDArray ExpandTabs(NDArray a, NDArray tabsize)
        {
            RequireMultiplyPromoterInteger("_expandtabs", a, tabsize);
            NDArray t64 = null;
            try
            {
                t64 = NDStringMethods.LoopInt64("_expandtabs", 1, tabsize, 8);
                var dims = NDTextLoop.BroadcastDims(a, t64);
                var result = new NDArray(SameNA(a.dtype), NDStringMethods.UfuncLayout(dims, a, t64), true);
                try
                {
                    RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
                    {
                        var key = new TextMethodKey(TextMethodOp.VExpandTabs, 1, Cell, Cell);
                        NDTextLoop.Execute(key, new[] { a, t64, result }, 2, null, (void*)aux, callerOut: false);
                    }, "expandtabs");
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
                NDStringMethods.DisposeIfTemp(t64, tabsize);
            }
        }

        /// <summary>
        ///     NumPy's <c>_partition</c> / <c>_rpartition</c> over <c>StringDType</c> (<c>string_partition_strided_loop</c>):
        ///     both operands must already be <c>StringDType</c> (no promoter casts a <c>str</c> separator), the three outputs
        ///     take the compatible NA object.
        /// </summary>
        /// <param name="a">The text operand.</param>
        /// <param name="sep">The separator operand.</param>
        /// <param name="back">True for <c>_rpartition</c>.</param>
        /// <returns>The three fresh parts.</returns>
        /// <exception cref="TypeError">An operand that is not <c>StringDType</c> (NumPy's no-loop text with three outputs);
        ///     two different NA objects.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An NA under a non-string rule; an empty separator.</exception>
        internal static (NDArray Before, NDArray Separator, NDArray After) Partition(NDArray a, NDArray sep, bool back)
        {
            string ufunc = back ? "_rpartition" : "_partition";
            if (!NDStringUfuncs.IsVString(a) || !NDStringUfuncs.IsVString(sep))
                throw new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types " +
                                    $"({NDStringUfuncs.DispatchClass(a)}, {NDStringUfuncs.DispatchClass(sep)}) -> (None, None, None)");
            CheckCompatibleNA(a.dtype, sep.dtype, out bool hasNA, out object na);
            var dims = NDTextLoop.BroadcastDims(a, sep);
            var descr = DTypeRegistry.VString.Descr(hasNA, na, a.dtype.VString.coerce && sep.dtype.VString.coerce);
            NDArray o1 = null, o2 = null, o3 = null;
            try
            {
                // Each output gets its own shape (a fresh array adopts its dimension array).
                o1 = new NDArray(descr, NDStringMethods.UfuncLayout(dims, a, sep), true);
                o2 = new NDArray(descr, NDStringMethods.UfuncLayout((long[])dims.Clone(), a, sep), true);
                o3 = new NDArray(descr, NDStringMethods.UfuncLayout((long[])dims.Clone(), a, sep), true);
                RunLocked(new[] { a, sep }, new[] { o1, o2, o3 }, a.dtype, aux =>
                {
                    var key = new TextMethodKey(TextMethodOp.VPartition, 1, Cell, Cell, Cell, Cell, Cell, back ? 1 : 0);
                    NDTextLoop.Execute(key, new[] { a, sep, o1, o2, o3 }, 2, null, (void*)aux, callerOut: false);
                }, ufunc);
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

        /// <summary>
        ///     NumPy's resolve_descriptors NA check of a two-text <c>StringDType</c> loop (find family, startswith /
        ///     endswith, the chars strips): the descriptors the loop sees (a <c>U</c> operand is the promoters'
        ///     <c>StringDType()</c>) must have compatible NA objects. It runs before the integer casts and the broadcast,
        ///     NumPy's order.
        /// </summary>
        /// <param name="a">The first text operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="b">The second text operand.</param>
        /// <exception cref="TypeError">Two different NA objects: <c>Cannot find a compatible null string value for null
        ///     strings 'nan' and 'None'</c>.</exception>
        internal static void CheckPairNA(NDArray a, NDArray b) => CheckCompatibleNA(LoopDescr(a), LoopDescr(b), out _, out _);

        /// <summary>The descriptor a <c>StringDType</c> loop sees for an operand: its own, or <c>StringDType()</c> for a <c>U</c>
        ///     operand (the promoters' cast).</summary>
        /// <param name="x">The operand (<c>T</c> or <c>U</c>).</param>
        /// <returns>The descriptor.</returns>
        private static DType LoopDescr(NDArray x) => NDStringUfuncs.IsVString(x) ? x.dtype : DTypeRegistry.VString.Descr(false, null, true);

        /// <summary>
        ///     NumPy's <c>string_multiply_promoter</c>, the promoter of <c>_zfill</c> and <c>_expandtabs</c> over
        ///     <c>StringDType</c>: it makes every integer operand <c>Int64DType</c> EXCEPT a uint64 one, which it leaves as it
        ///     is (its <c>is_integer_dtype</c> skips uint64 because <c>multiply</c> has a uint64 loop — these two ufuncs do
        ///     not), so the call then finds no loop. A float / bool operand never reaches the promoter (NumPy's generic
        ///     resolution casts it, or raises the input-cast error).
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <param name="text">The <c>StringDType</c> operand.</param>
        /// <param name="count">The integer operand (a width or a tab size, <c>np.asanyarray</c>'d), or null.</param>
        /// <exception cref="TypeError">A uint64 operand: <c>ufunc '_zfill' did not contain a loop with signature matching
        ///     types (&lt;class 'numpy.dtypes.StringDType'&gt;, &lt;class 'numpy.dtypes.UInt64DType'&gt;) -&gt; None</c>.</exception>
        private static void RequireMultiplyPromoterInteger(string ufunc, NDArray text, NDArray count)
        {
            if (count is not null && !count.WasPythonLiteral && count.typecode == NPTypeCode.UInt64)
                throw NDStringMethods.NoLoop(ufunc, text, count);
        }

        /// <summary>True for a <c>StringDType</c> or <c>U</c> operand (what a StringDType promoter accepts).</summary>
        /// <param name="x">The operand.</param>
        /// <returns>True when the promoter casts or keeps it.</returns>
        private static bool IsTextT(NDArray x) => NDStringUfuncs.IsVString(x) || NDStringMethods.IsStr(x);

        /// <summary>A <c>U</c> operand cast to <c>StringDType()</c> (the promoters' cast), a <c>StringDType</c> one as it is.</summary>
        /// <param name="x">The operand.</param>
        /// <returns>The <c>StringDType</c> operand (new for a cast; the caller disposes it).</returns>
        private static NDArray Promote(NDArray x) => NDStringUfuncs.IsVString(x) ? x : NDVStringComparison.ToDefaultVString(x);

        /// <summary>A fresh descriptor with an operand's NA object and coercion (<c>new_stringdtype_instance</c>).</summary>
        /// <param name="d">The operand's descriptor.</param>
        /// <returns>The output descriptor.</returns>
        private static DType SameNA(DType d)
        {
            var p = d.VString;
            return DTypeRegistry.VString.Descr(p.HasNAObject, p.na_object, p.coerce);
        }

        /// <summary>NumPy's <c>stringdtype_compatible_na</c>, returning the output NA object.</summary>
        /// <param name="d1">The first descriptor.</param>
        /// <param name="d2">The second descriptor.</param>
        /// <param name="hasNA">Whether the output has an NA object.</param>
        /// <param name="na">The output NA object.</param>
        /// <exception cref="TypeError">Two different NA objects.</exception>
        private static void CheckCompatibleNA(DType d1, DType d2, out bool hasNA, out object na)
        {
            if (!VStringParameters.CompatibleNA(d1.VString, d2.VString, out hasNA, out na))
                throw IncompatibleNA(d1.VString.na_object, d2.VString.na_object);
        }

        /// <summary>NumPy's <c>Cannot find a compatible null string value for null strings 'nan' and 'None'</c>.</summary>
        /// <param name="a">The first NA object.</param>
        /// <param name="b">The second NA object.</param>
        /// <returns>The exception (not thrown).</returns>
        private static TypeError IncompatibleNA(object a, object b)
            => new TypeError($"Cannot find a compatible null string value for null strings '{NDPyObject.Repr(a)}' and '{NDPyObject.Repr(b)}'");

        /// <summary>
        ///     Locks every allocator the loop touches (NumPy's <c>NpyString_acquire_allocators</c>), fills the call's
        ///     <see cref="VStringMethodAux"/> (arena bases, the deciding descriptor's NA rule and default string, the output
        ///     allocators' handles), runs <paramref name="run"/>, frees the scratch buffer and raises the first error the loop
        ///     recorded.
        /// </summary>
        /// <param name="inputs">The <c>StringDType</c> text inputs, in kernel order (A, B, C).</param>
        /// <param name="outputs">The <c>StringDType</c> outputs (fresh arrays).</param>
        /// <param name="ruleDescr">The descriptor whose NA rule the loop reads.</param>
        /// <param name="run">The pass, given the auxdata pointer.</param>
        /// <param name="ufunc">NumPy's ufunc name for the errors.</param>
        /// <exception cref="Exception">The loop's recorded error (<see cref="Raise"/>).</exception>
        private static void RunLocked(NDArray[] inputs, NDArray[] outputs, DType ruleDescr, Action<IntPtr> run, string ufunc)
        {
            var locks = new List<VStringAllocator>(inputs.Length + outputs.Length);
            foreach (var x in inputs)
                AddLock(locks, x.Storage.VStringSlice.Allocator);
            foreach (var o in outputs)
                AddLock(locks, o.Storage.VStringSlice.Allocator);
            byte[] defaultText = NDUtf8.Encode(ruleDescr.VString.default_string);
            var handles = new GCHandle[outputs.Length];
            VStringMethodAux aux = default;
            VStringAllocator.acquire_allocators(locks);
            try
            {
                aux.ArenaA = inputs.Length > 0 ? inputs[0].Storage.VStringSlice.Allocator.ArenaBase : null;
                aux.ArenaB = inputs.Length > 1 ? inputs[1].Storage.VStringSlice.Allocator.ArenaBase : null;
                aux.ArenaC = inputs.Length > 2 ? inputs[2].Storage.VStringSlice.Allocator.ArenaBase : null;
                aux.Rule = NDVStringCells.RuleOf(ruleDescr);
                // The bodies reach each output allocator through a handle: the auxdata is unmanaged memory.
                for (int i = 0; i < outputs.Length; i++)
                    handles[i] = GCHandle.Alloc(outputs[i].Storage.VStringSlice.Allocator);
                aux.Out0 = outputs.Length > 0 ? GCHandle.ToIntPtr(handles[0]) : IntPtr.Zero;
                aux.Out1 = outputs.Length > 1 ? GCHandle.ToIntPtr(handles[1]) : IntPtr.Zero;
                aux.Out2 = outputs.Length > 2 ? GCHandle.ToIntPtr(handles[2]) : IntPtr.Zero;
                // The default string stays pinned for the whole pass (its bytes are read as an NA's text).
                fixed (byte* def = defaultText)
                {
                    aux.DefaultText = defaultText.Length == 0 ? null : def;
                    aux.DefaultSize = defaultText.Length;
                    run((IntPtr)(&aux));
                }
            }
            finally
            {
                VStringAllocator.release_allocators(locks);
                foreach (var h in handles)
                    if (h.IsAllocated)
                        h.Free();
                // The pool takes a buffer back with the size it was taken with.
                if (aux.ScratchBuffer != null)
                    SizeBucketedBufferPool.Return((IntPtr)aux.ScratchBuffer, aux.ScratchCapacity);
            }
            if (aux.Status != 0)
                throw Raise((VStringMethodError)aux.Status, ufunc);
        }

        /// <summary>Adds an allocator to the lock list once (two operands may share one).</summary>
        /// <param name="locks">The list.</param>
        /// <param name="allocator">The allocator.</param>
        private static void AddLock(List<VStringAllocator> locks, VStringAllocator allocator)
        {
            foreach (var l in locks)
                if (ReferenceEquals(l, allocator))
                    return;
            locks.Add(allocator);
        }

        /// <summary>NumPy's exception for a recorded loop error.</summary>
        /// <param name="error">The error.</param>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static Exception Raise(VStringMethodError error, string ufunc) => error switch
        {
            VStringMethodError.LengthOfNull => new ValueError("The length of a null string is undefined"),
            VStringMethodError.FindNull => new ValueError($"'{ufunc}' not supported for null values that are not strings."),
            VStringMethodError.TailNull => new ValueError($"'{ufunc}' not supported for null values that are not nan-like or strings."),
            VStringMethodError.StripNull => new ValueError("Can only strip null values that are strings or NaN-like values"),
            VStringMethodError.StripDelimiterNull => new ValueError($"Cannot use a null string that is not a string as the {ufunc} delimiter"),
            VStringMethodError.ReplacePatternNull => new ValueError("Null values are not supported as search patterns or replacement strings for replace"),
            VStringMethodError.ReplaceSearchNull => new ValueError("Only string or NaN-like null strings can be used as search strings for replace"),
            VStringMethodError.ExpandTabsNull => new ValueError("Null values are not supported arguments for expandtabs"),
            VStringMethodError.ExpandTabsOverflow => new OverflowException("new string is too long"),
            VStringMethodError.PadNull => new ValueError($"Cannot {ufunc} null that is not a nan-like value"),
            VStringMethodError.PadOverflow => new OverflowException($"Overflow encountered in {ufunc}"),
            VStringMethodError.ZFillNull => new TypeError("Cannot zfill null string that is not a nan-like value"),
            VStringMethodError.PartitionNull => new ValueError($"Null values are not supported in {ufunc}"),
            VStringMethodError.EmptySeparator => new ValueError("empty separator"),
            VStringMethodError.SliceNull => new TypeError("Cannot slice null string"),
            VStringMethodError.Load => new OutOfMemoryException($"Failed to load string in {ufunc}"),
            VStringMethodError.NotFound => new ValueError("substring not found"),
            _ => new OutOfMemoryException($"Failed to allocate string in {ufunc}"),
        };

        // =============================================================================================================
        // Cell access and result writing
        // =============================================================================================================

        /// <summary>Records the loop's first error (later ones are dropped: NumPy's <c>npy_gil_error</c> keeps the first).</summary>
        /// <param name="aux">The call's state.</param>
        /// <param name="error">The error.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Fail(VStringMethodAux* aux, VStringMethodError error)
        {
            if (aux->Status == 0)
                aux->Status = (int)error;
        }

        /// <summary>
        ///     Loads a cell as text (<c>NpyString_load</c>); a corrupt arena cell records NumPy's MemoryError and reads as an
        ///     NA so the loop moves on.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="arena">Its allocator's arena base.</param>
        /// <param name="buf">Receives the text.</param>
        /// <param name="size">Receives the text's byte size.</param>
        /// <param name="aux">The call's state.</param>
        /// <returns>True for an NA.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Load(byte* cell, byte* arena, out byte* buf, out long size, VStringMethodAux* aux)
        {
            int r = NDVStringCells.Load(cell, arena, out buf, out size);
            if (r < 0)
            {
                Fail(aux, VStringMethodError.Load);
                return true;
            }
            return r != 0;
        }

        /// <summary>The descriptor's default string (what an NA reads as under the string / no-NA rules).</summary>
        /// <param name="aux">The call's state.</param>
        /// <param name="buf">Receives the text.</param>
        /// <param name="size">Receives its byte size.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Default(VStringMethodAux* aux, out byte* buf, out long size)
        {
            buf = aux->DefaultText;
            size = aux->DefaultSize;
        }

        /// <summary>
        ///     A scratch buffer of at least <paramref name="bytes"/> bytes from the bucketed pool (grown and kept for the rest
        ///     of the call; the driver returns it with its capacity).
        /// </summary>
        /// <param name="aux">The call's state.</param>
        /// <param name="bytes">The bytes needed.</param>
        /// <returns>The buffer, or null when the size cannot be allocated (the caller records a MemoryError).</returns>
        /// <remarks>Its contents do not survive a growth: the one user (a stepped slice's code-point index) rebuilds it per
        ///     element.</remarks>
        internal static byte* Scratch(VStringMethodAux* aux, long bytes)
        {
            if (bytes <= aux->ScratchCapacity)
                return aux->ScratchBuffer;
            long doubled = aux->ScratchCapacity > long.MaxValue / 2 ? long.MaxValue : aux->ScratchCapacity * 2;
            long cap = Math.Max(bytes, Math.Max(256, doubled));
            byte* fresh;
            try
            {
                fresh = (byte*)SizeBucketedBufferPool.Take(cap);
            }
            catch (OutOfMemoryException)
            {
                // The old buffer stays in place; the driver still returns it.
                return null;
            }
            if (aux->ScratchBuffer != null)
                SizeBucketedBufferPool.Return((IntPtr)aux->ScratchBuffer, aux->ScratchCapacity);
            aux->ScratchBuffer = fresh;
            aux->ScratchCapacity = cap;
            return fresh;
        }

        /// <summary>
        ///     Prepares a FRESH output cell for a text of <paramref name="size"/> bytes and returns where to write it (NumPy's
        ///     <c>load_new_string</c>: <c>NpyString_newemptysize</c>, then the buffer): the cell itself for up to 15 bytes,
        ///     else an arena allocation (the bump path of <see cref="VStringAllocator.AppendFresh"/> while the arena has room)
        ///     or a heap one.
        /// </summary>
        /// <param name="cell">The output cell (all zero: a fresh array's).</param>
        /// <param name="size">The text size.</param>
        /// <param name="allocator">The output allocator's <see cref="GCHandle"/>.</param>
        /// <param name="aux">The call's state (a failed allocation records a MemoryError).</param>
        /// <returns>The first text byte (the caller writes all <paramref name="size"/> bytes before the next allocation),
        ///     or null after a recorded failure.</returns>
        /// <remarks>The empty string leaves the all-zero cell as it is, the bytes NumPy's <c>pack_empty</c> leaves in a
        ///     fresh cell.</remarks>
        private static byte* NewString(byte* cell, long size, IntPtr allocator, VStringMethodAux* aux)
        {
            if ((ulong)size <= ShortMax)
            {
                if (size > 0)
                    cell[15] = (byte)(ShortFlags | size);
                return cell;
            }
            // Past NPY_MAX_STRING_SIZE (a negative size included) no allocation is attempted: NumPy's newemptysize fails.
            if ((ulong)size > VStringAllocator.NPY_MAX_STRING_SIZE)
            {
                Fail(aux, VStringMethodError.Memory);
                return null;
            }
            var alloc = (VStringAllocator)GCHandle.FromIntPtr(allocator).Target;
            byte* p = alloc.AppendFresh(cell, size);
            if (p != null)
                return p;
            if (alloc.newemptysize(size, cell) < 0)
            {
                Fail(aux, VStringMethodError.Memory);
                return null;
            }
            return alloc.VStringBuffer(cell);
        }

        /// <summary>Copies a text into a fresh output cell (<see cref="NewString"/>, then the bytes).</summary>
        /// <param name="cell">The output cell (all zero).</param>
        /// <param name="text">The text (not inside the output allocator's arena).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="allocator">The output allocator's <see cref="GCHandle"/>.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteCell(byte* cell, byte* text, long size, IntPtr allocator, VStringMethodAux* aux)
        {
            byte* p = NewString(cell, size, allocator, aux);
            if (p != null && size > 0)
                Buffer.MemoryCopy(text, p, size, size);
        }

        /// <summary>Writes an NA into a fresh output cell (<c>NpyString_pack_null</c> of an all-zero cell).</summary>
        /// <param name="cell">The output cell.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteNull(byte* cell) => cell[15] = VStringAllocator.NPY_STRING_MISSING;

        // =============================================================================================================
        // UTF-8 primitives (valid UTF-8, as every StringDType text is)
        // =============================================================================================================

        /// <summary>NumPy's <c>num_bytes_for_utf8_character</c>: the byte length the lead byte announces (0 for a
        ///     continuation byte).</summary>
        /// <param name="b">The byte.</param>
        /// <returns>1–4, or 0.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CharBytes(byte b) => b < 0x80 ? 1 : b < 0xC0 ? 0 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : b < 0xF8 ? 4 : 0;

        /// <summary>NumPy's <c>utf8_char_to_ucs4_code</c>: the code point at a lead byte.</summary>
        /// <param name="c">The lead byte's address.</param>
        /// <returns>The code point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Decode(byte* c)
        {
            uint b0 = c[0];
            if (b0 <= 0x7F)
                return b0;
            if (b0 <= 0xDF)
                return ((b0 << 6) + c[1]) - ((0xC0u << 6) + 0x80);
            if (b0 <= 0xEF)
                return ((b0 << 12) + ((uint)c[1] << 6) + c[2]) - ((0xE0u << 12) + (0x80u << 6) + 0x80);
            return ((b0 << 18) + ((uint)c[1] << 12) + ((uint)c[2] << 6) + c[3]) - ((0xF0u << 18) + (0x80u << 12) + (0x80u << 6) + 0x80);
        }

        /// <summary>NumPy's <c>ucs4_code_to_utf8_char</c>: writes a code point's UTF-8 bytes.</summary>
        /// <param name="code">The code point (valid).</param>
        /// <param name="c">The destination (room for 4 bytes).</param>
        /// <returns>The byte count.</returns>
        private static int Encode(uint code, byte* c)
        {
            if (code <= 0x7F)
            {
                c[0] = (byte)code;
                return 1;
            }
            if (code <= 0x07FF)
            {
                c[0] = (byte)(0xC0 | (code >> 6));
                c[1] = (byte)(0x80 | (code & 0x3F));
                return 2;
            }
            if (code <= 0xFFFF)
            {
                c[0] = (byte)(0xE0 | (code >> 12));
                c[1] = (byte)(0x80 | ((code >> 6) & 0x3F));
                c[2] = (byte)(0x80 | (code & 0x3F));
                return 3;
            }
            c[0] = (byte)(0xF0 | (code >> 18));
            c[1] = (byte)(0x80 | ((code >> 12) & 0x3F));
            c[2] = (byte)(0x80 | ((code >> 6) & 0x3F));
            c[3] = (byte)(0x80 | (code & 0x3F));
            return 4;
        }

        /// <summary>NumPy's <c>find_previous_utf8_character(c, 1)</c>: the lead byte before <paramref name="c"/>.</summary>
        /// <param name="c">A position after a character.</param>
        /// <returns>That character's lead byte.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte* Previous(byte* c)
        {
            do
                c--;
            while ((*c & 0xC0) == 0x80);
            return c;
        }

        /// <summary>
        ///     The number of lead bytes (code points) in <c>[s, s + n)</c>: every byte that is not a continuation byte
        ///     (0x80–0xBF), counted 32 bytes at a time.
        /// </summary>
        /// <param name="s">The first byte.</param>
        /// <param name="n">The byte count.</param>
        /// <returns>The code-point count.</returns>
        private static long CountLeads(byte* s, long n)
        {
            long cont = 0, i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var limit = Vector256.Create((sbyte)-64);
                for (; i + 32 <= n; i += 32)
                    // A continuation byte is 0x80..0xBF, i.e. -128..-65 as sbyte: below -64.
                    cont += BitOperations.PopCount(Vector256.LessThan(Vector256.Load((sbyte*)(s + i)), limit).ExtractMostSignificantBits());
            }
            for (; i < n; i++)
                if ((s[i] & 0xC0) == 0x80)
                    cont++;
            return n - cont;
        }

        /// <summary>NumPy's <c>num_codepoints</c> of a UTF-8 buffer: the code points without the trailing NULs.</summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <returns>The code-point count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long CodePoints(byte* s, long size)
        {
            while (size > 0 && s[size - 1] == 0)
                size--;
            return CountLeads(s, size);
        }

        /// <summary>
        ///     The address <paramref name="chars"/> code points into a buffer (NumPy's <c>Buffer + n</c> /
        ///     <c>find_start_end_locs</c>), never past the buffer's end.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="chars">The code points to step over.</param>
        /// <returns>The address.</returns>
        private static byte* Advance(byte* s, long size, long chars)
        {
            byte* end = s + size;
            byte* p = s;
            for (long k = 0; k < chars && p < end; k++)
            {
                int nb = CharBytes(*p);
                p += nb == 0 ? 1 : nb;
            }
            return p > end ? end : p;
        }

        // =============================================================================================================
        // Byte searches over any length (a span's length is an int; a StringDType text may be longer)
        // =============================================================================================================

        /// <summary>The first occurrence of <c>n[0, nn)</c> in <c>h[0, hn)</c>, or -1 (fastsearch's FAST_SEARCH).</summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hn">Its byte count.</param>
        /// <param name="n">The needle.</param>
        /// <param name="nn">Its byte count (at least 1).</param>
        /// <returns>The byte offset, or -1.</returns>
        /// <remarks>Past 2^31 - 1 bytes the haystack is searched in windows of <see cref="SearchWindow"/> starts that
        ///     overlap by the needle's length, so a match across a window boundary is found in the window of its start.</remarks>
        private static long IndexOf(byte* h, long hn, byte* n, long nn)
        {
            if (nn > hn)
                return -1;
            if (hn <= int.MaxValue)
                return new ReadOnlySpan<byte>(h, (int)hn).IndexOf(new ReadOnlySpan<byte>(n, (int)nn));
            if (nn > SearchWindow)
            {
                for (long s = 0; s <= hn - nn; s++)
                    if (h[s] == n[0] && EqualBytes(h + s, n, nn))
                        return s;
                return -1;
            }
            for (long lo = 0; lo <= hn - nn; lo += SearchWindow)
            {
                long len = Math.Min(SearchWindow + nn - 1, hn - lo);
                int r = new ReadOnlySpan<byte>(h + lo, (int)len).IndexOf(new ReadOnlySpan<byte>(n, (int)nn));
                if (r >= 0)
                    return lo + r;
            }
            return -1;
        }

        /// <summary>The last occurrence of <c>n[0, nn)</c> in <c>h[0, hn)</c>, or -1 (fastsearch's FAST_RSEARCH).</summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hn">Its byte count.</param>
        /// <param name="n">The needle.</param>
        /// <param name="nn">Its byte count (at least 1).</param>
        /// <returns>The byte offset, or -1.</returns>
        private static long LastIndexOf(byte* h, long hn, byte* n, long nn)
        {
            if (nn > hn)
                return -1;
            if (hn <= int.MaxValue)
                return new ReadOnlySpan<byte>(h, (int)hn).LastIndexOf(new ReadOnlySpan<byte>(n, (int)nn));
            if (nn > SearchWindow)
            {
                for (long s = hn - nn; s >= 0; s--)
                    if (h[s] == n[0] && EqualBytes(h + s, n, nn))
                        return s;
                return -1;
            }
            // Windows of starts [lo, hi], from the top: each covers its starts' whole matches.
            for (long hi = hn - nn; hi >= 0;)
            {
                long lo = Math.Max(0, hi - SearchWindow + 1);
                int r = new ReadOnlySpan<byte>(h + lo, (int)(hi - lo + nn)).LastIndexOf(new ReadOnlySpan<byte>(n, (int)nn));
                if (r >= 0)
                    return lo + r;
                hi = lo - 1;
            }
            return -1;
        }

        /// <summary>The first occurrence of a byte in <c>h[0, hn)</c>, or -1 (NumPy's <c>find_char</c>).</summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hn">Its byte count.</param>
        /// <param name="b">The byte.</param>
        /// <returns>The offset, or -1.</returns>
        private static long IndexOfByte(byte* h, long hn, byte b)
        {
            for (long lo = 0; lo < hn; lo += int.MaxValue)
            {
                int r = new ReadOnlySpan<byte>(h + lo, (int)Math.Min(int.MaxValue, hn - lo)).IndexOf(b);
                if (r >= 0)
                    return lo + r;
            }
            return -1;
        }

        /// <summary>The last occurrence of a byte in <c>h[0, hn)</c>, or -1 (NumPy's <c>rfind_char</c>).</summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hn">Its byte count.</param>
        /// <param name="b">The byte.</param>
        /// <returns>The offset, or -1.</returns>
        private static long LastIndexOfByte(byte* h, long hn, byte b)
        {
            for (long hi = hn; hi > 0;)
            {
                long lo = Math.Max(0, hi - int.MaxValue);
                int r = new ReadOnlySpan<byte>(h + lo, (int)(hi - lo)).LastIndexOf(b);
                if (r >= 0)
                    return lo + r;
                hi = lo;
            }
            return -1;
        }

        /// <summary>The occurrences of a byte in <c>h[0, hn)</c> (fastsearch's FAST_COUNT of a one-byte needle).</summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hn">Its byte count.</param>
        /// <param name="b">The byte.</param>
        /// <returns>The count.</returns>
        private static long CountByte(byte* h, long hn, byte b)
        {
            long count = 0;
            for (long lo = 0; lo < hn; lo += int.MaxValue)
                count += new ReadOnlySpan<byte>(h + lo, (int)Math.Min(int.MaxValue, hn - lo)).Count(b);
            return count;
        }

        /// <summary>Whether two byte ranges of one length are equal (any length).</summary>
        /// <param name="a">The first range.</param>
        /// <param name="b">The second range.</param>
        /// <param name="n">The byte count.</param>
        /// <returns>True when every byte matches.</returns>
        private static bool EqualBytes(byte* a, byte* b, long n)
        {
            for (long lo = 0; lo < n; lo += int.MaxValue)
            {
                int len = (int)Math.Min(int.MaxValue, n - lo);
                if (!new ReadOnlySpan<byte>(a + lo, len).SequenceEqual(new ReadOnlySpan<byte>(b + lo, len)))
                    return false;
            }
            return true;
        }

        // =============================================================================================================
        // str_len / find / rfind / count / startswith / endswith
        // =============================================================================================================

        /// <summary>
        ///     One element of <c>string_strlen_strided_loop</c>: the code points; an NA reads as the default string under a
        ///     string NA, and records NumPy's error otherwise (the loop goes on, as NumPy's <c>goto next_step</c> does).
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="aux">The call's state.</param>
        /// <returns>The length (0 for a refused NA).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long VLength(byte* cell, VStringMethodAux* aux)
        {
            // A short cell: the text is the cell itself, its size the low nibble (one vector: ShortCodePoints).
            if (IsShort(cell))
                return ShortCodePoints(cell, ShortSize(cell));
            if (Load(cell, aux->ArenaA, out byte* buf, out long size, aux))
            {
                if (aux->Rule != VStringNARule.String)
                {
                    Fail(aux, VStringMethodError.LengthOfNull);
                    return 0;
                }
                Default(aux, out buf, out size);
            }
            return CodePoints(buf, size);
        }

        /// <summary>
        ///     One element of <c>string_findlike_strided_loop</c> (find / rfind / count / index / rindex): the NA rule of the
        ///     first operand's descriptor (a string NA reads as its string, any other NA records NumPy's error), then
        ///     NumPy's UTF-8 search.
        /// </summary>
        /// <param name="a">The searched cell.</param>
        /// <param name="b">The needle cell.</param>
        /// <param name="start">The start index (code points).</param>
        /// <param name="end">The end index.</param>
        /// <param name="mode">0 find, 1 rfind, 2 count.</param>
        /// <param name="index">1 for index / rindex (a miss records <c>substring not found</c>).</param>
        /// <param name="aux">The call's state.</param>
        /// <returns>The index (or count), -1 for a miss.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long VFindLike(byte* a, byte* b, long start, long end, int mode, int index, VStringMethodAux* aux)
        {
            long r;
            if (ShortSearchSupported && IsShort(a) && IsShort(b))
            {
                // Two short cells (never an NA): the vector form, FindShort.
                r = FindShort(a, b, start, end, mode);
                if (index != 0 && r == -1)
                    Fail(aux, VStringMethodError.NotFound);
                return r;
            }
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(b, aux->ArenaB, out byte* s2, out long n2, aux);
            if (na1 || na2)
            {
                if (aux->Rule is VStringNARule.NaN or VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.FindNull);
                    return -1;
                }
                if (na1)
                    Default(aux, out s1, out n1);
                if (na2)
                    Default(aux, out s2, out n2);
            }
            r = FindCore(s1, n1, s2, n2, start, end, mode);
            if (index != 0 && r == -1)
                Fail(aux, VStringMethodError.NotFound);
            return r;
        }

        /// <summary>
        ///     NumPy's <c>string_find</c> / <c>string_rfind</c> / <c>string_count</c> over UTF-8 buffers (see the file header
        ///     for the trailing-NUL rules).
        /// </summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hs">Its byte size.</param>
        /// <param name="n">The needle.</param>
        /// <param name="ns">Its byte size (trailing NULs included).</param>
        /// <param name="start">The start (code points).</param>
        /// <param name="end">The end.</param>
        /// <param name="mode">0 find, 1 rfind, 2 count.</param>
        /// <returns>The code-point index, -1, or the count.</returns>
        internal static long FindCore(byte* h, long hs, byte* n, long ns, long start, long end, int mode)
        {
            long len1 = CodePoints(h, hs);
            long len2 = CodePoints(n, ns);
            NDTextElements.AdjustOffsets(ref start, ref end, len1);
            if (mode == 2)
            {
                if (end < start || end - start < len2)
                    return 0;
                if (len2 == 0)
                    return end - start + 1;
            }
            else
            {
                if (end - start < len2)
                    return -1;
                if (len2 == 0)
                    return mode == 0 ? start : end;
            }
            // find_start_end_locs: the byte range of code points [start, end) (the trailing NULs left out).
            byte* startLoc = Advance(h, hs, start);
            long hayLen = Advance(startLoc, h + hs - startLoc, end - start) - startLoc;
            if (mode == 2)
            {
                // fastsearch FAST_COUNT over the needle's full bytes: non-overlapping, greedy from the left.
                if (ns == 1)
                    return CountByte(startLoc, hayLen, n[0]);
                long count = 0, pos = 0;
                long i;
                while ((i = IndexOf(startLoc + pos, hayLen - pos, n, ns)) >= 0)
                {
                    count++;
                    pos += i + ns;
                }
                return count;
            }
            long p;
            if (len2 == 1 && n[0] < 0x80)
                // A one-code-point ASCII needle: find_char of that byte (its trailing NULs ignored).
                p = mode == 0 ? IndexOfByte(startLoc, hayLen, n[0]) : LastIndexOfByte(startLoc, hayLen, n[0]);
            else
                p = mode == 0 ? IndexOf(startLoc, hayLen, n, ns) : LastIndexOf(startLoc, hayLen, n, ns);
            if (p > 0)
                // utf8_character_index: the code points before the match.
                p = CountLeads(startLoc, p);
            return p >= 0 ? p + start : -1;
        }

        /// <summary>
        ///     One element of <c>string_startswith_endswith_strided_loop</c>: a NaN-like NA is False, another non-string NA
        ///     records NumPy's error, a string NA reads as its string; then NumPy's UTF-8 <c>tailmatch</c>.
        /// </summary>
        /// <param name="a">The tested cell.</param>
        /// <param name="b">The prefix / suffix cell.</param>
        /// <param name="start">The start (code points).</param>
        /// <param name="end">The end.</param>
        /// <param name="back">1 for endswith.</param>
        /// <param name="aux">The call's state.</param>
        /// <returns>1 for a match, 0 otherwise.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte VTailMatch(byte* a, byte* b, long start, long end, int back, VStringMethodAux* aux)
        {
            // Two short cells (never an NA): the vector form, TailMatchShort.
            if (ShortSearchSupported && IsShort(a) && IsShort(b))
                return TailMatchShort(a, b, start, end, back != 0);
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(b, aux->ArenaB, out byte* s2, out long n2, aux);
            if (na1 || na2)
            {
                if (aux->Rule == VStringNARule.NaN)
                    return 0;
                if (aux->Rule == VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.TailNull);
                    return 0;
                }
                if (na1)
                    Default(aux, out s1, out n1);
                if (na2)
                    Default(aux, out s2, out n2);
            }
            return TailMatchCore(s1, n1, s2, n2, start, end, back != 0);
        }

        /// <summary>
        ///     NumPy's UTF-8 <c>tailmatch</c>: the needle's full bytes compared at the start / end position; haystack bytes
        ///     past the text read as zero (see the file header).
        /// </summary>
        /// <param name="h">The haystack.</param>
        /// <param name="hs">Its byte size.</param>
        /// <param name="n">The needle.</param>
        /// <param name="ns">Its byte size.</param>
        /// <param name="start">The start (code points).</param>
        /// <param name="end">The end.</param>
        /// <param name="back">True for endswith.</param>
        /// <returns>1 for a match, 0 otherwise.</returns>
        internal static byte TailMatchCore(byte* h, long hs, byte* n, long ns, long start, long end, bool back)
        {
            long len1 = CodePoints(h, hs);
            long len2 = CodePoints(n, ns);
            NDTextElements.AdjustOffsets(ref start, ref end, len1);
            end -= len2;
            if (end < start)
                return 0;
            if (len2 == 0)
                return 1;
            byte* p = Advance(h, hs, back ? end : start);
            long avail = h + hs - p;
            long common = Math.Min(avail, ns);
            if (!EqualBytes(p, n, common))
                return 0;
            // Needle bytes past the haystack's text compare with zero.
            for (long k = common; k < ns; k++)
                if (n[k] != 0)
                    return 0;
            return 1;
        }

        // =============================================================================================================
        // strip / lstrip / rstrip
        // =============================================================================================================

        /// <summary>
        ///     One element of <c>string_lrstrip_whitespace_strided_loop</c>: an NA reads as the default string (string / no
        ///     NA), is packed as an NA (NaN-like) or records NumPy's error; then NumPy's UTF-8 strip.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="type">The strip type.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VStripWhitespace(byte* a, byte* o, int type, VStringMethodAux* aux)
        {
            if (Load(a, aux->ArenaA, out byte* s, out long size, aux))
            {
                if (aux->Rule == VStringNARule.NaN)
                {
                    WriteNull(o);
                    return;
                }
                if (aux->Rule == VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.StripNull);
                    return;
                }
                Default(aux, out s, out size);
            }
            StripWhitespaceCore(s, size, type, out long first, out long bytes);
            WriteCell(o, s + first, bytes, aux->Out0, aux);
        }

        /// <summary>
        ///     NumPy's UTF-8 <c>string_lrstrip_whitespace</c>: the byte offset and byte count of the kept text (see the file
        ///     header for the trailing scan from the buffer's last byte).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="type">The strip type.</param>
        /// <param name="first">Receives the kept text's byte offset.</param>
        /// <param name="bytes">Receives its byte count.</param>
        internal static void StripWhitespaceCore(byte* s, long size, int type, out long first, out long bytes)
        {
            long len = CodePoints(s, size);
            if (len == 0)
            {
                first = 0;
                bytes = 0;
                return;
            }
            long newStart = 0, numBytes = size;
            byte* t = s;
            if (type != NDTextElements.RightStrip)
            {
                while (newStart < len)
                {
                    if (!NDCharArray.IsPyWhitespace(Decode(t)))
                        break;
                    int nb = CharBytes(*t);
                    numBytes -= nb;
                    newStart++;
                    t += nb;
                }
            }
            long newStop = len;
            t = Previous(s + size);
            if (type != NDTextElements.LeftStrip)
            {
                while (newStop > newStart)
                {
                    // NumPy strips a NUL here too (*traverse_buf != 0 && !isspace keeps only a nonzero non-space).
                    uint c = Decode(t);
                    if (c != 0 && !NDCharArray.IsPyWhitespace(c))
                        break;
                    numBytes -= CharBytes(*t);
                    newStop--;
                    // NumPy never steps to character -1 (its start cannot be found in UTF-8).
                    if (newStop > 0)
                        t = Previous(t);
                }
            }
            first = Advance(s, size, newStart) - s;
            bytes = numBytes;
        }

        /// <summary>
        ///     One element of <c>string_lrstrip_chars_strided_loop</c>: the NA rules (a NaN-like NA delimiter records NumPy's
        ///     error and strips nothing; a NaN-like NA text is packed as an NA), then NumPy's UTF-8 chars strip.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="c">The strip-set cell.</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="type">The strip type.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VStripChars(byte* a, byte* c, byte* o, int type, VStringMethodAux* aux)
        {
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(c, aux->ArenaB, out byte* s2, out long n2, aux);
            if (na1 || na2)
            {
                if (aux->Rule is VStringNARule.String or VStringNARule.None)
                {
                    if (na1)
                        Default(aux, out s1, out n1);
                    if (na2)
                        Default(aux, out s2, out n2);
                }
                else if (aux->Rule == VStringNARule.NaN)
                {
                    if (na2)
                        Fail(aux, VStringMethodError.StripDelimiterNull);
                    if (na1)
                    {
                        WriteNull(o);
                        return;
                    }
                    // NumPy goes on with the null delimiter loaded as an empty string (and raises after the loop).
                    s2 = null;
                    n2 = 0;
                }
                else
                {
                    Fail(aux, VStringMethodError.StripNull);
                    return;
                }
            }
            StripCharsCore(s1, n1, s2, n2, type, out long first, out long bytes);
            WriteCell(o, s1 + first, bytes, aux->Out0, aux);
        }

        /// <summary>
        ///     NumPy's UTF-8 <c>string_lrstrip_chars</c>: the byte offset and byte count of the kept text. A one-byte character
        ///     is looked up in the first <c>len(chars)</c> BYTES of the set (NumPy's <c>find_char</c> over a code-point count),
        ///     a longer one by a byte search of the whole set.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="chars">The set.</param>
        /// <param name="csize">Its byte size.</param>
        /// <param name="type">The strip type.</param>
        /// <param name="first">Receives the kept text's byte offset.</param>
        /// <param name="bytes">Receives its byte count.</param>
        internal static void StripCharsCore(byte* s, long size, byte* chars, long csize, int type, out long first, out long bytes)
        {
            long len1 = CodePoints(s, size);
            if (len1 == 0)
            {
                first = 0;
                bytes = 0;
                return;
            }
            long len2 = chars == null ? 0 : CodePoints(chars, csize);
            if (len2 == 0)
            {
                first = 0;
                bytes = size;
                return;
            }
            long newStart = 0, numBytes = size;
            byte* t = s;
            if (type != NDTextElements.RightStrip)
            {
                while (newStart < len1)
                {
                    int nb = CharBytes(*t);
                    bool found = nb == 1 ? IndexOfByte(chars, len2, *t) >= 0 : IndexOf(chars, csize, t, nb) >= 0;
                    if (!found)
                        break;
                    numBytes -= nb;
                    newStart++;
                    t += nb;
                }
            }
            long newStop = len1;
            t = Previous(s + size);
            if (type != NDTextElements.LeftStrip)
            {
                while (newStop > newStart)
                {
                    int nb = CharBytes(*t);
                    bool found = nb == 1 ? IndexOfByte(chars, len2, *t) >= 0 : LastIndexOf(chars, csize, t, nb) >= 0;
                    if (!found)
                        break;
                    numBytes -= nb;
                    newStop--;
                    if (newStop > 0)
                        t = Previous(t);
                }
            }
            first = Advance(s, size, newStart) - s;
            bytes = numBytes;
        }

        // =============================================================================================================
        // _slice
        // =============================================================================================================

        /// <summary>
        ///     One element of <c>slice_strided_loop</c>: any NA records <c>Cannot slice null string</c>; then the code-point
        ///     slice (step 1 runs to the buffer's end when the stop passes the code points; another step writes nothing for
        ///     an empty selection, leaving the fresh <c>''</c> cell).
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="start">The slice start.</param>
        /// <param name="stop">The slice stop.</param>
        /// <param name="step">The slice step (nonzero: the wrapper refused 0).</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="aux">The call's state.</param>
        /// <remarks>NumPy indexes code points with an <c>int</c>, which a huge step overflows; the index here is a
        ///     <see cref="long"/> (the intended selection).</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VSlice(byte* a, long start, long stop, long step, byte* o, VStringMethodAux* aux)
        {
            if (Load(a, aux->ArenaA, out byte* s, out long size, aux))
            {
                Fail(aux, VStringMethodError.SliceNull);
                return;
            }
            long trimmed = size;
            while (trimmed > 0 && s[trimmed - 1] == 0)
                trimmed--;
            long n = CountLeads(s, trimmed);
            long length = NDTextElements.AdjustSliceIndices(n, ref start, ref stop, step);
            if (step == 1)
            {
                // A bound at or past the code points is the buffer's END (its trailing NULs included).
                long b0 = start < n ? Advance(s, trimmed, start) - s : size;
                long b1 = stop < n ? Advance(s, trimmed, stop) - s : size;
                WriteCell(o, s + b0, Math.Max(b1 - b0, 0), aux->Out0, aux);
                return;
            }
            if (length <= 0)
                return;
            if (n == trimmed)
            {
                // ASCII: the code-point offsets are the byte offsets.
                byte* p = NewString(o, length, aux->Out0, aux);
                if (p == null)
                    return;
                long idx = start;
                for (long k = 0; k < length; k++, idx += step)
                    p[k] = s[idx];
                return;
            }
            // NumPy's codepoint_offsets index (in the scratch buffer), then the selected code points' bytes.
            long* offs = (long*)Scratch(aux, n * sizeof(long));
            if (offs == null)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            long at = 0;
            for (long k = 0; k < n; k++)
            {
                offs[k] = at;
                int nb = CharBytes(s[at]);
                at += nb == 0 ? 1 : nb;
            }
            long total = 0;
            long j = start;
            for (long k = 0; k < length; k++, j += step)
                total += CharBytes(s[offs[j]]);
            byte* dst = NewString(o, total, aux->Out0, aux);
            if (dst == null)
                return;
            j = start;
            for (long k = 0; k < length; k++, j += step)
            {
                byte* src = s + offs[j];
                int nb = CharBytes(*src);
                Buffer.MemoryCopy(src, dst, nb, nb);
                dst += nb;
            }
        }

        // =============================================================================================================
        // center / ljust / rjust / zfill
        // =============================================================================================================

        /// <summary>
        ///     One element of <c>center_ljust_rjust_strided_loop</c>: the NA rules (a NaN-like NA packs an NA, a non-string NA
        ///     records NumPy's error), the size <c>fill_bytes × (width − len) + size</c> (OverflowError when the product
        ///     overflows, MemoryError past the largest string), then NumPy's UTF-8 <c>string_pad</c> with the caller's width
        ///     written straight into the output.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="width">The requested width (code points).</param>
        /// <param name="fill">The fill cell (one code point).</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="pos">The justification.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VPad(byte* a, long width, byte* fill, byte* o, int pos, VStringMethodAux* aux)
        {
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(fill, aux->ArenaB, out byte* s2, out long n2, aux);
            if (na1 || na2)
            {
                if (aux->Rule == VStringNARule.NaN)
                {
                    WriteNull(o);
                    return;
                }
                if (aux->Rule == VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.PadNull);
                    return;
                }
                if (na1)
                    Default(aux, out s1, out n1);
                if (na2)
                    Default(aux, out s2, out n2);
            }
            long ncp = CodePoints(s1, n1);
            long w = width < ncp ? ncp : width;
            long padCount = w - ncp;
            // NumPy reads the fill's first byte even for an empty fill (a NUL: one byte).
            int fb = n2 > 0 ? CharBytes(s2[0]) : 1;
            // npy_mul_sizes_with_overflow(fill_bytes, width - len), then + size (which wraps: a MemoryError below).
            if (padCount != 0 && fb > long.MaxValue / padCount)
            {
                Fail(aux, VStringMethodError.PadOverflow);
                return;
            }
            long newsize = unchecked(fb * padCount + n1);
            byte* buf = NewString(o, newsize, aux->Out0, aux);
            if (buf == null)
                return;
            long final = width > 0 ? width : 0;
            if (ncp >= final)
            {
                Buffer.MemoryCopy(s1, buf, n1, n1);
                return;
            }
            uint fillCode = n2 > 0 ? Decode(s2) : 0;
            byte* code = stackalloc byte[4];
            int cb = Encode(fillCode, code);
            long pad = final - ncp;
            long left = pos == NDTextElements.Center ? pad / 2 + (pad & final & 1) : pos == NDTextElements.Left ? 0 : pad;
            long right = pad - left;
            byte* p = buf;
            p = FillCode(p, code, cb, left);
            Buffer.MemoryCopy(s1, p, n1, n1);
            // out += len_codepoints: step over the copied text's code points (its trailing NULs stay behind it).
            p = Advance(p, n1, ncp);
            p = FillCode(p, code, cb, right);
            // The bytes NumPy leaves unwritten (a trailing-NUL tail the right padding did not reach): zero.
            if (p < buf + newsize)
                NativeMemory.Clear(p, (nuint)(buf + newsize - p));
        }

        /// <summary>Writes a code point's bytes <paramref name="count"/> times (NumPy's <c>buffer_memset</c> of a fill).</summary>
        /// <param name="p">The destination.</param>
        /// <param name="code">The code point's UTF-8 bytes.</param>
        /// <param name="cb">Their count.</param>
        /// <param name="count">The repetitions.</param>
        /// <returns>The position after the last repetition.</returns>
        private static byte* FillCode(byte* p, byte* code, int cb, long count)
        {
            if (cb == 1)
            {
                NativeMemory.Fill(p, (nuint)count, code[0]);
                return p + count;
            }
            for (long k = 0; k < count; k++, p += cb)
                Buffer.MemoryCopy(code, p, cb, cb);
            return p;
        }

        /// <summary>
        ///     One element of <c>zfill_strided_loop</c>: the NA rules (a NaN-like NA packs an NA, a non-string NA records
        ///     NumPy's TypeError), then <c>string_zfill</c> with <c>width = max(width, len)</c> written into the output.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="width">The requested width.</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VZFill(byte* a, long width, byte* o, VStringMethodAux* aux)
        {
            if (Load(a, aux->ArenaA, out byte* s, out long size, aux))
            {
                if (aux->Rule == VStringNARule.NaN)
                {
                    WriteNull(o);
                    return;
                }
                if (aux->Rule == VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.ZFillNull);
                    return;
                }
                Default(aux, out s, out size);
            }
            long ncp = CodePoints(s, size);
            long w = width < ncp ? ncp : width;
            long left = w - ncp;
            // (width - in_codepoints) + is.size: the leading '0's are one byte each.
            long outsize = unchecked(left + size);
            byte* buf = NewString(o, outsize, aux->Out0, aux);
            if (buf == null)
                return;
            NativeMemory.Fill(buf, (nuint)left, (byte)'0');
            Buffer.MemoryCopy(s, buf + left, size, size);
            // The sign test at offset width - len; a position past the buffer (an empty text) has no sign.
            if (left < outsize)
            {
                byte c = buf[left];
                if (c == '+' || c == '-')
                {
                    buf[left] = (byte)'0';
                    buf[0] = c;
                }
            }
        }

        // =============================================================================================================
        // replace / expandtabs / partition
        // =============================================================================================================

        /// <summary>
        ///     One element of <c>string_replace_strided_loop</c>: the NA rules, <c>count = min(count == -1 ? MAX : count,
        ///     found)</c>, then NumPy's UTF-8 <c>string_replace</c> (whole buffers, trailing NULs included).
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="old">The searched cell.</param>
        /// <param name="new">The replacement cell.</param>
        /// <param name="count">The requested count.</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VReplace(byte* a, byte* old, byte* @new, long count, byte* o, VStringMethodAux* aux)
        {
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(old, aux->ArenaB, out byte* s2, out long n2, aux);
            bool na3 = Load(@new, aux->ArenaC, out byte* s3, out long n3, aux);
            if (na1 || na2 || na3)
            {
                if (aux->Rule is VStringNARule.NaN or VStringNARule.Null)
                {
                    if (na2 || na3)
                    {
                        Fail(aux, VStringMethodError.ReplacePatternNull);
                        return;
                    }
                    if (aux->Rule == VStringNARule.NaN)
                    {
                        WriteNull(o);
                        return;
                    }
                    // NumPy records the error and goes on with the null text loaded as an empty string.
                    Fail(aux, VStringMethodError.ReplaceSearchNull);
                    s1 = null;
                    n1 = 0;
                }
                else
                {
                    if (na1)
                        Default(aux, out s1, out n1);
                    if (na2)
                        Default(aux, out s2, out n2);
                    if (na3)
                        Default(aux, out s3, out n3);
                }
            }
            long inCount = count == -1 ? long.MaxValue : count;
            long found = FindCore(s1, n1, s2, n2, 0, long.MaxValue, 2);
            ReplaceCore(s1, n1, s2, n2, s3, n3, Math.Min(inCount, found), o, aux);
        }

        /// <summary>
        ///     NumPy's UTF-8 <c>string_replace</c>, written straight into the output: the exact size first (each of the
        ///     <paramref name="count"/> replacements swaps <c>len(old)</c> bytes for <c>len(new)</c>; an old text of no code
        ///     points interleaves), then the copies. The old text is searched by its full bytes.
        /// </summary>
        /// <param name="s1">The text.</param>
        /// <param name="n1">Its byte size.</param>
        /// <param name="s2">The old text.</param>
        /// <param name="n2">Its byte size.</param>
        /// <param name="s3">The new text.</param>
        /// <param name="n3">Its byte size.</param>
        /// <param name="count">The replacement limit (<c>min(count, found)</c>: never more than the matches).</param>
        /// <param name="o">The output cell.</param>
        /// <param name="aux">The call's state.</param>
        private static void ReplaceCore(byte* s1, long n1, byte* s2, long n2, byte* s3, long n3, long count, byte* o, VStringMethodAux* aux)
        {
            long len1 = CodePoints(s1, n1);
            long len2 = CodePoints(s2, n2);
            long len3 = CodePoints(s3, n3);
            if (count <= 0 || len1 < len2 || (len2 <= 0 && len3 <= 0) || (len2 == len3 && Utf8Equal(s2, n2, s3, n3)))
            {
                WriteCell(o, s1, n1, aux->Out0, aux);
                return;
            }
            // The found matches over the code points are the first matches over the whole buffer too, so the loop
            // replaces exactly count of them; the interleave writes count replacements around count - 1 characters.
            long size = len2 > 0 ? n1 + count * (n3 - n2) : n1 + count * n3;
            byte* buf = NewString(o, size, aux->Out0, aux);
            if (buf == null)
                return;
            long w = 0, pos = 0;
            if (len2 > 0)
            {
                for (long time = 0; time < count; time++)
                {
                    long p = n2 == 1 ? IndexOfByte(s1 + pos, n1 - pos, s2[0]) : IndexOf(s1 + pos, n1 - pos, s2, n2);
                    if (p < 0)
                        break;
                    Buffer.MemoryCopy(s1 + pos, buf + w, p, p);
                    w += p;
                    pos += p;
                    Buffer.MemoryCopy(s3, buf + w, n3, n3);
                    w += n3;
                    pos += n2;
                }
            }
            else
            {
                while (count > 0)
                {
                    Buffer.MemoryCopy(s3, buf + w, n3, n3);
                    w += n3;
                    if (--count <= 0)
                        break;
                    int nb = CharBytes(s1[pos]);
                    if (nb == 0)
                        nb = 1;
                    Buffer.MemoryCopy(s1 + pos, buf + w, nb, nb);
                    w += nb;
                    pos += nb;
                }
            }
            long tail = n1 - pos;
            Buffer.MemoryCopy(s1 + pos, buf + w, tail, tail);
        }

        /// <summary>
        ///     NumPy's UTF-8 <c>Buffer::strcmp</c> equality: the code points of both texts equal, and the longer one's tail all
        ///     NUL (so trailing NULs never decide).
        /// </summary>
        /// <param name="a">The first text.</param>
        /// <param name="na">Its byte size.</param>
        /// <param name="b">The second text.</param>
        /// <param name="nb">Its byte size.</param>
        /// <returns>True when NumPy's strcmp answers 0.</returns>
        private static bool Utf8Equal(byte* a, long na, byte* b, long nb)
        {
            while (na > 0 && a[na - 1] == 0)
                na--;
            while (nb > 0 && b[nb - 1] == 0)
                nb--;
            return na == nb && EqualBytes(a, b, na);
        }

        /// <summary>
        ///     One element of <c>string_expandtabs_strided_loop</c>: an NA reads as the default string under a string NA and
        ///     records NumPy's error otherwise; then the UTF-8 length (OverflowError past 2^31 - 1) and the expansion
        ///     written into the output.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="tabsize">The tab size.</param>
        /// <param name="o">The output cell (fresh).</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VExpandTabs(byte* a, long tabsize, byte* o, VStringMethodAux* aux)
        {
            if (Load(a, aux->ArenaA, out byte* s, out long size, aux))
            {
                if (aux->Rule != VStringNARule.String)
                {
                    Fail(aux, VStringMethodError.ExpandTabsNull);
                    return;
                }
                Default(aux, out s, out size);
            }
            long len = CodePoints(s, size);
            long newLen = 0, linePos = 0;
            byte* t = s;
            // NumPy's string_expandtabs_length: npy_intp arithmetic (the overflow test sees a wrapped value too).
            unchecked
            {
                for (long i = 0; i < len; i++)
                {
                    uint ch = Decode(t);
                    int nb = CharBytes(*t);
                    if (ch == '\t')
                    {
                        if (tabsize > 0)
                        {
                            long incr = tabsize - linePos % tabsize;
                            linePos += incr;
                            newLen += incr;
                        }
                    }
                    else
                    {
                        linePos++;
                        newLen += nb;
                        if (ch == '\n' || ch == '\r')
                            linePos = 0;
                    }
                    if (newLen > NDTextElements.ExpandTabsMaxLength || newLen < 0)
                    {
                        Fail(aux, VStringMethodError.ExpandTabsOverflow);
                        return;
                    }
                    t += nb;
                }
            }
            byte* buf = NewString(o, newLen, aux->Out0, aux);
            if (buf == null)
                return;
            long w = 0;
            linePos = 0;
            t = s;
            unchecked
            {
                for (long i = 0; i < len; i++)
                {
                    uint ch = Decode(t);
                    int nb = CharBytes(*t);
                    if (ch == '\t')
                    {
                        if (tabsize > 0)
                        {
                            long incr = tabsize - linePos % tabsize;
                            linePos += incr;
                            NativeMemory.Fill(buf + w, (nuint)incr, (byte)' ');
                            w += incr;
                        }
                    }
                    else
                    {
                        linePos++;
                        Buffer.MemoryCopy(t, buf + w, nb, nb);
                        w += nb;
                        if (ch == '\n' || ch == '\r')
                            linePos = 0;
                    }
                    t += nb;
                }
            }
        }

        /// <summary>
        ///     One element of <c>string_partition_strided_loop</c>: an NA reads as the default string under a string NA and
        ///     records NumPy's error otherwise; an empty separator records <c>empty separator</c>; then the raw-byte search of
        ///     the separator's full bytes over the text's full bytes and the three copies.
        /// </summary>
        /// <param name="a">The input cell.</param>
        /// <param name="sep">The separator cell.</param>
        /// <param name="o1">The first part's cell (fresh).</param>
        /// <param name="o2">The separator part's cell.</param>
        /// <param name="o3">The third part's cell.</param>
        /// <param name="back">1 for <c>_rpartition</c>.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VPartition(byte* a, byte* sep, byte* o1, byte* o2, byte* o3, int back, VStringMethodAux* aux)
        {
            bool na1 = Load(a, aux->ArenaA, out byte* s1, out long n1, aux);
            bool na2 = Load(sep, aux->ArenaB, out byte* s2, out long n2, aux);
            if (na1 || na2)
            {
                if (aux->Rule != VStringNARule.String)
                {
                    Fail(aux, VStringMethodError.PartitionNull);
                    return;
                }
                if (na1)
                    Default(aux, out s1, out n1);
                if (na2)
                    Default(aux, out s2, out n2);
            }
            if (n2 == 0)
            {
                Fail(aux, VStringMethodError.EmptySeparator);
                return;
            }
            long idx = back != 0 ? LastIndexOf(s1, n1, s2, n2) : IndexOf(s1, n1, s2, n2);
            if (idx < 0)
            {
                // Not found: the whole text is the first part (_partition) or the third (_rpartition).
                WriteCell(o1, s1, back != 0 ? 0 : n1, aux->Out0, aux);
                WriteCell(o3, s1, back != 0 ? n1 : 0, aux->Out2, aux);
                return;
            }
            WriteCell(o1, s1, idx, aux->Out0, aux);
            WriteCell(o2, s2, n2, aux->Out1, aux);
            WriteCell(o3, s1 + idx + n2, n1 - idx - n2, aux->Out2, aux);
        }
    }
}
