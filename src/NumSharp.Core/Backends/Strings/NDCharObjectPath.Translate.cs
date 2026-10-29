using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;

// =============================================================================
// NDCharObjectPath.Translate.cs — np.strings.translate's driver (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py, multiarraymodule.c _vec_string_with_args)
//   a_arr = np.asarray(a)
//   if issubclass(a_arr.dtype.type, np.str_):  return _vec_string(a_arr, a_arr.dtype, 'translate', (table,))
//   else:                                       return _vec_string(a_arr, a_arr.dtype, 'translate',
//                                                                  [table] + _clean_args(deletechars))
//   So deletechars is IGNORED for U (never converted), dropped when None for S and StringDType, and passed to the method
//   otherwise — where str.translate (StringDType) refuses it ("str.translate() takes exactly one argument (2 given)") and
//   bytes.translate (S) takes it as the delete set. _vec_string checks that a is text ("string operation on non-string
//   array"), converts and broadcasts the arguments (the multi-iterator's "shape mismatch: objects cannot be broadcast to a
//   single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (2,)."), allocates a C-ordered result of
//   a's dtype over the broadcast shape and fills it in C order, ending at the first element that raises (the per-element
//   order: the operands' scalars, the method descriptor's check, its argument count, the method body, then setitem).
//
// HOW IT IS COMPUTED HERE
//   * Converted arguments: VecArg (the object path's np.asarray model; a C# dict is the 0-d object array holding it).
//   * A table (and delete set) of ONE element whose broadcast is a's shape — the common call — is compiled once
//     (StrTranslateTable / BytesTranslateTable, NDTranslate.cs) and the whole array is translated by the TextMethodOp kernel
//     (NDStringMethods.TranslateKernel for S / U, NDVStringMethods.TranslateKernel for StringDType), any layout of a. A
//     bytes.translate argument error is the call's (it raises before the first element is read); an element that raises
//     sets the kernel's error word, and then the C-order walk below runs the calls again to raise NumPy's first error.
//   * Any other broadcast (an array of tables, an array of delete sets, a StringDType call with deletechars): the C-order
//     walk over (a, table[, delete]) with the object path's element readers, one Python call per element as NumPy makes it.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary><c>_vec_string</c>'s TypeError when a <c>StringDType</c> element's result cannot be stored (a lone
        ///     surrogate: setitem's UTF-8 encoding fails, and the failure is reported under this text).</summary>
        private const string ResultTypeText = "result array type does not match underlying function";

        /// <summary>
        ///     NumPy's <c>np.strings.translate(a, table, deletechars)</c> (see the file header): a fresh C-ordered array of
        ///     <paramref name="a"/>'s dtype over the broadcast shape, each element <c>bytes.translate</c> / <c>str.translate</c>
        ///     of <paramref name="a"/>'s element.
        /// </summary>
        /// <param name="a">The strings (<c>np.asarray</c>'d by the caller).</param>
        /// <param name="table">The table (a dict, a str / bytes, None, a number, or an array_like of them).</param>
        /// <param name="deletechars">The delete set (null = Python's None: not passed); ignored for <c>U</c>.</param>
        /// <returns>The result (an <c>S</c> / <c>U</c> result keeps the element width, translated text cut to it; a
        ///     <c>StringDType</c> result has <paramref name="a"/>'s NA object and coercion).</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is not text; a table or delete set the method refuses; a
        ///     <c>StringDType</c> NA that is not a str; <paramref name="deletechars"/> with a <c>StringDType</c> operand; a
        ///     <c>StringDType</c> result holding a lone surrogate.</exception>
        /// <exception cref="ValueError">A mapping outside <c>range(0x110000)</c>; a bytes table not 256 bytes long.</exception>
        /// <exception cref="SystemError">A <c>U</c> unit past U+10FFFF the call would put in a str.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast (NumPy's multi-iterator text).</exception>
        /// <exception cref="NotSupportedException">An argument with no NumSharp form (an object array of lists).</exception>
        internal static NDArray Translate(NDArray a, object table, object deletechars)
        {
            var loop = NDStringMethods.ResolveOne(a);
            // _vec_string picks bytes.translate for S, str.translate for U and StringDType, and refuses anything else.
            if (loop == NDStringMethods.TextLoop.None)
                throw new TypeError(NonStringArrayText);
            bool vstring = loop == NDStringMethods.TextLoop.VString;
            bool unicode = loop == NDStringMethods.TextLoop.Str;
            // U: (table,) — deletechars is not even converted. S / T: [table] + _clean_args(deletechars).
            bool passDelete = !unicode && deletechars is not null;
            VecArg tab = null, del = null;
            try
            {
                tab = VecArg.Convert(table, "table");
                if (passDelete)
                    del = VecArg.Convert(deletechars, "deletechars");
                if (tab.Lists is not null || (del is not null && del.Lists is not null))
                    throw new NotSupportedException("An object array of lists as a translate argument has no NumSharp form.");
                var outer = del is null
                    ? Shape.BroadcastShape(a.Shape, tab.BroadcastShape)
                    : Shape.BroadcastShape(a.Shape, tab.BroadcastShape, del.BroadcastShape);
                // PyArray_SimpleNewFromDescr: C order whatever a's layout. S / U elements are written whole by every route;
                // a StringDType result starts with all-zero cells, as every fresh T array does.
                var result = vstring
                    ? new NDArray(NDVStringMethods.SameNADescr(a.dtype), outer.Clean(), true)
                    : NDArray.ForOverwrite(a.dtype, outer.Clean());
                try
                {
                    if (outer.size == 0)
                        return result;   // no element, no call: not even bytes.translate's argument errors
                    StrTranslateTable compiled = null;
                    bool single = tab.Array.size == 1 && (del is null || del.Array.size == 1) && SameDims(outer, a.Shape);
                    if (single && !(vstring && passDelete) && TranslateSingle(a, tab, del, loop, result, out compiled))
                        return result;
                    TranslateWalk(a, tab, del, loop, result, compiled);
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
                tab?.Dispose();
                del?.Dispose();
            }
        }

        /// <summary>Whether a broadcast shape has the operand's dimensions (the one-table route keeps a's shape).</summary>
        /// <param name="outer">The broadcast shape.</param>
        /// <param name="a">The operand's shape.</param>
        /// <returns>True when equal.</returns>
        private static bool SameDims(Shape outer, Shape a)
        {
            var o = outer.dimensions ?? Array.Empty<long>();
            var d = a.dimensions ?? Array.Empty<long>();
            if (o.Length != d.Length)
                return false;
            for (int i = 0; i < o.Length; i++)
                if (o[i] != d[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     The one-table route: the single table (and delete set) read and compiled once, the whole operand translated by
        ///     the kernel.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="tab">The table argument (one element).</param>
        /// <param name="del">The delete argument (one element), or null.</param>
        /// <param name="loop">The operand's loop.</param>
        /// <param name="result">The fresh result.</param>
        /// <param name="compiled">Receives the compiled <c>str.translate</c> table when an element raised (the walk reuses
        ///     it), else null.</param>
        /// <returns>True when the result is complete; false when an element raised and the walk must find NumPy's first
        ///     error.</returns>
        /// <exception cref="TypeError">A bytes.translate argument the call refuses (no buffer).</exception>
        /// <exception cref="ValueError">A bytes.translate table not 256 bytes long.</exception>
        /// <exception cref="SystemError">A <c>U</c> table of one unit past U+10FFFF (its scalar).</exception>
        private static bool TranslateSingle(NDArray a, VecArg tab, VecArg del, NDStringMethods.TextLoop loop, NDArray result,
                                            out StrTranslateTable compiled)
        {
            compiled = null;
            var readers = BuildReaders(tab, del);
            try
            {
                ReadSingle(tab, readers[0], out var tabElem);
                if (loop == NDStringMethods.TextLoop.Bytes)
                {
                    Elem delElem = default;
                    if (del is not null)
                        ReadSingle(del, readers[1], out delElem);
                    // bytes.translate checks its arguments before it reads the input: every call raises the same error, so
                    // the first element's call does.
                    var bytesTable = CompileBytesTable(in tabElem, del is not null, in delElem);
                    if (bytesTable.Error is not null)
                        throw bytesTable.Error.ToException();
                    var state = default(TranslateState);
                    bytesTable.Prepare(&state);
                    NDStringMethods.TranslateKernel(a, result, &state);
                    GC.KeepAlive(bytesTable);
                    return true;
                }
                // The dense range follows the call's work: a U element's units, about 16 characters per StringDType cell.
                long work = a.size * (loop == NDStringMethods.TextLoop.Str ? a.dtype.itemsize / 4 : 16);
                var table = CompileStrTable(in tabElem, StrTranslateTable.DenseLimitFor(work));
                var st = default(TranslateState);
                table.Prepare(&st);
                var handle = GCHandle.Alloc(table);
                try
                {
                    st.Table = GCHandle.ToIntPtr(handle);
                    if (loop == NDStringMethods.TextLoop.Str)
                        NDStringMethods.TranslateKernel(a, result, &st);
                    else
                        NDVStringMethods.TranslateKernel(a, result, &st);
                }
                finally
                {
                    handle.Free();
                }
                if (st.Status == 0)
                    return true;
                compiled = table;
                return false;
            }
            finally
            {
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        /// <summary>
        ///     The scalar of a one-element argument (<c>PyArray_ToScalar</c>), read under its allocator's lock (a
        ///     <c>StringDType</c> argument's text is decoded into the reader's scratch).
        /// </summary>
        /// <param name="arg">The argument (one element).</param>
        /// <param name="reader">Its reader.</param>
        /// <param name="e">Receives the object (valid until the reader's next read).</param>
        /// <exception cref="SystemError">A <c>U</c> element of one unit past U+10FFFF.</exception>
        private static void ReadSingle(VecArg arg, Operand reader, out Elem e)
        {
            var x = arg.Array;
            byte* p = (byte*)x.Address + x.Shape.offset * (long)NDIterRef.ElementBytes(x);
            reader.Allocator?.acquire_allocator();
            try
            {
                reader.Read(p, out e);
            }
            finally
            {
                reader.Allocator?.release_allocator();
            }
        }

        // =============================================================================================================
        // The tables of one element's scalar
        // =============================================================================================================

        /// <summary>
        ///     The <c>str.translate</c> table of a table object: a dict by its keys, a str / bytes as a sequence indexed by code
        ///     point, a Python list / tuple by position, a NumPy numeric scalar keeping everything, anything else
        ///     unsubscriptable.
        /// </summary>
        /// <param name="e">The table's scalar.</param>
        /// <param name="denseLimit">The longest dense range (0 for a table of one element of an array of tables).</param>
        /// <returns>The compiled table.</returns>
        private static StrTranslateTable CompileStrTable(in Elem e, int denseLimit)
        {
            switch (e.Kind)
            {
                case ElemKind.NpStr:
                case ElemKind.Str:
                    return StrTranslateTable.FromCodePoints(new ReadOnlySpan<uint>(e.Units, checked((int)e.Length)), denseLimit);
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    return StrTranslateTable.FromBytes(new ReadOnlySpan<byte>(e.Units, checked((int)e.Length)), denseLimit);
                case ElemKind.Number:
                    // np.int64(5)[97] raises IndexError ("invalid index to scalar variable."): a LookupError, so kept.
                    return StrTranslateTable.KeepAll(denseLimit);
                default:
                    return e.Obj switch
                    {
                        IDictionary dict => StrTranslateTable.FromDict(dict, denseLimit),
                        IList list => StrTranslateTable.FromList(list, denseLimit),
                        ITuple tuple => StrTranslateTable.FromList(TupleItems(tuple), denseLimit),
                        _ => StrTranslateTable.Unsubscriptable(TypeName(e)),
                    };
            }
        }

        /// <summary>A Python tuple's items as a list (indexed by position like the tuple).</summary>
        /// <param name="tuple">The tuple.</param>
        /// <returns>The items.</returns>
        private static IList TupleItems(ITuple tuple)
        {
            var items = new object[tuple.Length];
            for (int i = 0; i < items.Length; i++)
                items[i] = tuple[i];
            return items;
        }

        /// <summary>
        ///     <c>bytes.translate(table[, delete])</c>'s argument checks and trans_table: a None table maps every byte to itself,
        ///     any other must expose a buffer of exactly 256 bytes (checked first), then the delete set must expose a buffer.
        /// </summary>
        /// <param name="tab">The table's scalar.</param>
        /// <param name="hasDelete">Whether NumPy passes a delete set.</param>
        /// <param name="del">The delete set's scalar (when passed).</param>
        /// <returns>The table, or one carrying the call's error.</returns>
        private static BytesTranslateTable CompileBytesTable(in Elem tab, bool hasDelete, in Elem del)
        {
            bool hasTable = !IsNone(tab);
            byte* tb = null;
            long tl = 0;
            if (hasTable)
            {
                if (!TryBuffer(tab, out tb, out tl))
                    return BytesTranslateTable.Failed(TranslateError.Type($"a bytes-like object is required, not '{TypeName(tab)}'"));
                if (tl != 256)
                    return BytesTranslateTable.Failed(TranslateError.TableLength);
            }
            byte* db = null;
            long dl = 0;
            if (hasDelete && !TryBuffer(del, out db, out dl))
                return BytesTranslateTable.Failed(TranslateError.Type($"a bytes-like object is required, not '{TypeName(del)}'"));
            return BytesTranslateTable.Create(new ReadOnlySpan<byte>(tb, hasTable ? 256 : 0), hasTable,
                                              new ReadOnlySpan<byte>(db, checked((int)dl)));
        }

        // =============================================================================================================
        // The C-order walk (one call per element, NumPy's order)
        // =============================================================================================================

        /// <summary>The state of one translate walk, shared by its NDIter chunks.</summary>
        [NDBorrowed] // the readers belong to TranslateWalk, which releases them; the state only uses them during the walk
        private sealed class TranslateWalkState
        {
            /// <summary>The readers: a, the table, then the delete set when passed.</summary>
            internal Operand[] Readers;

            /// <summary>The operand's loop (S: bytes.translate; U / T: str.translate).</summary>
            internal NDStringMethods.TextLoop Loop;

            /// <summary>Whether NumPy passes a delete set.</summary>
            internal bool HasDelete;

            /// <summary>Whether the table argument has one element (its table is compiled once).</summary>
            internal bool SingleTable;

            /// <summary>Whether the table and delete arguments both have one element (S: one trans_table).</summary>
            internal bool SingleBytes;

            /// <summary>The <c>str.translate</c> table of a one-element table argument, once compiled.</summary>
            internal StrTranslateTable Table;

            /// <summary>The <c>bytes.translate</c> table of one-element arguments, once compiled.</summary>
            internal BytesTranslateTable BytesTable;

            /// <summary>The tables of the dict / list objects of an object array of tables, by object.</summary>
            internal Dictionary<object, StrTranslateTable> ObjectTables;

            /// <summary>The result's first element.</summary>
            internal byte* Output;

            /// <summary>The result's element size in bytes (16 for a <c>StringDType</c> cell).</summary>
            internal int OutputSize;

            /// <summary>The <c>StringDType</c> result's allocator (locked by the walk), or null.</summary>
            internal VStringAllocator OutputAllocator;

            /// <summary>The <c>StringDType</c> result's string NA text (UTF-8), or null when its NA is not a string.</summary>
            internal byte[] NAText;

            /// <summary>The next element's C-order position.</summary>
            internal long Position;

            /// <summary>Scratch: a <c>StringDType</c> element's translated code points.</summary>
            internal uint[] Points = new uint[64];

            /// <summary>Scratch: a <c>StringDType</c> element's UTF-8 encoding.</summary>
            internal byte[] Utf8 = new byte[64];
        }

        /// <summary>
        ///     The C-order walk of the translate calls (see the file header): every element computed in NumPy's order, ending
        ///     at the first element that raises.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="tab">The table argument.</param>
        /// <param name="del">The delete argument, or null.</param>
        /// <param name="loop">The operand's loop.</param>
        /// <param name="result">The result (C-ordered over the broadcast shape; its elements are overwritten).</param>
        /// <param name="compiled">The one-table route's compiled table (reused), or null.</param>
        /// <exception cref="Exception">The first element's error (see <see cref="Translate"/>).</exception>
        private static void TranslateWalk(NDArray a, VecArg tab, VecArg del, NDStringMethods.TextLoop loop, NDArray result,
                                          StrTranslateTable compiled)
        {
            using var self = VecArg.Convert(a, "a");
            var readers = BuildReaders(self, tab, del);
            try
            {
                var ops = new List<NDArray> { a, tab.Array };
                if (del is not null)
                    ops.Add(del.Array);
                bool vstring = loop == NDStringMethods.TextLoop.VString;
                var w = new TranslateWalkState
                {
                    Readers = readers,
                    Loop = loop,
                    HasDelete = del is not null,
                    SingleTable = tab.Array.size == 1,
                    SingleBytes = tab.Array.size == 1 && (del is null || del.Array.size == 1),
                    Table = compiled,
                    Output = (byte*)result.Storage.Address,
                    OutputSize = vstring ? VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING : (int)result.dtype.itemsize,
                };
                if (vstring)
                {
                    w.OutputAllocator = result.Storage.VStringSlice.Allocator;
                    w.NAText = NDVStringCells.RuleOf(result.dtype) == VStringNARule.String ? NDUtf8.Encode(result.dtype.VString.default_string) : null;
                    w.OutputAllocator.acquire_allocator();
                }
                try
                {
                    Walk(ops, readers, (dataptrs, strides, n, aux) => TranslateRun(dataptrs, strides, n, w));
                }
                finally
                {
                    w.OutputAllocator?.release_allocator();
                }
            }
            finally
            {
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        /// <summary>One NDIter chunk of the walk: each element's scalars, then its call.</summary>
        /// <param name="dataptrs">The operand pointers (a, table, delete).</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="w">The walk's state.</param>
        /// <exception cref="Exception">The first element's error.</exception>
        private static void TranslateRun(void** dataptrs, long* strides, long n, TranslateWalkState w)
        {
            byte* pa = (byte*)dataptrs[0];
            long sa = strides[0];
            byte* pt = (byte*)dataptrs[1];
            long stt = strides[1];
            byte* pd = w.HasDelete ? (byte*)dataptrs[2] : null;
            long sd = w.HasDelete ? strides[2] : 0;
            for (long i = 0; i < n; i++, pa += sa, pt += stt, pd += sd)
            {
                // PyArray_ToScalar of every operand, in argument order (a U one-unit text past U+10FFFF raises here).
                w.Readers[0].Read(pa, out var text);
                w.Readers[1].Read(pt, out var tab);
                Elem del = default;
                if (w.HasDelete)
                    w.Readers[2].Read(pd, out del);
                if (w.Loop == NDStringMethods.TextLoop.Bytes)
                    TranslateBytesElement(in text, in tab, in del, w);
                else
                    TranslateStrElement(in text, in tab, w);
                w.Position++;
            }
        }

        /// <summary>
        ///     One <c>str.translate</c> call: the descriptor's check (a <c>StringDType</c> NA that is not a str), the argument
        ///     count (a delete set is one argument too many), the translation, then the store (a <c>U</c> element cut to its
        ///     width; a <c>StringDType</c> element under setitem's rules).
        /// </summary>
        /// <param name="text">a's element.</param>
        /// <param name="tab">The table's element.</param>
        /// <param name="w">The walk's state.</param>
        /// <exception cref="TypeError">The descriptor's, the argument count's, a table's or the store's error.</exception>
        /// <exception cref="ValueError">A mapping outside <c>range(0x110000)</c>.</exception>
        /// <exception cref="SystemError">A unit past U+10FFFF put in a str.</exception>
        private static void TranslateStrElement(in Elem text, in Elem tab, TranslateWalkState w)
        {
            if (!IsStr(text))
                throw new TypeError($"descriptor 'translate' for 'str' objects doesn't apply to a '{TypeName(text)}' object");
            if (w.HasDelete)
                throw new TypeError("str.translate() takes exactly one argument (2 given)");
            var table = TableOf(in tab, w);
            var points = new ReadOnlySpan<uint>(text.Units, checked((int)text.Length));
            if (w.Loop == NDStringMethods.TextLoop.Str)
            {
                // The np.str_ result stored into a U element: cut to the width, the rest zero.
                uint* dst = (uint*)(w.Output + w.Position * w.OutputSize);
                int units = w.OutputSize / 4;
                var error = table.TranslateInto(points, new Span<uint>(dst, units), out long total);
                if (error is not null)
                    throw error.ToException();
                for (long k = Math.Min(total, units); k < units; k++)
                    dst[k] = 0;
                return;
            }
            // StringDType: the whole result, then setitem.
            long cap = Math.Max(points.Length * table.MaxReplacement, 16);
            if (w.Points.Length < cap && cap <= int.MaxValue)
                w.Points = new uint[cap];
            var err = table.TranslateInto(points, w.Points, out long length);
            if (err is not null)
                throw err.ToException();
            if (length > w.Points.Length)
            {
                // The capacity guess fell short (it is capped by the array size): the second pass has room for all of it.
                w.Points = new uint[checked((int)length)];
                table.TranslateInto(points, w.Points, out length);
            }
            StoreVString(w, new ReadOnlySpan<uint>(w.Points, 0, (int)length));
        }

        /// <summary>
        ///     <c>stringdtype_setitem</c> of a translated str: a surrogate fails its UTF-8 encoding (reported by
        ///     <c>_vec_string</c> as <see cref="ResultTypeText"/>); under a string NA object a result equal to it is the NA.
        /// </summary>
        /// <param name="w">The walk's state (the output cell, allocator and NA text).</param>
        /// <param name="points">The result's code points.</param>
        /// <exception cref="TypeError">A surrogate code point (<c>result array type does not match underlying function</c>).</exception>
        /// <exception cref="OutOfMemoryException">The text cannot be allocated.</exception>
        private static void StoreVString(TranslateWalkState w, ReadOnlySpan<uint> points)
        {
            int size = 0;
            foreach (uint cp in points)
            {
                if (cp - 0xD800u < 0x800u)
                    throw new TypeError(ResultTypeText);
                size += cp <= 0x7F ? 1 : cp <= 0x7FF ? 2 : cp <= 0xFFFF ? 3 : 4;
            }
            if (w.Utf8.Length < size)
                w.Utf8 = new byte[Math.Max(size, w.Utf8.Length * 2)];
            int n = 0;
            foreach (uint cp in points)
                n += new System.Text.Rune(cp).EncodeToUtf8(w.Utf8.AsSpan(n));
            byte* cell = w.Output + w.Position * w.OutputSize;
            if (w.NAText is not null && w.NAText.AsSpan().SequenceEqual(w.Utf8.AsSpan(0, n)))
            {
                if (w.OutputAllocator.pack_null(cell) < 0)
                    throw new OutOfMemoryException("Failed to pack string in translate");
                return;
            }
            fixed (byte* u = w.Utf8)
                if (w.OutputAllocator.pack(cell, u, n) < 0)
                    throw new OutOfMemoryException("Failed to pack string in translate");
        }

        /// <summary>
        ///     One <c>bytes.translate</c> call: the argument checks (table, then delete set), then the element's bytes mapped
        ///     or deleted into the <c>S</c> element, the rest zero.
        /// </summary>
        /// <param name="text">a's element (an <c>np.bytes_</c>).</param>
        /// <param name="tab">The table's element.</param>
        /// <param name="del">The delete set's element (when passed).</param>
        /// <param name="w">The walk's state.</param>
        /// <exception cref="TypeError">An argument without a buffer.</exception>
        /// <exception cref="ValueError">A table not 256 bytes long.</exception>
        private static void TranslateBytesElement(in Elem text, in Elem tab, in Elem del, TranslateWalkState w)
        {
            var table = w.BytesTable;
            if (table is null)
            {
                table = CompileBytesTable(in tab, w.HasDelete, in del);
                if (w.SingleBytes)
                    w.BytesTable = table;
            }
            if (table.Error is not null)
                throw table.Error.ToException();
            byte* dst = w.Output + w.Position * w.OutputSize;
            int k = table.TranslateInto(new ReadOnlySpan<byte>(text.Units, checked((int)text.Length)), new Span<byte>(dst, w.OutputSize));
            for (; k < w.OutputSize; k++)
                dst[k] = 0;
        }

        /// <summary>
        ///     The <c>str.translate</c> table of the current element: the one-element argument's table (compiled once), an
        ///     object's table by object (an object array repeats its dicts), or a fresh one for a str / bytes element.
        /// </summary>
        /// <param name="tab">The table's element.</param>
        /// <param name="w">The walk's state.</param>
        /// <returns>The table.</returns>
        private static StrTranslateTable TableOf(in Elem tab, TranslateWalkState w)
        {
            if (w.Table is not null)
                return w.Table;
            if (w.SingleTable)
                return w.Table = CompileStrTable(in tab, 0);
            if (tab.Kind == ElemKind.Object && tab.Obj is not null && (tab.Obj is IDictionary || tab.Obj is IList || tab.Obj is ITuple))
            {
                w.ObjectTables ??= new Dictionary<object, StrTranslateTable>(ReferenceEqualityComparer.Instance);
                if (!w.ObjectTables.TryGetValue(tab.Obj, out var cached))
                    w.ObjectTables[tab.Obj] = cached = CompileStrTable(in tab, 0);
                return cached;
            }
            return CompileStrTable(in tab, 0);
        }
    }
}
