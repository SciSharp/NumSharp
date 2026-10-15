using System;
using System.Collections;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Strings;

// =============================================================================
// np.char.array.cs — numpy.char's chararray constructor and np.char.array / np.char.asarray
// =============================================================================
//
// Ports of numpy/_core/defchararray.py, statement for statement (2.4.2):
//
//   chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0, strides=None, order='C')
//       ndarray.__new__(subtype, shape, (str_ or bytes_, int(itemsize)), [buffer, offset, strides,] order); a str
//       buffer is not a buffer but a FILLER: the array is allocated and `self[...] = filler` stores it in every element.
//
//   array(obj, itemsize=None, copy=True, unicode=None, order=None), in this order:
//       1. str / bytes   unicode from the type, itemsize = len(obj), shape = len(obj) // itemsize, then
//                        chararray(..., buffer=obj): a str FILLS every element ('abcdef' at itemsize 2 is
//                        ['ab', 'ab', 'ab'], not the documented chunks); bytes are a read-only view, chunked.
//       2. list / tuple  np.asarray(obj) first.
//       3. an S / U ndarray: view(chararray); itemsize = obj.itemsize (BYTES), // 4 for U; order= runs np.asarray,
//                        which drops the subclass; then astype((cls, itemsize)) when copy, or when the itemsize in
//                        CHARACTERS differs from obj.itemsize in BYTES — always for U, so asarray never shares a U array.
//       4. any other ndarray: `issubclass(obj.dtype.type, object)` is always true, so without an itemsize the array is
//                        obj.tolist() — its numbers become their Python str() at the longest value's width, a float32 /
//                        float16 by the DOUBLE's repr ('0.10000000149011612'), a StringDType by its str values.
//       5. everything else: np.array(obj, dtype=(cls[, itemsize]), order=order) and view(chararray), so a Python number,
//                        bool or None becomes its text (None is b'None').
//
// THE ORDER QUIRK. Branch 3 with order= returns the plain ndarray np.asarray made (NumPy's own type stubs declare a
// chararray). The C# return type is NDCharArray: the result is a chararray with NumPy's data, layout and flags.
//
// THE C# ARGUMENT MAP (the np.char module's text rule, np.char.cs): a string is Python's str, a byte[] Python's bytes
// (read-only, as a bytes object is immutable), an NDArray an ndarray, any other array / IList / ITuple / IEnumerable a
// list, a number / bool / BigInteger / null a Python scalar (Half, char and decimal are NumPy scalars of their dtype).
//
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        // =============================================================================================================
        // chararray / array / asarray
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0, strides=None, order='C')</c>:
        ///     a new chararray of <paramref name="itemsize"/>-character <c>S</c> (or <c>U</c>) elements. Without a buffer the
        ///     memory is <c>np.empty</c>'s (a <c>U</c> chararray reads empty strings, an <c>S</c> one holds whatever the
        ///     allocator recycled).
        /// </summary>
        /// <param name="shape">The dimensions (an <see cref="int"/>, a <c>long[]</c> or a tuple converts).</param>
        /// <param name="itemsize">The element width in characters. NumPy's <c>(dtype, itemsize)</c> tuple accepts 0 to
        ///     <c>2^31 - 1</c> for <c>S</c> and to <c>2^29 - 1</c> for <c>U</c>.</param>
        /// <param name="unicode"><c>true</c> for <c>U</c> (str), <c>false</c> for <c>S</c> (bytes).</param>
        /// <param name="buffer">
        ///     <c>null</c> for new memory. A C# <see cref="string"/> is NumPy's str FILLER: every element is set to it
        ///     (truncated to the width; an <c>S</c> chararray encodes it ASCII), and <paramref name="offset"/> /
        ///     <paramref name="strides"/> are ignored. Any other buffer is viewed without copying, exactly as
        ///     <see cref="np.ndarray(Shape, DType, object, long, long[], char)"/> views it: a <c>byte[]</c> / <c>Memory&lt;byte&gt;</c>
        ///     writable (NumPy's <c>bytearray</c>), a <c>ReadOnlyMemory&lt;byte&gt;</c> read-only (NumPy's <c>bytes</c>), an
        ///     <see cref="NDArray"/> through its bytes.
        /// </param>
        /// <param name="offset">The BYTE offset of the first element in <paramref name="buffer"/>.</param>
        /// <param name="strides">The BYTE strides over <paramref name="buffer"/>, or <c>null</c> for <paramref name="order"/>'s.</param>
        /// <param name="order"><c>'C'</c> or <c>'F'</c> (either case; <c>'A'</c> / <c>'K'</c> are C).</param>
        /// <returns>The new chararray: it owns its memory without a buffer or with a string filler, and views the buffer
        ///     otherwise.</returns>
        /// <exception cref="ValueError"><paramref name="itemsize"/> is negative or too large (<c>invalid itemsize in generic
        ///     type tuple</c>), a negative dimension (<c>negative dimensions are not allowed</c>), or an unknown order.</exception>
        /// <exception cref="NotSupportedException"><paramref name="itemsize"/> is 0: NumPy builds a zero-width <c>S0</c> /
        ///     <c>U0</c> array, which NumSharp's fixed-width string buffers cannot hold.</exception>
        /// <exception cref="UnicodeEncodeError">A string filler with a non-ASCII character into an <c>S</c> chararray.</exception>
        /// <exception cref="TypeError">A buffer that is too small (<c>buffer is too small for requested array</c>) or not a
        ///     bytes-like object.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.char.chararray.html</remarks>
        [NDScoped]
        public NDCharArray chararray(Shape shape, long itemsize = 1, bool unicode = false, object buffer = null, long offset = 0,
                                     long[] strides = null, char order = 'C')
        {
            // ndarray.__new__ converts the (dtype, itemsize) tuple before it reads the buffer or the order.
            var descr = TextDescr(unicode, itemsize);

            NDArray created;
            if (buffer is string filler)
            {
                // "unicode objects do not have the buffer interface": a str buffer is dropped, the array allocated as for no
                // buffer (offset and strides unused), and the str stored into every element.
                created = np.ndarray(shape, descr, order: order);
                created[Slice.Ellipsis] = filler;
            }
            else if (buffer is null)
            {
                created = np.ndarray(shape, descr, order: order);
            }
            else
            {
                created = np.ndarray(shape, descr, buffer, offset, strides, order);
            }

            // The new array is the chararray (NumPy allocates it as the subclass): the chararray takes over its storage, so
            // OWNDATA is the ndarray's (true without a buffer, false over one). The plain wrapper is reclaimed by the scope.
            return new NDCharArray(created.Storage);
        }

        /// <summary>
        ///     NumPy's <c>np.char.array(obj, itemsize=None, copy=True, unicode=None, order=None)</c>: a chararray from a string,
        ///     bytes, a list of them, a string array or a number array (whose numbers become their text).
        /// </summary>
        /// <param name="obj">
        ///     What to convert (see the class remarks for the C# argument map):
        ///     a <see cref="string"/> — <paramref name="itemsize"/> defaults to its length and every element is set to the
        ///     WHOLE string (<c>array("abcdef", itemsize: 2)</c> is <c>['ab', 'ab', 'ab']</c>, NumPy's behaviour, not its
        ///     documentation's chunks);
        ///     a <c>byte[]</c> — chunked into <paramref name="itemsize"/>-byte elements, a READ-ONLY view of the bytes;
        ///     an <c>S</c> / <c>U</c> <see cref="NDArray"/> — viewed, or copied (see <paramref name="copy"/>);
        ///     any other array — its elements as their text, the longest one setting the width (a float32 / float16 by the
        ///     double's repr, <c>0.10000000149011612</c>; an empty axis ends the shape as <c>tolist()</c> does: <c>(0, 3)</c>
        ///     gives <c>(0,)</c>); a list, a number, a bool or <c>null</c> (<c>None</c>) — through <c>np.asarray</c> / its text.
        /// </param>
        /// <param name="itemsize">The element width in characters, or <c>null</c> to discover it. <c>0</c> is NumPy's unsized
        ///     request (a string array keeps its width, a number array casts at the dtype's fixed width).</param>
        /// <param name="copy"><c>false</c> shares the memory of an <c>S</c> array (a <c>U</c> array is always copied: NumPy
        ///     compares the itemsize in characters with the array's in BYTES) and returns a chararray argument ITSELF.</param>
        /// <param name="unicode"><c>true</c> for <c>U</c>, <c>false</c> for <c>S</c>, <c>null</c> to follow the input (a number
        ///     array then becomes <c>S</c>). An <c>S</c> array with <c>unicode: true</c> and <c>copy: false</c> stays <c>S</c>,
        ///     as in NumPy.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (either case), or <c>null</c>.</param>
        /// <returns>The chararray. NumPy 2.4.2 returns a PLAIN ndarray when <paramref name="order"/> is given for a string
        ///     array (its <c>np.asarray</c> step drops the subclass); the C# result is a chararray with NumPy's data, layout and
        ///     flags.</returns>
        /// <exception cref="DivideByZeroException">A string or bytes <paramref name="obj"/> with an itemsize of 0 (its own
        ///     length 0 included): NumPy's <c>len(obj) // itemsize</c> raises <c>ZeroDivisionError</c>
        ///     (<c>integer division or modulo by zero</c>).</exception>
        /// <exception cref="ValueError">A negative or too large <paramref name="itemsize"/> (<c>invalid itemsize in generic
        ///     type tuple</c>), an unknown order, or a ragged list (NumPy's inhomogeneous-shape text).</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="TypeError">A <c>byte[]</c> with <c>unicode: true</c> holds fewer bytes than the <c>U</c> elements
        ///     need (<c>buffer is too small for requested array</c>).</exception>
        /// <exception cref="NotSupportedException">A string chararray of width 0 would result, or an element NumPy converts
        ///     as an object (an object array input).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.char.array.html</remarks>
        [NDScoped]
        public NDCharArray array(object obj, long? itemsize = null, bool copy = true, bool? unicode = null, char? order = null)
        {
            switch (obj)
            {
                case string text:
                    return FromStr(text, itemsize, unicode, order);
                case byte[] bytes:
                    return FromBytes(bytes, itemsize, unicode, order);
                case NDArray nd:
                    return FromNDArray(nd, itemsize, copy, unicode, order);
            }

            if (IsPythonSequence(obj))
            {
                // isinstance(obj, (list, tuple)): obj = np.asarray(obj) (the module's text rule builds it: a list of str is a U
                // array, a list of bytes an S array, numbers a number array), then the ndarray branches.
                return FromNDArray(TextOperand(obj, nameof(obj)), itemsize, copy, unicode, order);
            }

            // A Python scalar (a number, a bool, None, a Python int of any size) or a NumPy scalar (Half, char, decimal):
            // np.array(obj, dtype=(cls[, itemsize]), order=order).view(chararray) — the text NumPy's setitem stores for it.
            // (cls, 0) is the unsized class here, as in astype: np.array(12345, dtype=(bytes_, 0)) discovers b'12345'.
            var request = itemsize is null
                ? TextClass(unicode ?? false).singleton
                : TextDescr(unicode ?? false, itemsize.Value, unsizedAtZero: true);
            char layout = OrderResolver.ResolveBufferOrder(order ?? 'C');
            NDArray built = NDStringArrays.FromValues(Array.Empty<long>(), new[] { obj }, request, layout);
            return NDCharArray.ViewOf(built);
        }

        /// <summary>
        ///     NumPy's <c>np.char.asarray(obj, itemsize=None, unicode=None, order=None)</c>: <see cref="array"/> with
        ///     <c>copy: false</c> — an <c>S</c> array is shared (a chararray argument is returned ITSELF), a <c>U</c> array is
        ///     still copied (NumPy's itemsize comparison of characters with bytes).
        /// </summary>
        /// <param name="obj">What to convert (see <see cref="array"/>).</param>
        /// <param name="itemsize">The element width in characters, or <c>null</c> to discover it.</param>
        /// <param name="unicode"><c>true</c> for <c>U</c>, <c>false</c> for <c>S</c>, <c>null</c> to follow the input.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>, or <c>null</c>.</param>
        /// <returns>The chararray (see <see cref="array"/> for the order quirk).</returns>
        /// <exception cref="DivideByZeroException">A string or bytes <paramref name="obj"/> with an itemsize of 0.</exception>
        /// <exception cref="ValueError">A negative or too large <paramref name="itemsize"/>, an unknown order, or a ragged
        ///     list.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="TypeError">A <c>byte[]</c> with <c>unicode: true</c> holds fewer bytes than the elements need.</exception>
        /// <exception cref="NotSupportedException">A string chararray of width 0 would result.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.char.asarray.html</remarks>
        public NDCharArray asarray(object obj, long? itemsize = null, bool? unicode = null, char? order = null)
            => array(obj, itemsize, copy: false, unicode, order);

        // =============================================================================================================
        // The branches of np.char.array
        // =============================================================================================================

        /// <summary>
        ///     Branch 1 for a str: <c>chararray(len(obj) // itemsize, itemsize, unicode, buffer=obj, order)</c> — the str FILLS
        ///     every element.
        /// </summary>
        /// <param name="text">The str.</param>
        /// <param name="itemsize">The width, or <c>null</c> for the str's length in code points.</param>
        /// <param name="unicode">The class, or <c>null</c> for <c>U</c>.</param>
        /// <param name="order">The order, or <c>null</c> (C).</param>
        /// <returns>The filled chararray.</returns>
        /// <exception cref="DivideByZeroException">The itemsize is 0.</exception>
        /// <exception cref="ValueError">A negative or too large itemsize, or an unknown order.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII str into <c>S</c>.</exception>
        private NDCharArray FromStr(string text, long? itemsize, bool? unicode, char? order)
        {
            long length = NDStringCodec.CodePointCount(text);
            long size = itemsize ?? length;
            return chararray(PythonFloorDivide(length, size), size, unicode ?? true, text, order: order ?? 'C');
        }

        /// <summary>
        ///     Branch 1 for bytes: <c>chararray(len(obj) // itemsize, itemsize, unicode, buffer=obj, order)</c> — a READ-ONLY
        ///     view of the bytes (a Python bytes object is immutable), chunked into elements.
        /// </summary>
        /// <param name="bytes">The bytes (viewed, pinned while the chararray lives).</param>
        /// <param name="itemsize">The width, or <c>null</c> for the byte count.</param>
        /// <param name="unicode">The class, or <c>null</c> for <c>S</c>.</param>
        /// <param name="order">The order, or <c>null</c> (C).</param>
        /// <returns>The chararray view.</returns>
        /// <exception cref="DivideByZeroException">The itemsize is 0.</exception>
        /// <exception cref="ValueError">A negative or too large itemsize, or an unknown order.</exception>
        /// <exception cref="TypeError">A <c>U</c> request the bytes are too short for.</exception>
        private NDCharArray FromBytes(byte[] bytes, long? itemsize, bool? unicode, char? order)
        {
            long size = itemsize ?? bytes.LongLength;
            return chararray(PythonFloorDivide(bytes.LongLength, size), size, unicode ?? false, new ReadOnlyMemory<byte>(bytes),
                             order: order ?? 'C');
        }

        /// <summary>Branches 3 and 4: an ndarray (after a list's <c>np.asarray</c>).</summary>
        /// <param name="nd">The array.</param>
        /// <param name="itemsize">The width, or <c>null</c>.</param>
        /// <param name="copy">Whether to copy a string array that could be shared.</param>
        /// <param name="unicode">The class, or <c>null</c> to follow the input.</param>
        /// <param name="order">The order, or <c>null</c>.</param>
        /// <returns>The chararray.</returns>
        /// <exception cref="ValueError">A bad itemsize or order.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        private NDCharArray FromNDArray(NDArray nd, long? itemsize, bool copy, bool? unicode, char? order)
            => nd.Storage.IsFixedString
                ? FromStringArray(nd, itemsize, copy, unicode, order)
                : FromOtherArray(nd, itemsize, unicode, order);

        /// <summary>
        ///     Branch 3: an <c>S</c> / <c>U</c> ndarray. NumPy views it as a chararray, takes its itemsize in BYTES (divided by 4
        ///     for <c>U</c>), runs <c>np.asarray(obj, order)</c> when an order is given (a plain array) and casts with
        ///     <c>astype((cls, itemsize))</c> when <paramref name="copy"/> is set or the itemsize in characters differs from the
        ///     array's in bytes.
        /// </summary>
        /// <param name="nd">The string array (a chararray included).</param>
        /// <param name="itemsize">The width, or <c>null</c> for the array's.</param>
        /// <param name="copy">Whether to cast (copy) even when nothing has to change.</param>
        /// <param name="unicode">The class, or <c>null</c> for the array's.</param>
        /// <param name="order">The order, or <c>null</c>.</param>
        /// <returns>The chararray: <paramref name="nd"/> itself when it is a chararray that needs no cast and no order, a view
        ///     when nothing is copied, a new owning chararray after a cast.</returns>
        /// <exception cref="ValueError">An unknown order, or a negative / too large itemsize when a cast is needed.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> → <c>S</c> cast of a non-ASCII character.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> → <c>U</c> cast of a byte above 0x7F.</exception>
        private static NDCharArray FromStringArray(NDArray nd, long? itemsize, bool copy, bool? unicode, char? order)
        {
            bool isUnicode = ((StringDTypeMeta)nd.dtype.Meta).IsUnicode;
            // if not isinstance(obj, chararray): obj = obj.view(chararray) — a new view object; a chararray stays itself.
            NDArray current = nd as NDCharArray ?? NDCharArray.ViewOf(nd);
            long size = itemsize ?? (isUnicode ? current.itemsize / 4 : current.itemsize);
            bool toUnicode = unicode ?? isUnicode;

            if (order is char layout)
            {
                // asnarray(obj, order=order): np.asarray drops the subclass — a plain view when the layout already satisfies
                // the order, a plain copy otherwise.
                OrderResolver.Convert(layout);
                using var plain = current.view((DType)null);
                current = np.asarray(plain, order: layout);
                if (ReferenceEquals(current, plain))
                    current = plain.view((DType)null);
            }

            // NumPy compares the itemsize in CHARACTERS with obj.itemsize in BYTES, so a U array is always cast (copied); and
            // a class change alone (unicode against S data) never casts (`isinstance(obj, str_)` is false for an array).
            if (copy || size != current.itemsize)
                current = current.astype(TextDescr(toUnicode, size, unsizedAtZero: true), copy: true);

            return current as NDCharArray ?? NDCharArray.TakeOver(current, nd);
        }

        /// <summary>
        ///     Branch 4: an ndarray that is not fixed-width text (numbers, bools, StringDType, a chararray a <c>view(dtype)</c>
        ///     left numeric). Without an itemsize NumPy converts <c>obj.tolist()</c> (its elements as Python objects) with the
        ///     unsized class, so each number becomes its <c>str()</c> and the longest sets the width; with an itemsize it casts
        ///     the array to that width.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="itemsize">The width, or <c>null</c>.</param>
        /// <param name="unicode">The class, or <c>null</c> (<c>S</c>).</param>
        /// <param name="order">The order, or <c>null</c>.</param>
        /// <returns>A chararray VIEW of the new text array (OWNDATA false, NumPy's <c>val.view(chararray)</c>).</returns>
        /// <exception cref="ValueError">A bad itemsize or order.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII StringDType text into <c>S</c>.</exception>
        private static NDCharArray FromOtherArray(NDArray nd, long? itemsize, bool? unicode, char? order)
        {
            bool toUnicode = unicode ?? false;
            NDArray text;
            if (itemsize is null)
            {
                // obj = obj.tolist(); then np.array(list, dtype=cls, order=order): the elements as the Python objects tolist
                // makes, the shape as deep as tolist's nesting.
                var request = TextClass(toUnicode).singleton;
                char layout = OrderResolver.ResolveBufferOrder(order ?? 'K');
                text = nd.size > 0 && nd.typecode is NPTypeCode.Double or NPTypeCode.Single or NPTypeCode.Half
                    ? FloatsAsText(nd, toUnicode, layout)
                    : NDStringArrays.FromValues(ToListShape(nd), ToListValues(nd), request, layout);
            }
            else
            {
                // np.array(obj, dtype=(cls, itemsize), order=order): the cast (copy) at that width, the input's layout for no
                // order ('K'); itemsize 0 is the unsized class, cast at the dtype's fixed text width.
                var descr = TextDescr(toUnicode, itemsize.Value, unsizedAtZero: true);
                text = np.asarray(nd, descr, order ?? 'K', copy: true);
            }

            return NDCharArray.ViewOf(text);
        }

        /// <summary>
        ///     Branch 4 for a non-empty float16 / float32 / float64 array without an itemsize: the same text array as
        ///     <c>np.array(obj.tolist(), dtype=cls, order=order)</c>, built from the array instead of one object per element.
        ///     <c>tolist()</c> makes Python floats (a float16 / float32 element widened EXACTLY to its double), and the
        ///     conversion writes each one's <c>repr()</c> at the longest one's width. NumPy's (and NumSharp's) float64 → text
        ///     cast writes that same <c>repr()</c> (probed against Python's <c>repr</c> for 200,000 random doubles and the
        ///     specials: none differ), so one cast at the class's fixed width (<c>S32</c> / <c>U32</c>), the longest text, and a
        ///     narrowing copy give the result with one formatting per element where the object route formatted twice (once
        ///     to discover the width, once to store).
        /// </summary>
        /// <param name="nd">The non-empty float array (any layout).</param>
        /// <param name="unicode">The class.</param>
        /// <param name="layout"><c>'C'</c> or <c>'F'</c>: the result's layout (<c>np.array</c>'s order).</param>
        /// <returns>The text array: <paramref name="nd"/>'s shape, the longest repr's width, owning its data.</returns>
        private static NDArray FloatsAsText(NDArray nd, bool unicode, char layout)
        {
            // The widening copy and the fixed-width text are temporaries of this call: released on every path.
            using var widened = nd.typecode == NPTypeCode.Double ? null : nd.astype(np.float64, copy: true, order: layout);
            using var wide = (widened ?? nd).astype(TextClass(unicode).singleton, copy: true, order: layout);
            long width = Math.Max(LongestText(wide), 1);
            return wide.astype(TextDescr(unicode, width), copy: true, order: layout);
        }

        /// <summary>
        ///     The longest element of a fixed-width string array in characters, trailing NULs (the padding) not counted:
        ///     the width <c>np.array</c> discovers from the elements' text.
        /// </summary>
        /// <param name="text">An <c>S</c> / <c>U</c> array that owns a dense buffer (every element is read once, in memory
        ///     order).</param>
        /// <returns>The longest text length; 0 when every element is empty.</returns>
        private static unsafe long LongestText(NDArray text)
        {
            var slice = text.Storage.StringSlice;
            byte* element = slice.Address + text.Shape.offset * (long)slice.ItemSize;
            long longest = 0;
            for (long i = 0, count = text.size; i < count; i++, element += slice.ItemSize)
                longest = Math.Max(longest, NDCharArray.TrimmedUnits(element, slice.Chars, slice.IsUnicode));
            return longest;
        }

        // =============================================================================================================
        // Helpers
        // =============================================================================================================

        /// <summary>
        ///     The shape <c>np.array(a.tolist())</c> rebuilds: <paramref name="nd"/>'s dimensions up to and including the first
        ///     one of length 0 — a nested list ends there, so <c>(0, 3)</c> becomes <c>(0,)</c> and <c>(2, 0, 3)</c>
        ///     <c>(2, 0)</c>.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The dimensions of the rebuilt array.</returns>
        private static long[] ToListShape(NDArray nd)
        {
            long[] dims = nd.Shape.dimensions;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] == 0)
                    return dims.AsSpan(0, i + 1).ToArray();
            return dims;
        }

        /// <summary>
        ///     <c>a.tolist()</c>'s elements in C order as the Python objects NumPy converts: a float16 / float32 element is the
        ///     Python float of its exact double value (so its text is the double's repr), every other element the value the
        ///     storage reads (an integer, a bool, a complex, a StringDType str or NA object).
        /// </summary>
        /// <param name="nd">The array (any layout).</param>
        /// <returns>The elements, one per position.</returns>
        private static object[] ToListValues(NDArray nd)
        {
            long count = nd.size;
            var values = new object[count];
            bool widen = nd.typecode is NPTypeCode.Half or NPTypeCode.Single;
            for (long i = 0; i < count; i++)
            {
                object value = nd.GetAtIndex(i);
                // ndarray.tolist() returns Python floats for float16 / float32 (exact widenings); the C# Half would be read
                // as a NumPy float16 scalar and printed with its own shortest digits.
                values[i] = !widen ? value : value is Half h ? (double)h : (double)(float)value;
            }

            return values;
        }

        /// <summary>
        ///     Whether <paramref name="obj"/> is a Python sequence for <c>np.char.array</c>'s <c>isinstance(obj, (list,
        ///     tuple))</c>: any array, list, tuple or enumerable that is not a string, a <c>byte[]</c> or an <see cref="NDArray"/>.
        /// </summary>
        /// <param name="obj">The argument.</param>
        /// <returns>True for a sequence.</returns>
        private static bool IsPythonSequence(object obj)
            => obj is Array or ITuple or IEnumerable && obj is not string && obj is not byte[] && obj is not NDArray;

        /// <summary>The fixed-width text class: <c>StrDType</c> (<c>U</c>) or <c>BytesDType</c> (<c>S</c>).</summary>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <returns>The class.</returns>
        private static StringDTypeMeta TextClass(bool unicode) => unicode ? DTypeRegistry.Str : DTypeRegistry.Bytes;

        /// <summary>
        ///     NumPy's <c>(str_ or bytes_, int(itemsize))</c> descriptor tuple: a sized <c>U</c> / <c>S</c> descriptor, with
        ///     the tuple's own range check.
        /// </summary>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <param name="itemsize">The width in characters.</param>
        /// <param name="unsizedAtZero">True where NumPy reads 0 as the UNSIZED class (<c>astype</c> and <c>np.array</c> adapt it
        ///     to the data); false where it would build a zero-width array (the constructor).</param>
        /// <returns>The descriptor; the unsized class singleton for a 0 width when <paramref name="unsizedAtZero"/>.</returns>
        /// <exception cref="ValueError">A negative width, or one whose byte size does not fit a C <c>int</c>
        ///     (<c>invalid itemsize in generic type tuple</c>).</exception>
        /// <exception cref="NotSupportedException">A 0 width for the constructor: NumSharp cannot hold an <c>S0</c> /
        ///     <c>U0</c> array.</exception>
        private static DType TextDescr(bool unicode, long itemsize, bool unsizedAtZero = false)
        {
            var cls = TextClass(unicode);
            // _convert_from_tuple: the itemsize must be >= 0 and its byte size an int (2^31 - 1 for S, 2^29 - 1 for U).
            if (itemsize < 0 || itemsize > cls.MaxChars)
                throw new ValueError("invalid itemsize in generic type tuple");
            if (itemsize == 0)
            {
                if (unsizedAtZero)
                    return cls.singleton;
                throw new NotSupportedException(
                    $"np.char.chararray with itemsize 0: NumPy builds a zero-width {(unicode ? "'<U0'" : "'|S0'")} array here, which " +
                    "NumSharp's fixed-width string buffers cannot hold. Give a width of at least 1.");
            }

            return cls.Descr(itemsize);
        }

        /// <summary>Python's <c>a // b</c> for integers: the floor of the quotient (<c>-7 // 2</c> is <c>-4</c>).</summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The floor quotient.</returns>
        /// <exception cref="DivideByZeroException"><paramref name="b"/> is 0 (Python's <c>integer division or modulo by
        ///     zero</c>).</exception>
        private static long PythonFloorDivide(long a, long b)
        {
            if (b == 0)
                throw new DivideByZeroException("integer division or modulo by zero");
            long q = a / b;
            // C# truncates toward zero; Python floors: step down when the signs differ and the division is inexact.
            if ((a % b != 0) && ((a < 0) ^ (b < 0)))
                q--;
            return q;
        }
    }
}
