using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using NumSharp.IO;

// =============================================================================
// NDCharObjectPath.Codec.cs — np.strings.encode / np.strings.decode (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py, multiarraymodule.c _vec_string)
//   encode(a, encoding=None, errors=None) = _to_bytes_or_str_array(_vec_string(a, object_, 'encode',
//                                               _clean_args(encoding, errors)), np.bytes_(b''))
//   decode(a, encoding=None, errors=None) = the same with 'decode' and np.str_('')
//   * _clean_args passes the arguments up to the FIRST None: encode(a, None, 'ignore') is UTF-8 strict.
//   * _vec_string picks the method by a's dtype: bytes.encode does not exist ("type object 'bytes' has no attribute
//     'encode'"), str.decode neither (U and StringDType: "type object 'str' has no attribute 'decode'"), a non-text a is
//     "string operation on non-string array". Then it converts the arguments with np.asarray and broadcasts them with a
//     (NumPy's multi-iterator: "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0
//     with shape (2,) and arg 1 with shape (3,)."), and calls the method per element in C order with every operand's
//     NumPy scalar (PyArray_ToScalar: a first; a U one-unit text past U+10FFFF is CPython's SystemError there). The
//     first element that raises ends the call.
//   * The method, per element (CPython 3.12): the descriptor check (str.encode applied to a StringDType NA that is not a
//     str: "descriptor 'encode' for 'str' objects doesn't apply to a 'NoneType' object"); Argument Clinic's checks —
//     encoding, then errors: a str ("encode() argument 'encoding' must be str, not numpy.bytes_"), its UTF-8 (a lone
//     surrogate is UnicodeEncodeError "'utf-8' codec can't encode character '\ud800' in position 0: surrogates not
//     allowed"), no NUL ("embedded null character"); then the codec (NDCodec): bytes.decode of EMPTY bytes is '' before
//     any lookup, so an unknown or non-text codec is no error for an empty S element.
//   * _to_bytes_or_str_array: no element → result.astype(S1) / astype(U1) (the broadcast shape kept); otherwise
//     np.asarray(result.tolist()) cast to S / U as wide as the longest result (at least 1) — trailing NULs count for the
//     width though the element drops them (encode(['a'], 'utf-16-le') is S2 holding b'a'). encode of StringDType text is
//     S too (np.bytes_ is the template), never StringDType.
//
// HOW IT IS COMPUTED HERE
//   One C-order NDIter walk over (a, encoding?, errors?) — each operand its own dtype, no casting. The arguments are read
//   and validated once per distinct element ADDRESS (a 0-d argument, the common call, is validated at the first element
//   and never again; a varying one is compared by content), and its codec looked up once per name. Each element is coded
//   with NDCodec (CPython's codecs and error handlers) and moved into the result's row (CodecRows: the width chosen from
//   the first 64 elements, the rows re-laid out for a longer element, narrowed once at the end). Once the rows exist, the
//   common cases are written STRAIGHT into the row in one pass: a U element under UTF-8 / ASCII / Latin-1 whose code
//   points are all below 0x80 (Latin-1: 0x100) is narrowed (its bytes ARE its encoding), an S element under UTF-8 /
//   ASCII whose bytes are all ASCII (Latin-1: any) is widened, and a StringDType element under UTF-8 is copied (its cell
//   holds UTF-8 already); any other element runs the codec into the arena and is copied into its row.
//   The chunk loop (CodecRun), the element routines and the rows' copies are [MethodImpl(AggressiveOptimization)], for
//   the reason NDCharObjectPath.CodecFast.cs's header gives: a routine entered once per chunk runs once per call over a
//   contiguous array, and tiering would keep it, and every routine it calls without inlining, on tier-0 code for a
//   short-lived process's first calls.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary>CPython's AttributeError for <c>bytes.encode</c> (an <c>S</c> operand of <c>np.strings.encode</c>).</summary>
        private const string BytesEncodeText = "type object 'bytes' has no attribute 'encode'";

        /// <summary>CPython's AttributeError for <c>str.decode</c> (a <c>U</c> / <c>StringDType</c> operand of
        ///     <c>np.strings.decode</c>).</summary>
        private const string StrDecodeText = "type object 'str' has no attribute 'decode'";

        /// <summary>CPython's ValueError for a NUL in an encoding / errors argument (Argument Clinic's <c>strlen</c> check).</summary>
        private const string EmbeddedNullText = "embedded null character";

        /// <summary>
        ///     Test hook: when set on the calling thread, every element runs the codec into the arena (no direct row writes) —
        ///     the general route the fast paths must agree with byte for byte.
        /// </summary>
        [ThreadStatic] internal static bool CodecFastPathsDisabled;

        /// <summary>
        ///     NumPy's <c>np.strings.encode(a, encoding, errors)</c> (see the file header): <c>str.encode</c> per element of a
        ///     <c>U</c> / <c>StringDType</c> operand.
        /// </summary>
        /// <param name="a">The text (<c>np.asarray</c>'d by the caller).</param>
        /// <param name="encoding">The encoding (null: none, UTF-8); converted as <c>np.asarray</c> converts it.</param>
        /// <param name="errors">The error handler (null: none, strict; ignored when <paramref name="encoding"/> is null).</param>
        /// <returns>A fresh C-ordered <c>S</c> array as wide as the longest encoding (at least 1), or an empty <c>S1</c> array
        ///     over an empty broadcast.</returns>
        /// <exception cref="TypeError">a is not text; a StringDType NA that is not a str; an argument that is not a str.</exception>
        /// <exception cref="AttributeError">a is <c>S</c>.</exception>
        /// <exception cref="ValueError">An argument with a NUL; a ragged argument list.</exception>
        /// <exception cref="UnicodeEncodeError">An unencodable character (or a surrogate in an argument).</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec; an unknown handler name at the first error.</exception>
        /// <exception cref="SystemError">A <c>U</c> one-unit element past U+10FFFF.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="NotSupportedException">A codec NumSharp does not implement; an object array of lists as an
        ///     argument.</exception>
        internal static NDArray Encode(NDArray a, object encoding, object errors) => Code(a, encoding, errors, encode: true);

        /// <summary>
        ///     NumPy's <c>np.strings.decode(a, encoding, errors)</c> (see the file header): <c>bytes.decode</c> per element of
        ///     an <c>S</c> operand.
        /// </summary>
        /// <param name="a">The bytes (<c>np.asarray</c>'d by the caller).</param>
        /// <param name="encoding">The encoding (null: none, UTF-8); converted as <c>np.asarray</c> converts it.</param>
        /// <param name="errors">The error handler (null: none, strict; ignored when <paramref name="encoding"/> is null).</param>
        /// <returns>A fresh C-ordered <c>U</c> array as wide as the longest decoding (at least 1), or an empty <c>U1</c> array
        ///     over an empty broadcast.</returns>
        /// <exception cref="TypeError">a is not text; an argument that is not a str; <c>xmlcharrefreplace</c> /
        ///     <c>namereplace</c> at an error.</exception>
        /// <exception cref="AttributeError">a is <c>U</c> or <c>StringDType</c>.</exception>
        /// <exception cref="ValueError">An argument with a NUL; a ragged argument list.</exception>
        /// <exception cref="UnicodeDecodeError">Undecodable bytes.</exception>
        /// <exception cref="UnicodeEncodeError">A surrogate in an argument.</exception>
        /// <exception cref="UnicodeError">The <c>undefined</c> codec over a non-empty element.</exception>
        /// <exception cref="LookupError">An unknown or non-text codec (at the first non-empty element); an unknown handler name
        ///     at the first error.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="NotSupportedException">A codec NumSharp does not implement; an object array of lists as an
        ///     argument.</exception>
        internal static NDArray Decode(NDArray a, object encoding, object errors) => Code(a, encoding, errors, encode: false);

        /// <summary>The shared driver of <see cref="Encode"/> and <see cref="Decode"/> (see the file header).</summary>
        /// <param name="a">The operand.</param>
        /// <param name="encoding">The encoding argument, or null.</param>
        /// <param name="errors">The errors argument, or null.</param>
        /// <param name="encode">True for <c>str.encode</c>, false for <c>bytes.decode</c>.</param>
        /// <returns>The result.</returns>
        private static NDArray Code(NDArray a, object encoding, object errors, bool encode)
        {
            var loop = NDStringMethods.ResolveOne(a);
            // _vec_string: the method by a's dtype, before any argument is converted.
            if (loop == NDStringMethods.TextLoop.None)
                throw new TypeError(NonStringArrayText);
            if (encode && loop == NDStringMethods.TextLoop.Bytes)
                throw new AttributeError(BytesEncodeText);
            if (!encode && loop != NDStringMethods.TextLoop.Bytes)
                throw new AttributeError(StrDecodeText);
            // _clean_args: everything from the first None on is dropped (errors without an encoding never reaches CPython).
            if (encoding is null)
                errors = null;
            using var enc = encoding is null ? null : VecArg.Convert(encoding, "encoding");
            using var err = errors is null ? null : VecArg.Convert(errors, "errors");
            if (enc?.Lists is not null || err?.Lists is not null)
                throw new NotSupportedException("An object array of lists as np.strings.encode / decode's encoding or errors has no NumSharp form.");
            var shapes = new List<Shape> { a.Shape };
            if (enc is not null)
                shapes.Add(enc.BroadcastShape);
            if (err is not null)
                shapes.Add(err.BroadcastShape);
            var outer = np.broadcast_shapes(shapes.ToArray());
            // _to_bytes_or_str_array: an empty result is result.astype(np.asarray(np.bytes_(b'')).dtype) — S1 (U1).
            if (outer.size == 0)
                return new NDArray(encode ? DTypeRegistry.Bytes.Descr(1) : DTypeRegistry.Str.Descr(1), outer.Clean(), true);

            using var self = VecArg.Convert(a, "a");
            var readers = BuildReaders(self, enc, err);
            CodecWalk w = null;
            try
            {
                w = new CodecWalk(readers, encode, enc is not null, err is not null, outer);
                var ops = new List<NDArray> { a };
                if (enc is not null)
                    ops.Add(enc.Array);
                if (err is not null)
                    ops.Add(err.Array);
                Walk(ops, readers, (dataptrs, strides, n, aux) => CodecRun(dataptrs, strides, n, w));
                return w.Rows.FinishRows();
            }
            finally
            {
                w?.Dispose();
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        // =============================================================================================================
        // The walk's state
        // =============================================================================================================

        /// <summary>
        ///     The state of one encode / decode call shared by its NDIter chunks: the readers, the validated arguments and the
        ///     codec they name, the result rows, and the scratch a StringDType element's code points are decoded into.
        /// </summary>
        /// <remarks>
        ///     Owns the rows and the scratch (pool buffers, returned by <see cref="Dispose"/>); the readers belong to the driver.
        ///     Single-threaded: the walk visits its chunks in C order on the calling thread.
        /// </remarks>
        private sealed class CodecWalk : IDisposable
        {
            /// <summary>True for <c>str.encode</c>, false for <c>bytes.decode</c>.</summary>
            internal readonly bool EncodeMode;

            /// <summary>a's reader (its kind, width, StringDType allocator and NA object).</summary>
            [NDBorrowed] // the driver creates and disposes the readers around the walk
            internal readonly Operand AReader;

            /// <summary>The encoding argument's reader, or null when no encoding is given.</summary>
            [NDBorrowed]
            internal readonly Operand EncReader;

            /// <summary>The errors argument's reader, or null when none is given (or the encoding is not).</summary>
            [NDBorrowed]
            internal readonly Operand ErrReader;

            /// <summary>The result rows (bytes for encode, code points for decode).</summary>
            internal readonly CodecRows Rows;

            /// <summary>The code points of a StringDType element decoded for a codec other than UTF-8.</summary>
            internal readonly NDCodecSink Points;

            /// <summary>The address the encoding argument was last read from (an element address is its identity), or null.</summary>
            private byte* _encAddress;

            /// <summary>The address the errors argument was last read from, or null.</summary>
            private byte* _errAddress;

            /// <summary>The validated encoding name of the last read element (null: no encoding argument).</summary>
            private string _encodingName;

            /// <summary>The code points of the last validated encoding element (for a varying argument's content compare).</summary>
            private uint[] _encodingUnits = Array.Empty<uint>();

            /// <summary>The code points of the last validated errors element.</summary>
            private uint[] _errorsUnits = Array.Empty<uint>();

            /// <summary>Whether the arguments read last still need Argument Clinic's checks.</summary>
            private bool _argsPending;

            /// <summary>The codec of the current element's encoding, or null before it is looked up.</summary>
            internal NDCodec Codec;

            /// <summary>The errors policy of the current element.</summary>
            internal NDCodecErrors Errors = NDCodecErrors.Strict;

            /// <summary>The name the current <see cref="Codec"/> was looked up by (a varying argument re-looks only on a change).</summary>
            private string _codecName;

            /// <summary>Creates the state of one call.</summary>
            /// <param name="readers">The readers: a's, then the encoding's and the errors' when given.</param>
            /// <param name="encode">True for encode.</param>
            /// <param name="hasEncoding">Whether an encoding argument is given.</param>
            /// <param name="hasErrors">Whether an errors argument is given.</param>
            /// <param name="outer">The broadcast shape (the result's shape, at least one element).</param>
            /// <exception cref="OutOfMemoryException">The scratch cannot be allocated.</exception>
            internal CodecWalk(Operand[] readers, bool encode, bool hasEncoding, bool hasErrors, Shape outer)
            {
                EncodeMode = encode;
                AReader = readers[0];
                EncReader = hasEncoding ? readers[1] : null;
                ErrReader = hasErrors ? readers[2] : null;
                _argsPending = hasEncoding;
                Rows = new CodecRows(encode ? 1 : 4, outer.size, outer);
                Points = new NDCodecSink(4, 64);
                // No encoding argument: CPython's encoding == NULL is UTF-8, looked up nowhere.
                if (!hasEncoding)
                    Codec = NDCodec.Utf8;
            }

            /// <summary>The number of elements <see cref="EstimateRows"/> looks at (the rows' sample).</summary>
            private const int EstimateWindow = 64;

            /// <summary>The element stride of the chunk the general route is in (set by the chunk loop before each general
            ///     element), so the element routines can look ahead from their element.</summary>
            internal long WindowStride;

            /// <summary>The number of elements left in that chunk, the current one included.</summary>
            internal long WindowCount;

            /// <summary>
            ///     The position before which every element is known to fit the rows, because <see cref="EstimateRows"/> sized
            ///     each one exactly (the namereplace pre-pass); 0 otherwise. Valid while the rows only grow, which they do: a
            ///     re-layout widens them.
            /// </summary>
            internal long SizedThrough;

            /// <summary>
            ///     Starts the rows before the first element is stored, at the longest result among the next
            ///     <see cref="EstimateWindow"/> elements of the chunk from <paramref name="pa"/> — each one coded by the codec
            ///     into a scratch output (<see cref="EstimateLength"/>), so the width is exact for the window, error handlers
            ///     included — and every element can then take a fast run from the first (no sampling through the result rows).
            ///     An element past the estimate re-lays the rows out; an overshoot is narrowed once at the end.
            /// </summary>
            /// <param name="pa">The first element (the one being coded).</param>
            /// <exception cref="OutOfMemoryException">The rows cannot be allocated.</exception>
            internal void EstimateRows(byte* pa)
            {
                if (EncodeMode && AReader.Kind == OperandKind.Unicode
                    && Codec.Kind is PythonCodecKind.Ascii or PythonCodecKind.Latin1 && IsNameReplace(Errors))
                {
                    // namereplace rows are wide (a name per unencodable character) and their longest element is rare: a
                    // re-layout of ~300-byte rows late in a long call copies megabytes (and a final narrowing as many), so the
                    // whole chunk is sized exactly instead — the memoized name lengths of the unencodable characters, gathered
                    // eight at a time (NameReplacedWidest). Nothing here raises (a one-unit text past U+10FFFF is the walk's
                    // SystemError, met in C order).
                    uint limit = Codec.Kind == PythonCodecKind.Latin1 ? 0x100u : 0x80u;
                    long widest = NameReplacedWidest(pa, WindowStride, WindowCount, AReader.Chars, limit);
                    Rows.StartRowsEarly(widest);
                    // Every element of this chunk fits the rows: the namereplace run skips their sizing.
                    SizedThrough = Rows.Position + WindowCount;
                    return;
                }
                long window = Math.Min(EstimateWindow, WindowCount);
                long longest = 0;
                for (long i = 0; i < window; i++, pa += WindowStride)
                    longest = Math.Max(longest, EstimateLength(pa));
                Rows.StartRowsEarly(longest);
            }

            /// <summary>
            ///     The length (in result units) the current codec gives one element: the element coded into the estimate scratch.
            ///     An element the walk will refuse or whose coding raises sizes as 0 and leaves its error to the walk — the
            ///     estimate looks AHEAD of the element being coded, and an error of a later element must not be raised before the
            ///     earlier elements' own errors (C order).
            /// </summary>
            /// <param name="pa">The element.</param>
            /// <returns>The length, or 0.</returns>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private long EstimateLength(byte* pa)
            {
                var sink = EstimateSink;
                sink.Count = 0;
                try
                {
                    switch (AReader.Kind)
                    {
                        case OperandKind.Bytes:
                        {
                            int n = NDCharArray.TrimmedUnits(pa, AReader.Chars, false);
                            if (n > 0)
                                Codec.Decode(pa, n, Errors, sink);
                            break;
                        }
                        case OperandKind.Unicode:
                        {
                            int n = NDCharArray.TrimmedUnits(pa, AReader.Chars, true);
                            // A one-unit element past U+10FFFF is the walk's SystemError, not a length.
                            if (n == 1 && *(uint*)pa > 0x10FFFF)
                                return 0;
                            Codec.Encode((uint*)pa, n, Errors, sink);
                            break;
                        }
                        default:
                        {
                            // A StringDType cell: its UTF-8 decoded to code points (an NA cell is the walk's to read).
                            if (AReader.Allocator.load(pa, out byte* buf, out long size) != 0)
                                return 0;
                            var points = Points;
                            points.Count = 0;
                            AppendUtf8Points(buf, size, points);
                            Codec.Encode((uint*)points.Data, points.Count, Errors, sink);
                            break;
                        }
                    }
                }
                catch (Exception) when (true)
                {
                    // Any error of a looked-ahead element (a codec error, an unknown handler name, the undefined codec): the
                    // walk raises it at that element, in C order.
                    return 0;
                }
                return sink.Count;
            }

            /// <summary>The scratch output of <see cref="EstimateLength"/> (the result's unit: bytes for encode, code points for
            ///     decode), created at the first estimate.</summary>
            private NDCodecSink EstimateSink => _estimateSink ??= new NDCodecSink(EncodeMode ? 1 : 4, 256);

            /// <summary>The backing field of <see cref="EstimateSink"/>.</summary>
            private NDCodecSink _estimateSink;

            /// <summary>Whether an encoding argument is given (an operand of the walk after a).</summary>
            internal bool HasEncoding => EncReader is not null;

            /// <summary>
            ///     Whether a fast run may code the next elements: the rows exist, the codec is known, and the argument elements at
            ///     <paramref name="pe"/> / <paramref name="pr"/> are the ones read and checked last (their PyArray_ToScalar and
            ///     Argument Clinic's checks cannot raise again; the caller guarantees they do not move inside the chunk).
            /// </summary>
            /// <param name="pe">The encoding element's address (null without an encoding).</param>
            /// <param name="pr">The errors element's address (null without errors).</param>
            /// <returns>True when the fast run applies.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal bool FastReady(byte* pe, byte* pr)
                => !CodecFastPathsDisabled && Codec is not null && Rows.RowsActive && !_argsPending && pe == _encAddress
                   && pr == _errAddress;

            /// <summary>Whether an errors argument is given.</summary>
            internal bool HasErrors => ErrReader is not null;

            /// <summary>
            ///     PyArray_ToScalar of the arguments at the current element (a SystemError for a <c>U</c> one-unit argument past
            ///     U+10FFFF), skipped for an address read before; a new element is marked for Argument Clinic's checks
            ///     (<see cref="CheckArgs"/>), which the caller runs after a's descriptor check.
            /// </summary>
            /// <param name="pe">The encoding element's address (null without an encoding).</param>
            /// <param name="pr">The errors element's address (null without errors).</param>
            /// <exception cref="SystemError">A one-unit <c>U</c> argument past U+10FFFF.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void ReadArgs(byte* pe, byte* pr)
            {
                if (pe == _encAddress && pr == _errAddress)
                    return;
                ReadArgsSlow(pe, pr);
            }

            /// <summary>The element reads of <see cref="ReadArgs"/> for an address not read before.</summary>
            /// <param name="pe">The encoding element's address.</param>
            /// <param name="pr">The errors element's address.</param>
            /// <exception cref="SystemError">A one-unit <c>U</c> argument past U+10FFFF.</exception>
            private void ReadArgsSlow(byte* pe, byte* pr)
            {
                if (pe != _encAddress)
                {
                    EncReader.Read(pe, out _encElem);
                    _encAddress = pe;
                    _argsPending = true;
                }
                if (pr != _errAddress)
                {
                    ErrReader.Read(pr, out _errElem);
                    _errAddress = pr;
                    _argsPending = true;
                }
            }

            /// <summary>The encoding element read last (valid until the encoding reader reads again).</summary>
            private Elem _encElem;

            /// <summary>The errors element read last.</summary>
            private Elem _errElem;

            /// <summary>
            ///     Argument Clinic's checks of the elements read last (encoding, then errors) when they changed, and the name
            ///     and policy they give; the codec is looked up by <see cref="EnsureCodec"/>.
            /// </summary>
            /// <exception cref="TypeError">An argument that is not a str.</exception>
            /// <exception cref="UnicodeEncodeError">An argument holding a surrogate.</exception>
            /// <exception cref="ValueError">An argument holding a NUL.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void CheckArgs()
            {
                if (_argsPending)
                    CheckArgsSlow();
            }

            /// <summary>The checks of <see cref="CheckArgs"/> for elements not validated before.</summary>
            /// <exception cref="TypeError">An argument that is not a str.</exception>
            /// <exception cref="UnicodeEncodeError">An argument holding a surrogate.</exception>
            /// <exception cref="ValueError">An argument holding a NUL.</exception>
            private void CheckArgsSlow()
            {
                string fn = EncodeMode ? "encode" : "decode";
                if (HasEncoding && !SameText(in _encElem, _encodingUnits))
                {
                    _encodingName = ArgumentText(in _encElem, fn, "encoding", out _encodingUnits);
                    Codec = null;   // looked up again (by name) at the element that needs it
                }
                if (HasErrors && !SameText(in _errElem, _errorsUnits))
                {
                    string name = ArgumentText(in _errElem, fn, "errors", out _errorsUnits);
                    Errors = new NDCodecErrors(name);
                }
                _argsPending = false;
            }

            /// <summary>
            ///     Whether a str element holds the same code points as the argument validated last (a varying argument array
            ///     repeats a few encodings; only a str can have been validated).
            /// </summary>
            /// <param name="e">The element.</param>
            /// <param name="units">The code points validated last.</param>
            /// <returns>True when the element is a str with exactly those code points.</returns>
            private static bool SameText(in Elem e, uint[] units)
            {
                if (!IsStr(e) || e.Length != units.Length || units.Length == 0)
                    return false;
                return new ReadOnlySpan<uint>(e.Units, units.Length).SequenceEqual(units);
            }

            /// <summary>
            ///     Argument Clinic's conversion of one encoding / errors argument: a str (np.str_ or str), whose UTF-8 holds no
            ///     surrogate (<c>PyUnicode_AsUTF8AndSize</c>) and no NUL (the <c>strlen</c> check); the name as a .NET string.
            /// </summary>
            /// <param name="e">The element.</param>
            /// <param name="fn">The method (<c>encode</c> / <c>decode</c>).</param>
            /// <param name="parameter">The parameter (<c>encoding</c> / <c>errors</c>).</param>
            /// <param name="units">Receives a copy of the element's code points (the content-compare key).</param>
            /// <returns>The name: each code point a character (U+10000.. as a surrogate pair), a unit past U+10FFFF as the four
            ///     U+FFFD CPython prints for the four invalid bytes it encodes it as.</returns>
            /// <exception cref="TypeError"><c>encode() argument 'encoding' must be str, not numpy.bytes_</c> (None reads
            ///     <c>None</c>).</exception>
            /// <exception cref="UnicodeEncodeError">A surrogate (the run of them, CPython's UTF-8 text).</exception>
            /// <exception cref="ValueError"><c>embedded null character</c>.</exception>
            private static string ArgumentText(in Elem e, string fn, string parameter, out uint[] units)
            {
                if (!IsStr(e))
                {
                    // _PyArg_BadArgument: "None" for Python's None, the type's tp_name otherwise.
                    string type = IsNone(in e) ? "None" : TypeName(in e);
                    throw new TypeError($"{fn}() argument '{parameter}' must be str, not {type}");
                }
                uint* p = (uint*)e.Units;
                long n = e.Length;
                // PyUnicode_AsUTF8AndSize: the strict UTF-8 encoder raises over a run of surrogates.
                for (long i = 0; i < n; i++)
                {
                    if (p[i] - 0xD800u <= 0x7FFu)
                    {
                        long end = i + 1;
                        while (end < n && p[end] - 0xD800u <= 0x7FFu)
                            end++;
                        throw NDCodec.EncodeError("utf-8", p, i, end, "surrogates not allowed");
                    }
                }
                for (long i = 0; i < n; i++)
                    if (p[i] == 0)
                        throw new ValueError(EmbeddedNullText);
                units = new ReadOnlySpan<uint>(p, checked((int)n)).ToArray();
                var sb = new System.Text.StringBuilder(checked((int)n));
                for (long i = 0; i < n; i++)
                {
                    uint u = p[i];
                    if (u > 0x10FFFF)
                        sb.Append("����");
                    else if (u > 0xFFFF)
                        sb.Append(char.ConvertFromUtf32((int)u));
                    else
                        sb.Append((char)u);
                }
                return sb.ToString();
            }

            /// <summary>
            ///     The codec of the current element (looked up by name once per distinct name; no encoding argument is UTF-8):
            ///     <c>PyUnicode_AsEncodedString</c>'s / <c>PyUnicode_Decode</c>'s lookup.
            /// </summary>
            /// <returns>The codec.</returns>
            /// <exception cref="LookupError">An unknown or non-text codec.</exception>
            /// <exception cref="NotSupportedException">A codec NumSharp does not implement.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal NDCodec EnsureCodec() => Codec ?? LookupCodec();

            /// <summary>The lookup of <see cref="EnsureCodec"/>.</summary>
            /// <returns>The codec.</returns>
            /// <exception cref="LookupError">An unknown or non-text codec.</exception>
            /// <exception cref="NotSupportedException">A codec NumSharp does not implement.</exception>
            private NDCodec LookupCodec()
            {
                // A varying encoding array that returns to a name looked up before keeps that codec.
                if (_codecName is null || !string.Equals(_codecName, _encodingName, StringComparison.Ordinal))
                {
                    _lastCodec = NDCodec.Lookup(_encodingName, EncodeMode);
                    _codecName = _encodingName;
                }
                Codec = _lastCodec;
                return Codec;
            }

            /// <summary>The codec <see cref="_codecName"/> names.</summary>
            private NDCodec _lastCodec;

            /// <summary>Returns the rows and the scratch to the pool.</summary>
            public void Dispose()
            {
                Rows.Dispose();
                Points.Dispose();
                _estimateSink?.Dispose();
                _estimateSink = null;
            }
        }

        // =============================================================================================================
        // The result rows
        // =============================================================================================================

        /// <summary>
        ///     The <c>S</c> (encode) / <c>U</c> (decode) result of one call, built row by row in C order: the first
        ///     <see cref="RowSample"/> elements are kept in the arena to choose the width, a longer element later re-lays the
        ///     rows out, and <see cref="FinishRows"/> narrows them once if the width overshot the longest element.
        /// </summary>
        /// <remarks>
        ///     The codecs append to <see cref="Arena"/>; <see cref="StoreRow"/> moves the element just coded into its row and
        ///     empties the arena. A fast path writes an element straight into its row (<see cref="RowBase"/>,
        ///     <see cref="RowWidth"/>) and commits it with <see cref="CommitDirect"/>. Owns the arena and the result until
        ///     <see cref="FinishRows"/> hands it out. Internal rather than private so that a unit test can drive the growth
        ///     policy directly (<see cref="FastWidthLimit"/>: the width only changes the speed, which no result shows).
        /// </remarks>
        internal sealed class CodecRows : IDisposable
        {
            /// <summary>The number of first elements kept in the arena to choose the width.</summary>
            internal const int RowSample = 64;

            /// <summary>The arena: the element being coded (or, while the width is sampled, the first elements back to back).</summary>
            internal readonly NDCodecSink Arena;

            /// <summary>The unit size in bytes: 1 (an <c>S</c> result) or 4 (a <c>U</c> result).</summary>
            private readonly int _unit;

            /// <summary>The number of elements (the broadcast size, at least 1).</summary>
            private readonly long _size;

            /// <summary>The result's shape (C order).</summary>
            private readonly Shape _resultShape;

            /// <summary>The lengths of the sampled elements.</summary>
            private readonly long[] _sampleLengths = new long[RowSample];

            /// <summary>The result being filled, or null while the width is sampled (and after <see cref="FinishRows"/>).</summary>
            private NDArray _rows;

            /// <summary>The first row of <see cref="_rows"/>.</summary>
            private byte* _rowBase;

            /// <summary>The width of <see cref="_rows"/> in units.</summary>
            private int _rowWidth;

            /// <summary>How many times the rows were re-laid out (the first few grow to the exact length, later ones by half again).</summary>
            private int _relayouts;

            /// <summary>
            ///     The widest row, in units, that the call's fast run writes; each run sets it when it starts (0 before any run
            ///     has, or for a run that writes any width). <see cref="GrowRows"/> never grows the rows past it for an element
            ///     no wider than it.
            /// </summary>
            /// <remarks>
            ///     Rows wider than the run's limit send every later element of the call down the run's per-element exact-write
            ///     path, which codes an element holding an error through the codec. Growing by half again from about 112 bytes
            ///     carried ASCII <c>xmlcharrefreplace</c> rows to 168, past the narrow run's 128, although no element was longer
            ///     than 114: 97% of the elements of 100K CJK texts then took the exact-write path.
            /// </remarks>
            internal int FastWidthLimit;

            /// <summary>The next element's C-order position.</summary>
            internal long Position;

            /// <summary>The longest element so far, in units.</summary>
            internal long MaxLength;

            /// <summary>Creates the rows of a result.</summary>
            /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
            /// <param name="size">The number of elements.</param>
            /// <param name="outer">The result's shape.</param>
            /// <exception cref="OutOfMemoryException">The arena cannot be allocated.</exception>
            internal CodecRows(int unit, long size, Shape outer)
            {
                _unit = unit;
                _size = size;
                _resultShape = outer.Clean();
                // Room for the sampled elements at a typical length; the arena doubles past it.
                Arena = new NDCodecSink(unit, RowSample * 32);
            }

            /// <summary>Whether the rows exist (the width was chosen): a fast path may then write an element into its row.</summary>
            internal bool RowsActive => _rows is not null;

            /// <summary>The first row (valid until the rows are re-laid out).</summary>
            internal byte* RowBase => _rowBase;

            /// <summary>The rows' width in units.</summary>
            internal int RowWidth => _rowWidth;

            /// <summary>
            ///     Stores the element just coded (the last <paramref name="length"/> units of the arena) as the row at
            ///     <see cref="Position"/> and advances it; while the width is sampled the element stays in the arena, and the
            ///     sample's last element starts the rows.
            /// </summary>
            /// <param name="length">The element's length in units.</param>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            internal void StoreRow(long length)
            {
                if (length > MaxLength)
                    MaxLength = length;
                if (_rows is null)
                {
                    _sampleLengths[Position] = length;
                    Position++;
                    if (Position == RowSample || Position == _size)
                        StartRows();
                    return;
                }
                if (length > _rowWidth)
                    GrowRows(length);
                // The vector row copy reads whole vectors from the element's start: room past it in the arena (reserving may
                // move the arena, so the element's address is taken after).
                Arena.Reserve(VectorRowReach);
                CopyRow(Arena.Data + (Arena.Count - length) * _unit, length, Position);
                Arena.Count = 0;
                Position++;
            }

            /// <summary>The number of elements (the result's size).</summary>
            internal long Size => _size;

            /// <summary>
            ///     The first position whose row, written by a store of <paramref name="bytes"/> bytes from its start, could pass
            ///     the result's end: a fast path stores whole vectors (their tail spilling into the following rows, which C order
            ///     writes later) only before it.
            /// </summary>
            /// <param name="bytes">The store's size in bytes.</param>
            /// <returns>The limit position (rows before it take the vector store).</returns>
            internal long VectorStoreLimit(int bytes)
            {
                long rowBytes = (long)_rowWidth * _unit;
                return _size - (bytes + rowBytes - 1) / rowBytes;
            }

            /// <summary>
            ///     Commits an element a fast path wrote straight into its row (its units, then zeros to the row's width): the
            ///     longest length, then the next position.
            /// </summary>
            /// <param name="length">The element's length (at most <see cref="RowWidth"/>).</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void CommitDirect(long length)
            {
                if (length > MaxLength)
                    MaxLength = length;
                Position++;
            }

            /// <summary>
            ///     Starts the rows at a known width before any element was stored (no sampling): the caller's bound of every
            ///     element's length, at least 1.
            /// </summary>
            /// <param name="width">The width in units.</param>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            internal void StartRowsEarly(long width)
            {
                _rowWidth = checked((int)Math.Max(1, width));
                // ForOverwrite: every row is written whole (its units, then zeros) before the result is handed out, and a call
                // that raises disposes the rows unread. A U result through new NDArray(...) would take NumPy's NPY_NEEDS_INIT
                // zero fill, which above a size is fresh OS pages: every call then page-faulted while it wrote its rows and
                // paid a VirtualFree when the result was disposed (decode of 100K S8 words: ~10 ns of ~13 per element).
                _rows = NDArray.ForOverwrite(RowDescr(_rowWidth), _resultShape);
                _rowBase = (byte*)_rows.Storage.Address;
            }

            /// <summary>The descriptor of a result width.</summary>
            /// <param name="width">The width in characters.</param>
            /// <returns><c>S{width}</c> or <c>U{width}</c>.</returns>
            private DType RowDescr(int width) => _unit == 1 ? DTypeRegistry.Bytes.Descr(width) : DTypeRegistry.Str.Descr(width);

            /// <summary>Allocates the result at the sample's longest length (at least 1) and stores the sampled elements.</summary>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            private void StartRows()
            {
                _rowWidth = checked((int)Math.Max(1, MaxLength));
                // Written whole before it is handed out (see StartRowsEarly): no NUL fill.
                _rows = NDArray.ForOverwrite(RowDescr(_rowWidth), _resultShape);
                _rowBase = (byte*)_rows.Storage.Address;
                // The vector row copies load whole vectors from an element's start: room past the last sampled element.
                Arena.Reserve(VectorRowReach);
                byte* src = Arena.Data;
                for (long i = 0; i < Position; i++)
                {
                    CopyRow(src, _sampleLengths[i], i);
                    src += _sampleLengths[i] * _unit;
                }
                Arena.Count = 0;
            }

            /// <summary>
            ///     Re-lays the rows written so far out at a larger width (zero-padded): the exact length for the first few
            ///     re-layouts, at least half again afterwards (so increasing lengths cost O(log width) passes), but never past
            ///     <see cref="FastWidthLimit"/> for an element no wider than it.
            /// </summary>
            /// <param name="length">The new element's length.</param>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            private void GrowRows(long length)
            {
                long wanted = _relayouts < 4 ? length : Math.Max(length, _rowWidth + _rowWidth / 2L);
                // Capped at the fast run's limit when the element fits it: every later element no wider than the limit then
                // fits too, so the cap costs at most one more re-layout, while rows past the limit would take the slow path
                // for the rest of the call (see FastWidthLimit).
                if (length <= FastWidthLimit && wanted > FastWidthLimit)
                    wanted = FastWidthLimit;
                int width = checked((int)wanted);
                // The rows written so far are copied (padded) below, the later ones are written as their elements come.
                var grown = NDArray.ForOverwrite(RowDescr(width), _resultShape);
                CopyRows(_rowBase, _rowWidth, (byte*)grown.Storage.Address, width, Position);
                _rows.Dispose();
                _rows = grown;
                _rowBase = (byte*)grown.Storage.Address;
                _rowWidth = width;
                _relayouts++;
            }

            /// <summary>Hands the finished result out: as wide as the longest element (at least 1), narrowed by one copy when the
            ///     rows' width overshot it.</summary>
            /// <returns>The result (owned by the caller from now on).</returns>
            /// <exception cref="OutOfMemoryException">The narrowed result cannot be allocated.</exception>
            internal NDArray FinishRows()
            {
                int width = checked((int)Math.Max(1, MaxLength));
                if (width != _rowWidth)
                {
                    // Every row is copied into it: no NUL fill.
                    var narrow = NDArray.ForOverwrite(RowDescr(width), _resultShape);
                    CopyRows(_rowBase, _rowWidth, (byte*)narrow.Storage.Address, width, _size);
                    _rows.Dispose();
                    _rows = narrow;
                }
                var result = _rows;
                _rows = null;
                _rowBase = null;
                return result;
            }

            /// <summary>How far past an element's start the vector row copy reads, in arena units: two 32-byte vectors of a
            ///     bytes row (64 units) or four 8-unit vectors of a code-point row (32 units), at most 64.</summary>
            private const int VectorRowReach = 64;

            /// <summary>
            ///     Writes one element into row <paramref name="row"/>: its units, then zeros to the row's width. A row of up to 64
            ///     bytes (<c>S</c>) or 32 code points (<c>U</c>) is written by whole masked 32-byte stores, which may also cover
            ///     the start of the following rows — rewritten when their elements are stored, as rows are stored in C order —
            ///     but never past the result's end (the last rows take the exact copy).
            /// </summary>
            /// <param name="src">The element's units (the arena has <see cref="VectorRowReach"/> units of room past them).</param>
            /// <param name="length">Their number (at most the width).</param>
            /// <param name="row">The row.</param>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private void CopyRow(byte* src, long length, long row)
            {
                int width = _rowWidth;
                if (_unit == 1)
                {
                    byte* dst = _rowBase + row * width;
                    int vectors = (width + 31) / 32;
                    if (Vector256.IsHardwareAccelerated && vectors <= 2 && row * width + vectors * 32L <= _size * width)
                    {
                        // The lane numbers 0..31 (Vector256<T>.Indices is .NET 9+; this file builds for net8.0 too).
                        var lanes = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
                        var count = Vector256.Create((byte)length);
                        (Vector256.Load(src) & Vector256.LessThan(lanes, count)).Store(dst);
                        if (vectors == 2)
                            (Vector256.Load(src + 32) & Vector256.LessThan(lanes + Vector256.Create((byte)32), count)).Store(dst + 32);
                        return;
                    }
                    Buffer.MemoryCopy(src, dst, length, length);
                    new Span<byte>(dst + length, (int)(width - length)).Clear();
                }
                else
                {
                    uint* dst = (uint*)_rowBase + row * width;
                    int vectors = (width + 7) / 8;
                    if (Vector256.IsHardwareAccelerated && vectors <= 4 && row * width + vectors * 8L <= _size * width)
                    {
                        var lanes = Vector256.Create(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u);
                        var count = Vector256.Create((uint)length);
                        uint* s = (uint*)src;
                        for (int k = 0; k < vectors; k++)
                        {
                            (Vector256.Load(s + 8 * k) & Vector256.LessThan(lanes, count)).Store(dst + 8 * k);
                            lanes += Vector256.Create(8u);
                        }
                        return;
                    }
                    Buffer.MemoryCopy(src, dst, length * 4, length * 4);
                    new Span<uint>(dst + length, (int)(width - length)).Clear();
                }
            }

            /// <summary>Copies the first <paramref name="rows"/> rows of one width into rows of another (zero-padded when
            ///     wider; a narrower width never cuts an element, which <see cref="MaxLength"/> bounds).</summary>
            /// <param name="src">The source rows.</param>
            /// <param name="srcWidth">Their width in units.</param>
            /// <param name="dst">The destination rows.</param>
            /// <param name="dstWidth">Their width in units.</param>
            /// <param name="rows">The number of rows.</param>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private void CopyRows(byte* src, int srcWidth, byte* dst, int dstWidth, long rows)
            {
                long keep = Math.Min(srcWidth, dstWidth) * (long)_unit;
                long pad = (dstWidth - Math.Min(srcWidth, dstWidth)) * (long)_unit;
                long srcRow = (long)srcWidth * _unit, dstRow = (long)dstWidth * _unit;
                long i = 0;
                if (Vector256.IsHardwareAccelerated && keep <= 32 && dstRow <= 32)
                {
                    // One 32-byte load per source row (masked to the kept bytes: the bytes past them are zero, the padding)
                    // and one 32-byte store per destination row — the load may read the following source rows and the store
                    // writes over the following destination rows, which the next iterations rewrite, so both stop where a
                    // whole vector would pass their buffer's end; those last rows take the exact copy.
                    var lanes = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
                    var mask = Vector256.LessThan(lanes, Vector256.Create((byte)keep));
                    long srcEnd = _size * srcRow, dstEnd = _size * dstRow;
                    for (; i < rows; i++)
                    {
                        if (i * srcRow + 32 > srcEnd || i * dstRow + 32 > dstEnd)
                            break;
                        (Vector256.Load(src + i * srcRow) & mask).Store(dst + i * dstRow);
                    }
                }
                for (; i < rows; i++)
                {
                    byte* s = src + i * srcRow;
                    byte* d = dst + i * dstRow;
                    Buffer.MemoryCopy(s, d, keep, keep);
                    if (pad > 0)
                        new Span<byte>(d + keep, (int)pad).Clear();
                }
            }

            /// <summary>Returns the arena and an unfinished result to the pool.</summary>
            public void Dispose()
            {
                Arena.Dispose();
                _rows?.Dispose();
                _rows = null;
                _rowBase = null;
            }
        }

        // =============================================================================================================
        // The chunk loop
        // =============================================================================================================

        /// <summary>
        ///     One NDIter chunk of an encode / decode call: per element in C order, PyArray_ToScalar of every operand (a first),
        ///     a's descriptor check, the arguments' checks, the codec, the element coded into its row (see the file header).
        /// </summary>
        /// <param name="dataptrs">The operands' data pointers: a, then the encoding and the errors when given.</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements in the chunk.</param>
        /// <param name="w">The call's state.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void CodecRun(void** dataptrs, long* strides, long n, CodecWalk w)
        {
            byte* pa = (byte*)dataptrs[0];
            long sa = strides[0];
            int k = 1;
            byte* pe = null, pr = null;
            long se = 0, sr = 0;
            if (w.HasEncoding)
            {
                pe = (byte*)dataptrs[k];
                se = strides[k];
                k++;
            }
            if (w.HasErrors)
            {
                pr = (byte*)dataptrs[k];
                sr = strides[k];
            }
            var reader = w.AReader;
            bool stationary = se == 0 && sr == 0;
            long i = 0;
            while (i < n)
            {
                // A fast run (CodecFast.cs) codes elements with everything per call hoisted — once the rows exist, the codec
                // is known and the arguments, which must not move inside the chunk, were checked at these addresses — up to
                // the first element it does not handle, which then takes the general route below.
                if (stationary && w.FastReady(pe, pr))
                {
                    long done = FastRun(w, pa, sa, n - i);
                    i += done;
                    pa += done * sa;
                    if (i == n)
                        return;
                }
                w.WindowStride = sa;
                w.WindowCount = n - i;
                switch (reader.Kind)
                {
                    case OperandKind.Unicode:
                        EncodeUnicodeElement(w, pa, pe, pr);
                        break;
                    case OperandKind.VString:
                        EncodeVStringElement(w, pa, pe, pr);
                        break;
                    default:
                        DecodeBytesElement(w, pa, pe, pr);
                        break;
                }
                i++;
                pa += sa;
                pe += se;
                pr += sr;
            }
        }

        /// <summary>
        ///     One <c>U</c> element of an encode: its np.str_ (trailing NULs dropped; a one-unit text past U+10FFFF is CPython's
        ///     SystemError), the arguments, then the codec — narrowed straight into its row when every code point is below the
        ///     codec's one-byte limit (UTF-8 / ASCII: 0x80, Latin-1: 0x100), else through the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The element.</param>
        /// <param name="pe">The encoding element (null without one).</param>
        /// <param name="pr">The errors element (null without one).</param>
        /// <exception cref="Exception">Whatever <see cref="CodecWalk.ReadArgs"/>, <see cref="CodecWalk.CheckArgs"/>,
        ///     <see cref="CodecWalk.EnsureCodec"/> or the codec raises.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void EncodeUnicodeElement(CodecWalk w, byte* pa, byte* pe, byte* pr)
        {
            uint* units = (uint*)pa;
            int n = NDCharArray.TrimmedUnits(pa, w.AReader.Chars, true);
            if (n == 1 && units[0] > 0x10FFFF)
                throw new SystemError(InvalidMaxCharText);
            if (w.HasEncoding)
            {
                w.ReadArgs(pe, pr);
                w.CheckArgs();
            }
            var codec = w.EnsureCodec();
            var rows = w.Rows;
            if (!CodecFastPathsDisabled && !rows.RowsActive && rows.Position == 0 && codec.Kind != PythonCodecKind.Undefined)
                w.EstimateRows(pa);
            if (!CodecFastPathsDisabled && rows.RowsActive && n <= rows.RowWidth)
            {
                uint limit = codec.Kind switch
                {
                    PythonCodecKind.Utf8 or PythonCodecKind.Ascii => 0x80u,
                    PythonCodecKind.Latin1 => 0x100u,
                    _ => 0u,
                };
                if (limit != 0)
                {
                    int width = rows.RowWidth;
                    byte* row = rows.RowBase + rows.Position * width;
                    // Narrowed optimistically: when a code point is at or past the limit the row is rewritten by the codec
                    // route below (StoreRow writes the whole row).
                    if (NarrowIfBelow(units, n, row, limit))
                    {
                        for (int z = n; z < width; z++)
                            row[z] = 0;
                        rows.CommitDirect(n);
                        return;
                    }
                }
            }
            var arena = rows.Arena;
            long start = arena.Count;
            codec.Encode(units, n, w.Errors, arena);
            rows.StoreRow(arena.Count - start);
        }

        /// <summary>
        ///     One <c>StringDType</c> element of an encode: a str (its UTF-8 cell), the NA object's str, or the descriptor's
        ///     TypeError for an NA object that is not a str; then the codec — a UTF-8 cell copied straight into its row under
        ///     UTF-8 (and under ASCII / Latin-1 when it is all ASCII), else decoded to code points and coded through the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The element's cell.</param>
        /// <param name="pe">The encoding element (null without one).</param>
        /// <param name="pr">The errors element (null without one).</param>
        /// <exception cref="TypeError"><c>descriptor 'encode' for 'str' objects doesn't apply to a 'NoneType' object</c>.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        /// <exception cref="Exception">Whatever the arguments or the codec raise.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void EncodeVStringElement(CodecWalk w, byte* pa, byte* pe, byte* pr)
        {
            var reader = w.AReader;
            int isNull = reader.Allocator.load(pa, out byte* buf, out long size);
            if (isNull < 0)
                throw new OutOfMemoryException("Failed to load string");
            if (w.HasEncoding)
                w.ReadArgs(pe, pr);
            string naText = null;
            if (isNull == 1)
            {
                // stringdtype_getitem: the NA object itself (a str NA is that str); without an NA object the empty str.
                var p = reader.Params;
                if (p.HasNAObject && p.na_object is not string)
                    throw new TypeError($"descriptor 'encode' for 'str' objects doesn't apply to a '{NDPyObject.TypeName(p.na_object)}' object");
                naText = p.HasNAObject ? (string)p.na_object : string.Empty;
            }
            if (w.HasEncoding)
                w.CheckArgs();
            var codec = w.EnsureCodec();
            var rows = w.Rows;
            if (!CodecFastPathsDisabled && !rows.RowsActive && rows.Position == 0 && codec.Kind != PythonCodecKind.Undefined)
                w.EstimateRows(pa);
            if (naText is null && !CodecFastPathsDisabled && rows.RowsActive && size <= rows.RowWidth
                && (codec.Kind == PythonCodecKind.Utf8 || ((codec.Kind is PythonCodecKind.Ascii or PythonCodecKind.Latin1)
                                                           && new ReadOnlySpan<byte>(buf, (int)size).IndexOfAnyExceptInRange((byte)0, (byte)0x7F) < 0)))
            {
                // A StringDType cell holds valid UTF-8 without surrogates: under UTF-8 its bytes ARE the encoding, and so
                // are ASCII bytes under ASCII / Latin-1.
                int width = rows.RowWidth;
                byte* row = rows.RowBase + rows.Position * width;
                Buffer.MemoryCopy(buf, row, size, size);
                for (long z = size; z < width; z++)
                    row[z] = 0;
                rows.CommitDirect(size);
                return;
            }
            // The str's code points: the cell's UTF-8 decoded (valid by construction), or the NA object's str.
            var points = w.Points;
            points.Count = 0;
            if (naText is not null)
                AppendStringPoints(naText, points);
            else
                AppendUtf8Points(buf, size, points);
            var arena = rows.Arena;
            long start = arena.Count;
            codec.Encode((uint*)points.Data, points.Count, w.Errors, arena);
            rows.StoreRow(arena.Count - start);
        }

        /// <summary>
        ///     One <c>S</c> element of a decode: its np.bytes_ (trailing NULs dropped), the arguments, then — for a non-empty
        ///     element only, as <c>PyUnicode_FromEncodedObject</c> returns '' for empty bytes before any lookup — the codec:
        ///     widened straight into its row when every byte is ASCII (UTF-8 / ASCII) or for any bytes (Latin-1), else through
        ///     the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The element.</param>
        /// <param name="pe">The encoding element (null without one).</param>
        /// <param name="pr">The errors element (null without one).</param>
        /// <exception cref="Exception">Whatever the arguments or the codec raise.</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DecodeBytesElement(CodecWalk w, byte* pa, byte* pe, byte* pr)
        {
            int n = NDCharArray.TrimmedUnits(pa, w.AReader.Chars, false);
            if (w.HasEncoding)
            {
                w.ReadArgs(pe, pr);
                w.CheckArgs();
            }
            var rows = w.Rows;
            if (n == 0)
            {
                // '' without a codec: an empty row.
                if (rows.RowsActive)
                {
                    new Span<uint>((uint*)rows.RowBase + rows.Position * rows.RowWidth, rows.RowWidth).Clear();
                    rows.CommitDirect(0);
                }
                else
                {
                    rows.StoreRow(0);
                }
                return;
            }
            var codec = w.EnsureCodec();
            if (!CodecFastPathsDisabled && !rows.RowsActive && rows.Position == 0 && codec.Kind != PythonCodecKind.Undefined)
                w.EstimateRows(pa);
            if (!CodecFastPathsDisabled && rows.RowsActive && n <= rows.RowWidth
                && codec.Kind is PythonCodecKind.Utf8 or PythonCodecKind.Ascii or PythonCodecKind.Latin1)
            {
                int width = rows.RowWidth;
                uint* row = (uint*)rows.RowBase + rows.Position * width;
                // Widened optimistically: a byte past 0x7F under UTF-8 / ASCII sends the element through the codec, which
                // rewrites the row.
                if (WidenIfAscii(pa, n, row, codec.Kind == PythonCodecKind.Latin1))
                {
                    for (int z = n; z < width; z++)
                        row[z] = 0;
                    rows.CommitDirect(n);
                    return;
                }
            }
            var arena = rows.Arena;
            long start = arena.Count;
            codec.Decode(pa, n, w.Errors, arena);
            rows.StoreRow(arena.Count - start);
        }

        // =============================================================================================================
        // The fast transcodes
        // =============================================================================================================

        /// <summary>
        ///     Writes <paramref name="n"/> code points as bytes (each truncated to its low byte) and reports whether every one
        ///     was below <paramref name="limit"/> (0x80 or 0x100) — the bytes are then the code points' UTF-8 / ASCII / Latin-1
        ///     encoding; otherwise the caller discards what was written.
        /// </summary>
        /// <param name="src">The code points (an element: its trimmed length is at most its width, so the loads stay inside it).</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The row (at least <paramref name="n"/> bytes).</param>
        /// <param name="limit">The exclusive limit, a power of two (an OR of the units below it stays below it).</param>
        /// <returns>True when every code point is below the limit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool NarrowIfBelow(uint* src, int n, byte* dst, uint limit)
        {
            int i = 0;
            uint acc = 0;
            if (Vector256.IsHardwareAccelerated && n >= 8)
            {
                var orv = Vector256<uint>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    var v = Vector256.Load(src + i);
                    orv |= v;
                    // Two narrowing steps keep lane order: 8 uints → 8 ushorts (the lower half) → 8 bytes (the lower 8).
                    var w16 = Vector256.Narrow(v, v).GetLower();
                    var b8 = Vector128.Narrow(w16, w16);
                    *(ulong*)(dst + i) = b8.AsUInt64().ToScalar();
                }
                // The OR of the eight lanes: fold the halves, then the four lanes.
                var o4 = orv.GetLower() | orv.GetUpper();
                acc = o4.GetElement(0) | o4.GetElement(1) | o4.GetElement(2) | o4.GetElement(3);
            }
            for (; i < n; i++)
            {
                uint u = src[i];
                acc |= u;
                dst[i] = (byte)u;
            }
            return acc < limit;
        }

        /// <summary>
        ///     Writes <paramref name="n"/> bytes as code points (zero-extended) and reports whether the result is the bytes'
        ///     decoding: always for Latin-1, for UTF-8 / ASCII when every byte is below 0x80 (otherwise the caller discards
        ///     what was written).
        /// </summary>
        /// <param name="src">The bytes (an element: its trimmed length is at most its width).</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The row (at least <paramref name="n"/> code points).</param>
        /// <param name="latin1">True for Latin-1 (every byte decodes to its value).</param>
        /// <returns>True when the written code points are the decoding.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool WidenIfAscii(byte* src, int n, uint* dst, bool latin1)
        {
            int i = 0;
            uint acc = 0;
            if (Vector256.IsHardwareAccelerated && n >= 8)
            {
                ulong orBits = 0;
                for (; i + 8 <= n; i += 8)
                {
                    ulong eight = *(ulong*)(src + i);
                    orBits |= eight;
                    // 8 bytes → 8 code points: two widening steps (byte → ushort → uint), lane order kept.
                    var v16 = Vector128.WidenLower(Vector128.CreateScalarUnsafe(eight).AsByte());
                    Vector256.WidenLower(v16.ToVector256Unsafe()).Store(dst + i);
                }
                acc = (orBits & 0x8080808080808080UL) != 0 ? 0x80u : 0u;
            }
            for (; i < n; i++)
            {
                byte b = src[i];
                acc |= b;
                dst[i] = b;
            }
            return latin1 || acc < 0x80;
        }

        /// <summary>Appends the code points of a valid UTF-8 text (a StringDType cell) to a code-point output.</summary>
        /// <param name="src">The UTF-8 bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="points">The output.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void AppendUtf8Points(byte* src, long size, NDCodecSink points)
        {
            points.Reserve(size);
            uint* dst = (uint*)points.Data + points.Count;
            long n = 0, i = 0;
            while (i < size)
            {
                byte b = src[i];
                if (b < 0x80)
                {
                    dst[n++] = b;
                    i++;
                    continue;
                }
                var span = new ReadOnlySpan<byte>(src + i, (int)Math.Min(size - i, 4));
                System.Text.Rune.DecodeFromUtf8(span, out var rune, out int consumed);
                dst[n++] = (uint)rune.Value;
                i += consumed;
            }
            points.Count += n;
        }

        /// <summary>Appends a .NET string's code points (a surrogate pair one, a lone surrogate its own) to a code-point output.</summary>
        /// <param name="s">The string.</param>
        /// <param name="points">The output.</param>
        private static void AppendStringPoints(string s, NDCodecSink points)
        {
            points.Reserve(s.Length);
            uint* dst = (uint*)points.Data + points.Count;
            long n = 0;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    dst[n++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                    i++;
                }
                else
                {
                    dst[n++] = c;
                }
            }
            points.Count += n;
        }
    }
}
