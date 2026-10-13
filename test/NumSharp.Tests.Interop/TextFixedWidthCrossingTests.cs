using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     numpy's fixed-width text dtypes <c>S</c> (bytes) and <c>U</c> (UCS-4 code points) crossing the bridge — strings
    ///     phase 1, work package P1.8. They are ordinary buffers: NumSharp's <c>S&lt;n&gt;</c> / <c>U&lt;n&gt;</c> hold the
    ///     same bytes numpy's do (NUL-padded, native-endian code points), so both directions are zero-copy VIEWS with shared
    ///     writes, for every layout numpy and NumSharp can describe; only what no view can express falls to a copy
    ///     (big-endian code points are byte-swapped; a stride that is not a whole number of elements is linearized).
    /// </summary>
    /// <remarks>
    ///     Every comparison is against LIVE numpy over the same memory: the canonical C-order bytes on both sides (the
    ///     strongest equality — it sees the NUL padding, the code-point width and the byte order), plus element values and
    ///     the flags. Test data covers ASCII, BMP and astral text, embedded and trailing NULs, empty and full-width elements,
    ///     bytes that are not UTF-8, and UCS-4 units no .NET string can hold.
    /// </remarks>
    [TestClass]
    public class TextFixedWidthCrossingTests : InteropTestBase
    {
        /// <summary>The text elements the U layouts are built from (24 of them, so 2-D layouts are 4 × 6).</summary>
        private static readonly string[] UnicodeValues = BuildUnicodeValues();

        /// <summary>
        ///     24 texts of at most five code points: ASCII, BMP (Hebrew, CJK), astral (an emoji, U+10FFFF), an embedded NUL,
        ///     the empty string and full-width elements. Built from code points so no tool or file encoding can alter them.
        /// </summary>
        /// <returns>The texts.</returns>
        private static string[] BuildUnicodeValues()
        {
            string aleph = ((char)0x05D0).ToString();
            string han = ((char)0x6F22).ToString();
            string grin = char.ConvertFromUtf32(0x1F600);
            string last = char.ConvertFromUtf32(0x10FFFF);
            var values = new List<string>
            {
                "a", "bc", "def", "ghij", "klmno", "",
                aleph, aleph + han, grin, grin + "x", "x" + grin + "y", last,
                "a\0b", "\0z", "pq", "rstuv", han + han + han, grin + grin,
                "w", "", "xyz", aleph + "1", "end", "12345",
            };
            return values.ToArray();
        }

        /// <summary>The 24 base NumSharp U5 array (fresh, C-contiguous).</summary>
        /// <returns>The array.</returns>
        private static NDArray UnicodeBase() => np.array(UnicodeValues).astype(np.dtype("U5"));

        /// <summary>The 24 base NumSharp S5 array: bytes that include 0x80 / 0xFF (not UTF-8) and embedded NULs.</summary>
        /// <returns>The array.</returns>
        private static NDArray BytesBase()
        {
            var nd = np.zeros(new Shape(24), np.dtype("S5"));
            for (int i = 0; i < 24; i++)
            {
                // Distinct per element: a NUL in the middle for some, high bytes for others, full width for the rest.
                byte[] value = (i % 4) switch
                {
                    0 => new[] { (byte)('a' + i % 26) },
                    1 => new byte[] { 0x80, (byte)i, 0xFF },
                    2 => new byte[] { (byte)'x', 0x00, (byte)'y' },
                    _ => new[] { (byte)'0', (byte)'1', (byte)'2', (byte)'3', (byte)('4' + i % 6) },
                };
                nd.SetValue(value, i);
            }

            return nd;
        }

        /// <summary>The NumSharp layouts every export is checked over, built from a 24-element base.</summary>
        /// <param name="b">The base array (any fixed-width text dtype).</param>
        /// <returns>(name, view) pairs: C, F, stepped, reversed, transposed, broadcast, offset, 0-d, empty and one element.</returns>
        private static IEnumerable<(string Name, NDArray View)> NumSharpLayouts(NDArray b)
        {
            yield return ("C 2-D", b.reshape(4, 6));
            yield return ("F 2-D", np.asfortranarray(b.reshape(4, 6)));
            yield return ("stepped [::3]", b["::3"]);
            yield return ("reversed [::-1]", b["::-1"]);
            yield return ("transposed", b.reshape(4, 6).T);
            yield return ("reversed rows", b.reshape(4, 6)["::-1, 1:5"]);
            yield return ("broadcast", np.broadcast_to(b["0:6"], new Shape(3, 6)));
            yield return ("offset (split child)", np.split(b, 4)[2]);
            yield return ("0-d", b["5"]);
            yield return ("empty view in a buffer", b["3:3"]);
            yield return ("one element", b["7:8"]);
        }

        /// <summary>
        ///     NumSharp → numpy, every layout, S and U of two widths: the export is a VIEW numpy reads with the same dtype,
        ///     shape, byte strides, writeability and C-order bytes, and a write on either side is seen on the other.
        /// </summary>
        [TestMethod]
        public void Export_EveryLayout_IsAViewWithTheSameBytes()
        {
            foreach (string kind in new[] { "U5", "U1", "S5", "S1" })
            {
                NDArray b = kind[0] == 'U' ? UnicodeBase() : BytesBase();
                if (kind.EndsWith("1"))
                    b = b.astype(np.dtype(kind));   // truncates each element to its first character / byte
                foreach (var (name, view) in NumSharpLayouts(b))
                {
                    string cell = $"{kind} / {name}";
                    int itemsize = (int)view.dtype.itemsize;
                    using (Gil())
                    {
                        using PyObject x = view.ToNumpy();
                        x.dtype_str().Should().Be(view.dtype.str, cell);
                        x.shape_dims().Should().Equal(view.Shape.Dimensions, cell);
                        x.writeable().Should().Be(view.Shape.IsWriteable, cell);
                        if (view.size > 1)
                            x.strides_bytes().Should().Equal(view.Shape.Strides.Select(s => s * itemsize), cell);
                        ByteContract.AssertSameBytes(view, x, cell);
                        if (view.size == 0 || !view.Shape.IsWriteable)
                            continue;

                        // numpy writes element 0 of the view; NumSharp reads it at the same coordinates (and back).
                        long[] origin = new long[view.ndim];
                        Scope.Set("x", x);
                        string literal = kind[0] == 'U' ? "'Q'" : "b'Q'";
                        PyExec(view.ndim == 0 ? $"x[()] = {literal}" : $"x[{string.Join(",", origin)}] = {literal}");
                        object read = view.GetValue(origin);
                        if (kind[0] == 'U')
                            read.Should().Be("Q", cell);
                        else
                            ((byte[])read).Should().Equal(new[] { (byte)'Q' }, cell);
                        if (kind[0] == 'U')
                            view.SetValue("R", origin);
                        else
                            view.SetValue(new[] { (byte)'R' }, origin);
                        PyStr(view.ndim == 0 ? "x[()]" : $"x[{string.Join(",", origin)}]").Should().Be(kind[0] == 'U' ? "R" : "b'R'", cell);
                        PyExec("del x");
                    }
                }
            }
        }

        /// <summary>
        ///     <see cref="NDArrayPythonInterop.ToNumpyCopy"/> of every layout is an independent numpy array with the same
        ///     bytes and dtype: it owns writeable memory in the order numpy's own copy of the view picks (C, or F for the F
        ///     and transposed layouts), and a write into it leaves the NumSharp source unchanged.
        /// </summary>
        [TestMethod]
        public void ExportCopy_EveryLayout_IsIndependentWithTheSameBytes()
        {
            foreach (NDArray b in new[] { UnicodeBase(), BytesBase() })
            {
                bool unicode = b.dtype.kind == 'U';
                foreach (var (name, view) in NumSharpLayouts(b))
                {
                    using (Gil())
                    {
                        using PyObject x = view.ToNumpyCopy();
                        x.dtype_str().Should().Be(view.dtype.str, name);
                        ByteContract.AssertSameBytes(view, x, name);
                        using PyObject owns = Python.np.with(
                            "a.flags.owndata and a.flags.writeable and (a.flags.c_contiguous or a.flags.f_contiguous)", ("a", x));
                        owns.IsTrue().Should().BeTrue($"{name}: a copy owns contiguous, writeable memory");
                        if (view.size == 0)
                            continue;

                        // numpy writes element 0 of the copy; the NumSharp source keeps its value.
                        long[] origin = new long[view.ndim];
                        object before = view.GetValue(origin);
                        Scope.Set("x", x);
                        string at = view.ndim == 0 ? "x[()]" : $"x[{string.Join(",", origin)}]";
                        PyExec($"{at} = {(unicode ? "'Q'" : "b'Q'")}");
                        object after = view.GetValue(origin);
                        if (unicode)
                            after.Should().Be(before, $"{name}: the copy is independent");
                        else
                            ((byte[])after).Should().Equal((byte[])before, $"{name}: the copy is independent");
                        PyExec("del x");
                    }
                }
            }
        }

        /// <summary>The numpy layouts every import is checked over (bound to <c>src</c> in the test scope first).</summary>
        private static readonly (string Name, string Expr)[] NumpyLayouts =
        {
            ("C 2-D", "src.reshape(4, 6)"),
            ("F 2-D", "np.asfortranarray(src.reshape(4, 6))"),
            ("stepped [::3]", "src[::3]"),
            ("reversed [::-1]", "src[::-1]"),
            ("transposed", "src.reshape(4, 6).T"),
            ("reversed rows", "src.reshape(4, 6)[::-1, 1:5]"),
            ("broadcast", "np.broadcast_to(src[:6], (3, 6))"),
            ("offset slice", "src[13:]"),
            ("0-d", "src[5, ...]"),
            ("empty slice", "src[3:3]"),
            ("one element", "src[7:8]"),
            ("read-only", "ro"),
        };

        /// <summary>
        ///     numpy → NumSharp, every numpy layout (the read-only and broadcast ones included), S and U: the view has the
        ///     same dtype, shape and bytes, the same writeability, and shares writes both ways; the copy has the same bytes
        ///     and is independent.
        /// </summary>
        /// <remarks>
        ///     <c>src</c> is rebuilt from the pristine <c>src0</c> before every layout: the write-through check writes into
        ///     <c>src</c>'s memory, and a later layout that reads the written element would otherwise start from changed
        ///     data. The copy is held to the element's value BEFORE numpy's write, read from the view at the same moment the
        ///     copy was made.
        /// </remarks>
        [TestMethod]
        public void Import_EveryLayout_ViewsAndCopiesWithTheSameBytes()
        {
            foreach (string kind in new[] { "U5", "U1", "S5", "S1" })
            {
                NDArray b = kind[0] == 'U' ? UnicodeBase() : BytesBase();
                if (kind.EndsWith("1"))
                    b = b.astype(np.dtype(kind));
                ExportCopyTo("src0", b);   // numpy-born memory: an owning numpy copy of the base
                foreach (var (name, expr) in NumpyLayouts)
                {
                    string cell = $"{kind} / {name}";
                    PyExec("src = src0.copy(); ro = src0.copy(); ro.flags.writeable = False");
                    PyExec($"v = {expr}");
                    bool numpyWriteable = PyBool("v.flags.writeable");
                    using (NDArray view = ViewOf("v", allowReadonly: true))
                    using (NDArray copy = ImportOf("v"))
                    {
                        view.dtype.Should().Be(np.dtype(kind), cell);
                        view.Shape.IsWriteable.Should().Be(numpyWriteable, cell);
                        copy.dtype.Should().Be(np.dtype(kind), cell);
                        using (Gil())
                        {
                            using PyObject v = Scope.Get("v");
                            view.Shape.Dimensions.Should().Equal(v.shape_dims(), cell);
                            ByteContract.AssertSameBytes(view, v, cell);
                            ByteContract.AssertSameBytes(copy, v, cell);
                        }

                        if (view.size == 0 || !numpyWriteable)
                            continue;
                        long[] origin = new long[view.ndim];
                        string original = ReadText(view, origin, kind);
                        string at = view.ndim == 0 ? "v[()]" : $"v[{string.Join(",", origin)}]";
                        PyExec(kind[0] == 'U' ? $"{at} = 'Z'" : $"{at} = b'Z'");
                        ReadText(view, origin, kind).Should().Be("Z", $"{cell}: numpy's write is visible through the view");
                        ReadText(copy, origin, kind).Should().Be(original, $"{cell}: the copy is independent");
                    }
                }

                PyExec("del src0, src, ro, v");
            }
        }

        /// <summary>One fixed-width text element as a string (a U element as its text, an S element as its ASCII bytes).</summary>
        /// <param name="nd">The array.</param>
        /// <param name="coords">The element's coordinates.</param>
        /// <param name="kind">The dtype spelling (<c>U…</c> or <c>S…</c>).</param>
        /// <returns>The element.</returns>
        private static string ReadText(NDArray nd, long[] coords, string kind)
            => kind[0] == 'U' ? (string)nd.GetValue(coords) : System.Text.Encoding.Latin1.GetString((byte[])nd.GetValue(coords));

        /// <summary>
        ///     The element values agree with numpy's own reading: an embedded NUL is kept, TRAILING NULs are padding (numpy
        ///     strips them; NumSharp reads the same), an empty element is <c>''</c>, a full-width element has no terminator.
        /// </summary>
        [TestMethod]
        public void Values_EmbeddedAndTrailingNuls_FullWidth_MatchNumpy()
        {
            PyExec("u = np.array(['a\\x00b', 'c\\x00', '\\x00', '', 'xyz'], dtype='U3')\n" +
                   "s = np.array([b'a\\x00b', b'c\\x00', b'\\x00', b'', b'xyz'], dtype='S3')");
            using NDArray u = ViewOf("u");
            using NDArray s = ViewOf("s");
            string[] expected = { "a\0b", "c", "", "", "xyz" };
            for (int i = 0; i < expected.Length; i++)
            {
                u.GetValue(i).Should().Be(expected[i], $"U element {i}");
                PyStr($"u[{i}] == {ToPythonLiteral(expected[i])}").Should().Be("True", $"numpy reads U element {i} the same");
                ((byte[])s.GetValue(i)).Should().Equal(System.Text.Encoding.ASCII.GetBytes(expected[i]), $"S element {i}");
            }
        }

        /// <summary>A Python string literal for an ASCII text that may hold NUL characters.</summary>
        /// <param name="text">The text.</param>
        /// <returns>The literal, NULs escaped.</returns>
        private static string ToPythonLiteral(string text) => "'" + text.Replace("\0", "\\x00") + "'";

        /// <summary>
        ///     Big-endian UCS-4 text (aligned and at an odd address) cannot be a native-endian view: the view refuses with the
        ///     copy route in the message, and the copy byte-swaps every code point into a native <c>U</c> array — astral code
        ///     points included.
        /// </summary>
        [TestMethod]
        public void BigEndianText_CopiesByteSwapped_ViewRefused()
        {
            PyExec("be = np.array(['ab', '\\U0001F600', 'xyz'], dtype='>U3')\n" +
                   "le = be.astype('<U3')\n" +
                   "beu = np.frombuffer(bytearray(b'\\x00' + be.tobytes()), dtype='>U3', offset=1)");
            foreach (string name in new[] { "be", "beu" })
            {
                ((Action)(() => ViewOf(name, allowReadonly: true).Dispose()))
                    .Should().Throw<NotSupportedException>().WithMessage("*big-endian UCS-4*byte-swaps*U3*", name);
                using NDArray copy = ImportOf(name);
                copy.dtype.Should().Be(np.dtype("U3"), name);
                copy.GetValue(1).Should().Be(char.ConvertFromUtf32(0x1F600), name);
                using (Gil())
                {
                    using PyObject le = Scope.Get("le");
                    ByteContract.AssertSameBytes(copy, le, $"{name}: every code point byte-reversed");
                }
            }
        }

        /// <summary>
        ///     A U array at an odd address is still a view (x86 / arm64 read unaligned code points), and NumSharp reports it
        ///     unaligned exactly as numpy does; a stride that is not a whole number of elements (a structured field) falls to
        ///     a linearized copy.
        /// </summary>
        [TestMethod]
        public void UnalignedAndFieldStrides_ViewWhenTheyCan_CopyWhenTheyCannot()
        {
            PyExec("un = np.frombuffer(bytearray(1 + 3 * 8), dtype='U2', offset=1)\n" +
                   "un[...] = ['ab', 'c', '\\U0001F600']\n" +
                   "rec = np.zeros(3, dtype=[('a', 'U2'), ('b', 'i4')]); rec['a'] = ['ab', 'c', 'd']\n" +
                   "fld = rec['a']");
            PyBool("un.flags.aligned").Should().BeFalse("numpy's U alignment is 4 bytes");
            using (NDArray un = ViewOf("un"))
            {
                un.flags.aligned.Should().BeFalse("NumSharp computes alignment from the address, like numpy");
                un.GetValue(2).Should().Be(char.ConvertFromUtf32(0x1F600));
                un.SetValue("zz", 0);
                PyStr("un[0]").Should().Be("zz", "the unaligned view shares memory");
            }

            ((Action)(() => ViewOf("fld").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*stride 12 bytes is not a multiple of itemsize 8*");
            using (NDArray copy = ImportOf("fld"))
            {
                copy.dtype.Should().Be(np.dtype("U2"));
                copy.GetValue(0).Should().Be("ab");
                copy.GetValue(2).Should().Be("d");
            }
        }

        /// <summary>
        ///     <see cref="NDArrayPythonInterop.ToMemoryView"/> of a C-contiguous S / U array is its raw bytes (format 'B'), the
        ///     bytes numpy's own <c>tobytes()</c> gives; a <c>StringDType</c> array has no bytes that are its text, so it is
        ///     refused with numpy's own words.
        /// </summary>
        [TestMethod]
        public void MemoryView_OfFixedWidthText_IsItsRawBytes_StringDTypeRefused()
        {
            NDArray u = np.array(new[] { "ab", "c" }).astype(np.dtype("U2"));
            ExportMemoryViewTo("mu", u);
            ExportCopyTo("cu", u);
            PyBool("bytes(mu) == cu.tobytes()").Should().BeTrue("the memoryview is the UCS-4 bytes");
            PyStr("mu.format").Should().Be("B");
            PyExec("del mu");

            NDArray t = np.array(new object[] { "a", "b" }, np.dtypes.StringDType());
            ((Action)(() => NDArrayPythonInterop.ToMemoryView(t)))
                .Should().Throw<NotSupportedException>().WithMessage("*cannot include dtype 'numpy.dtypes.StringDType' in a buffer*");
        }

        /// <summary>
        ///     numpy's text scalars are 0-d text to the bridge, as to <c>np.asarray</c>: an <c>np.str_</c> views its own 0-d
        ///     UCS-4 buffer (read-only), an <c>np.bytes_</c> is the 0-d <c>S&lt;n&gt;</c> it is (not its n bytes); an EMPTY one
        ///     has no byte to share, so it copies as numpy reads it, a one-character empty element.
        /// </summary>
        [TestMethod]
        public void TextScalars_AreZeroDText_EmptyOnesCopy()
        {
            PyExec("sc = np.str_('h\\u00e9llo')\nbc = np.bytes_(b'a\\x80')\nes = np.str_('')\neb = np.bytes_(b'')");
            using (NDArray sc = ViewOf("sc", allowReadonly: true))
            {
                sc.ndim.Should().Be(0);
                sc.dtype.Should().Be(np.dtype("U5"));
                sc.GetValue().Should().Be("h" + (char)0xE9 + "llo");
                sc.Shape.IsWriteable.Should().BeFalse("a numpy scalar is immutable");
            }

            using (NDArray bc = ViewOf("bc", allowReadonly: true))
            {
                bc.ndim.Should().Be(0);
                bc.dtype.Should().Be(np.dtype("S2"), "np.asarray(np.bytes_(b'a\\x80')) is a 0-d |S2");
                ((byte[])bc.GetValue()).Should().Equal(new byte[] { (byte)'a', 0x80 });
            }

            using (NDArray bcCopy = ImportOf("bc"))
                bcCopy.dtype.Should().Be(np.dtype("S2"), "the copy path reads np.bytes_ the same way");
            using (NDArray plainBytes = ImportOf("b'a\\x80'"))
                plainBytes.dtype.Should().Be(np.dtype("u1"), "a plain bytes object stays raw bytes");

            ((Action)(() => ViewOf("es", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*zero-width*");
            using (NDArray es = ImportOf("es"))
            {
                es.dtype.Should().Be(np.dtype("U1"), "np.asarray(np.str_('')) is '<U1'");
                es.GetValue().Should().Be("");
            }

            using (NDArray eb = ImportOf("eb"))
            {
                eb.dtype.Should().Be(np.dtype("S1"));
                ((byte[])eb.GetValue()).Should().BeEmpty();
            }
        }

        /// <summary>
        ///     The codec's three modes over fixed-width text: Auto and View decode a numpy U / S array as a VIEW (shared
        ///     writes), Copy as an independent copy; encode is a view in Auto / View and a copy in Copy.
        /// </summary>
        [TestMethod]
        public void Codec_Modes_FixedWidthText()
        {
            var auto = new NumpyCodec();
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View, EncodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy, EncodeMode = NumpyCodecMode.Copy });
            PyExec("cu = np.array(['ab', 'c'], dtype='U2')");
            using (Gil())
            {
                using PyObject cu = Scope.Get("cu");
                foreach (var (codec, shares) in new[] { (auto, true), (viewOnly, true), (copyOnly, false) })
                {
                    codec.TryDecode(cu, out NDArray nd).Should().BeTrue();
                    using (nd)
                    {
                        nd.dtype.Should().Be(np.dtype("U2"));
                        Scope.Exec("cu[1] = 'q'");
                        nd.GetValue(1).Should().Be(shares ? "q" : "c");
                        Scope.Exec("cu[1] = 'c'");
                    }
                }

                // An encoder's result is handed to pythonnet, which never disposes it; EncoderHandoff disposes it at the
                // NEXT encode on the thread. So each encoded array is used before the next encode.
                NDArray ns = np.array(new[] { "xy", "z" }).astype(np.dtype("U2"));
                using (PyObject viewEncoded = auto.TryEncode(ns))
                {
                    ns.SetValue("w", 0);
                    using PyObject first = viewEncoded.InvokeMethod("__getitem__", new PyInt(0));
                    first.ToString().Should().Be("w", "Auto encodes a view");
                }

                using (PyObject copyEncoded = copyOnly.TryEncode(ns))
                {
                    ns.SetValue("v", 0);
                    using PyObject firstCopy = copyEncoded.InvokeMethod("__getitem__", new PyInt(0));
                    firstCopy.ToString().Should().Be("w", "Copy encodes an independent copy");
                }
            }
        }

        /// <summary>PyTorch has no string tensors: <see cref="TorchInterop.ToTorch"/> refuses text before any export.</summary>
        [TestMethod]
        public void ToTorch_RefusesText_BeforeAnyExport()
        {
            int exports = NDArrayPythonInterop.LiveExports;
            foreach (NDArray text in new[]
                     {
                         np.array(new[] { "a" }), np.array(new[] { "a" }).astype(np.dtype("S1")),
                         np.array(new object[] { "a" }, np.dtypes.StringDType()),
                     })
            {
                ((Action)(() => text.ToTorch()))
                    .Should().Throw<NotSupportedException>().WithMessage("*PyTorch has no string tensors*");
            }

            NDArrayPythonInterop.LiveExports.Should().Be(exports, "nothing was exported");
        }
    }
}
