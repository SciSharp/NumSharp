using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Strings;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     NEP 55 <c>StringDType</c> (<c>T</c>) arrays crossing the bridge — strings phase 1, work package P1.8, decision D7:
    ///     an exact COPY in both directions through numpy's own <c>NpyString_*</c> C API. numpy keeps each element's text in
    ///     an allocator owned by the array's descriptor, outside the buffer, so no view is possible; the copy duplicates
    ///     every element with <c>NpyString_dup</c>'s rules (an NA stays NA, an empty string stays empty, a short string moves
    ///     as its 16-byte cell, a longer one is loaded from one allocator and packed into the other).
    /// </summary>
    /// <remarks>
    ///     <para>Each crossing is held to three equalities against LIVE numpy: the DESCRIPTOR (<c>na_object</c> and
    ///     <c>coerce</c>), the element VALUES with NA identity (NA vs a text equal to the NA object's <c>str()</c>), and the
    ///     CELLS — <c>tobytes()</c> of the copy equals numpy's own copy's (an import) or NumSharp's own C-order copy's (an
    ///     export), which pins every arena offset and flag, so the cells are the ones the side's own copy builds.</para>
    ///     <para>The source data holds short (≤ 15-byte), medium (arena, 1-byte prefix) and long (arena, 8-byte prefix)
    ///     strings, a heap string (a later, longer rewrite), NA cells, an initialized empty string, an embedded NUL and
    ///     non-BMP text.</para>
    /// </remarks>
    [TestClass]
    public class TextStringDTypeCrossingTests : InteropTestBase
    {
        /// <summary>The Python helpers of this class: the value encoding and numpy's NA kinds.</summary>
        private const string Helpers =
            "import math\n" +
            "from numpy.dtypes import StringDType\n" +
            "def _enc_one(x):\n" +
            "    if x is None: return 'N:None'\n" +
            "    if isinstance(x, (float, np.floating)) and x != x: return 'N:nan'\n" +
            "    if isinstance(x, str): return 's:' + x.encode('utf-8').hex()\n" +
            "    return 'o:' + repr(x)\n" +
            "def _enc(a):\n" +
            "    return [_enc_one(x) for x in (a.ravel().tolist() if a.ndim else [a.item()])]\n";

        /// <summary>
        ///     The missing-data kinds both sides agree on: numpy's spelling and the NumSharp descriptor it maps to (decision
        ///     D5: <c>null</c> = None, a float NaN = nan, a string = a string NA, numbers / bool / bytes as Python numbers).
        /// </summary>
        /// <returns>(name, numpy <c>na_object=</c> argument, NumSharp descriptor for a <c>coerce</c> value).</returns>
        private static IEnumerable<(string Name, string NumpyArgs, Func<bool, DType> Descr)> NAKinds()
        {
            yield return ("absent", "", c => np.dtypes.StringDType(c));
            yield return ("None", "na_object=None", c => np.dtypes.StringDType((object)null, c));
            yield return ("nan", "na_object=float('nan')", c => np.dtypes.StringDType(double.NaN, c));
            yield return ("np.float64 nan", "na_object=np.float64('nan')", c => np.dtypes.StringDType(double.NaN, c));
            yield return ("'NA'", "na_object='NA'", c => np.dtypes.StringDType("NA", c));
            yield return ("''", "na_object=''", c => np.dtypes.StringDType("", c));
            yield return ("int 5", "na_object=5", c => np.dtypes.StringDType(5L, c));
            yield return ("True", "na_object=True", c => np.dtypes.StringDType(na_object: true, coerce: c));
            yield return ("1.5", "na_object=1.5", c => np.dtypes.StringDType(1.5, c));
            yield return ("b'x'", "na_object=b'x'", c => np.dtypes.StringDType(new[] { (byte)'x' }, c));
            yield return ("complex", "na_object=complex(1, 2)", c => np.dtypes.StringDType(new Complex(1, 2), c));
            yield return ("np.float16 nan", "na_object=np.float16('nan')", c => np.dtypes.StringDType(Half.NaN, c));
        }

        /// <summary>The numpy expression of a <c>StringDType</c> with the given parameters.</summary>
        /// <param name="numpyArgs">The <c>na_object=...</c> argument, or empty.</param>
        /// <param name="coerce">numpy's <c>coerce</c>.</param>
        /// <returns><c>StringDType(...)</c>.</returns>
        private static string NumpyDType(string numpyArgs, bool coerce)
            => "StringDType(" + string.Join(", ", new[] { numpyArgs, coerce ? "" : "coerce=False" }.Where(s => s.Length > 0)) + ")";

        /// <summary>
        ///     The NumSharp side of <c>_enc</c>: each element in C order as <c>N:None</c> / <c>N:nan</c> (NA objects), <c>s:</c> +
        ///     the UTF-8 hex of a text, or <c>o:</c> + Python's repr of any other NA object.
        /// </summary>
        /// <param name="nd">A NumSharp <c>StringDType</c> array of any layout.</param>
        /// <returns>The encoded values.</returns>
        private static List<string> Encode(NDArray nd)
        {
            var result = new List<string>();
            if (nd.size == 0)
                return result;
            NDArray flat = nd.ndim == 0 ? nd.reshape(1) : nd.flatten();
            for (long i = 0; i < flat.size; i++)
            {
                object v = flat.GetValue(i);
                result.Add(v switch
                {
                    null => "N:None",
                    double d when double.IsNaN(d) => "N:nan",
                    Half h when Half.IsNaN(h) => "N:nan",
                    string s => "s:" + Convert.ToHexString(Encoding.UTF8.GetBytes(s)).ToLowerInvariant(),
                    _ => "o:" + NDPyObject.Repr(v),
                });
            }

            return result;
        }

        /// <summary>The numpy side's encoded values of <paramref name="expr"/> (the class helpers must be defined).</summary>
        /// <param name="expr">A numpy expression in the test scope.</param>
        /// <returns>The encoded values.</returns>
        private List<string> NumpyEncode(string expr)
        {
            using (Gil())
            {
                using PyObject list = Scope.Eval($"_enc({expr})");
                using var tuple = new PyList(list);
                var result = new List<string>();
                for (int i = 0; i < tuple.Length(); i++)
                {
                    using PyObject item = tuple[i];
                    result.Add(item.As<string>());
                }

                return result;
            }
        }

        /// <summary>The raw bytes a numpy expression's <c>tobytes()</c> gives (a <c>StringDType</c> array's cells).</summary>
        /// <param name="expr">A numpy expression in the test scope.</param>
        /// <returns>The bytes.</returns>
        private byte[] NumpyBytes(string expr)
        {
            using (Gil())
            {
                using PyObject b = Scope.Eval($"({expr}).tobytes()");
                return b.As<byte[]>();
            }
        }

        /// <summary>
        ///     Builds <c>src</c> in the test scope: 24 numpy <c>StringDType</c> elements with every cell kind — short, medium,
        ///     long, heap (a later longer rewrite), an initialized empty string, an embedded NUL, non-BMP text — and NA cells
        ///     when the descriptor has an NA object.
        /// </summary>
        /// <param name="numpyDType">The numpy descriptor expression.</param>
        /// <param name="hasNA">Whether NA cells can be written.</param>
        private void BuildNumpySource(string numpyDType, bool hasNA)
        {
            PyExec(
                $"_dt = {numpyDType}\n" +
                "_vals = ['s', 'short15charsabc', 'a medium arena string over 15 bytes', 'L' * 300, '',\n" +
                "         'e\\u0301\\u6f22\\U0001F600', 'nul\\x00inside', 'end\\x00'] * 3\n" +
                "src = np.array(_vals, dtype=_dt)\n" +
                "src[3] = 'q' * 1000\n" +           // a rewrite LONGER than its arena slot moves the text to the heap
                "src[12] = 'r' * 40\n" +
                (hasNA ? "src[4] = _dt.na_object\nsrc[17] = _dt.na_object\n" : ""));
        }

        /// <summary>The numpy layouts of <c>src</c> every import is checked over.</summary>
        private static readonly (string Name, string Expr)[] NumpyLayouts =
        {
            ("C 1-D", "src"),
            ("C 2-D", "src.reshape(4, 6)"),
            ("F 2-D", "np.asfortranarray(src.reshape(4, 6))"),
            ("transposed", "src.reshape(4, 6).T"),
            ("reversed", "src[::-1]"),
            ("stepped rows", "src.reshape(4, 6)[::-1, ::2]"),
            ("broadcast", "np.broadcast_to(src[:6], (3, 6))"),
            ("offset slice", "src[13:]"),
            ("0-d", "src[3, ...]"),
            ("empty", "src[5:5]"),
            ("one element", "src[17:18]"),
        };

        /// <summary>
        ///     numpy → NumSharp for every NA kind × both <c>coerce</c> values × every layout: the copy has the same descriptor,
        ///     the same values (NA identity included) and the same cells numpy's own C-order copy builds; a view is refused
        ///     with the copy route.
        /// </summary>
        [TestMethod]
        public void Import_EveryNAKind_EveryLayout_SameDescriptorValuesAndCells()
        {
            PyExec(Helpers);
            foreach (var (naName, numpyArgs, descr) in NAKinds())
            foreach (bool coerce in new[] { true, false })
            {
                bool hasNA = naName != "absent";
                BuildNumpySource(NumpyDType(numpyArgs, coerce), hasNA);
                DType expected = descr(coerce);
                foreach (var (layout, expr) in NumpyLayouts)
                {
                    string cell = $"{naName} / coerce={coerce} / {layout}";
                    PyExec($"v = {expr}");
                    using NDArray copy = ImportOf("v");
                    // numpy's na_eq_cmp is not reflexive for an np.float16 NaN (two such dtypes are unequal even in pure
                    // numpy) and NumSharp ports it: the repr pins the parameters for every kind, equality where it holds.
                    copy.dtype.ToString().Should().Be(expected.ToString(), cell);
                    copy.dtype.HasNAObject.Should().Be(hasNA, cell);
                    if (naName != "np.float16 nan")
                        copy.dtype.Should().Be(expected, cell);
                    copy.Shape.Dimensions.Should().Equal(PyDims("v"), cell);
                    Encode(copy).Should().Equal(NumpyEncode("v"), cell);
                    copy.tobytes().Should().Equal(NumpyBytes("np.array(v, order='C', copy=True)"), $"{cell}: the cells of numpy's own copy");
                    ((Action)(() => ViewOf("v", allowReadonly: true).Dispose()))
                        .Should().Throw<NotSupportedException>().WithMessage("*has no buffer form*copy*", cell);
                }
            }
        }

        /// <summary>The shape of a numpy expression.</summary>
        /// <param name="expr">The expression.</param>
        /// <returns>The dimensions.</returns>
        private long[] PyDims(string expr)
        {
            using (Gil())
            {
                using PyObject a = Scope.Eval(expr);
                return a.shape_dims();
            }
        }

        /// <summary>
        ///     The 24 NumSharp source elements matching <see cref="BuildNumpySource"/>: every cell kind, NA cells at 4 and 17
        ///     when the descriptor has an NA object, and a heap string at 3 (a rewrite longer than the arena slot).
        /// </summary>
        /// <param name="descr">The descriptor.</param>
        /// <returns>The array.</returns>
        /// <remarks>
        ///     The NA cells are written with the descriptor's OWN <c>na_object</c>, as the numpy twin writes
        ///     <c>_dt.na_object</c>: a store is NA only when <c>na_eq_cmp</c> calls the value equal to that object, and for
        ///     an <c>np.float16</c> NaN only identity does (it is not a Python float, so the NaN rule does not apply). A
        ///     second boxed <see cref="Half.NaN"/> is therefore the TEXT <c>'nan'</c> under <c>coerce=True</c> and numpy's
        ///     <c>ValueError</c> under <c>coerce=False</c>, in both libraries.
        /// </remarks>
        private static NDArray NumSharpSource(DType descr)
        {
            string[] pattern =
            {
                "s", "short15charsabc", "a medium arena string over 15 bytes", new string('L', 300), "",
                "e" + (char)0x0301 + (char)0x6F22 + char.ConvertFromUtf32(0x1F600), "nul\0inside", "end\0",
            };
            var values = new object[24];
            for (int i = 0; i < 24; i++)
                values[i] = pattern[i % pattern.Length];
            NDArray nd = np.array(values, descr);
            nd.SetValue(new string('q', 1000), 3);
            nd.SetValue(new string('r', 40), 12);
            if (descr.HasNAObject)
            {
                nd.SetValue(descr.na_object, 4);
                nd.SetValue(descr.na_object, 17);
            }

            return nd;
        }

        /// <summary>The NumSharp layouts every export is checked over.</summary>
        /// <param name="b">The 24-element source.</param>
        /// <returns>(name, view) pairs.</returns>
        private static IEnumerable<(string Name, NDArray View)> NumSharpLayouts(NDArray b)
        {
            yield return ("C 1-D", b);
            yield return ("C 2-D", b.reshape(4, 6));
            yield return ("transposed", b.reshape(4, 6).T);
            yield return ("reversed", b["::-1"]);
            yield return ("stepped rows", b.reshape(4, 6)["::-1, ::2"]);
            yield return ("broadcast", np.broadcast_to(b["0:6"], new Shape(3, 6)));
            yield return ("offset (split child)", np.split(b, 4)[2]);
            yield return ("0-d", b["3"]);
            yield return ("empty", b["5:5"]);
            yield return ("one element", b["17:18"]);
        }

        /// <summary>
        ///     NumSharp → numpy for every NA kind × both <c>coerce</c> values × every layout: the numpy copy has the same
        ///     descriptor and values, its cells are the ones NumSharp's own C-order copy builds, and the zero-copy verb refuses
        ///     with the copy route.
        /// </summary>
        [TestMethod]
        public void Export_EveryNAKind_EveryLayout_SameDescriptorValuesAndCells()
        {
            PyExec(Helpers);
            foreach (var (naName, numpyArgs, descr) in NAKinds())
            foreach (bool coerce in new[] { true, false })
            {
                NDArray source = NumSharpSource(descr(coerce));
                foreach (var (layout, view) in NumSharpLayouts(source))
                {
                    string cell = $"{naName} / coerce={coerce} / {layout}";
                    using (Gil())
                    {
                        using PyObject x = view.ToNumpyCopy();
                        Scope.Set("x", x);
                    }

                    // The numpy descriptor prints as NumSharp's (an np.float64 NA object comes back as the Python float it
                    // maps to, and numpy calls the two equal); equality where numpy's na_eq_cmp is reflexive.
                    PyStr("repr(x.dtype)").Should().Be(view.dtype.ToString(), cell);
                    if (naName != "np.float16 nan")
                        PyBool($"x.dtype == {NumpyDType(numpyArgs, coerce)}").Should().BeTrue(cell);
                    PyStr("type(x.dtype).__name__").Should().Be("StringDType", cell);
                    PyDims("x").Should().Equal(view.Shape.Dimensions, cell);
                    NumpyEncode("x").Should().Equal(Encode(view), cell);
                    NumpyBytes("x").Should().Equal(np.copy(view, order: 'C').tobytes(), $"{cell}: NumSharp's own C-order copy builds these cells");
                    ((Action)(() => view.ToNumpy()))
                        .Should().Throw<NotSupportedException>().WithMessage("*has no buffer form*ToNumpyCopy*", cell);
                    PyExec("del x");
                }
            }
        }

        /// <summary>
        ///     An export's cells are exactly numpy's own construction from the same values (<c>np.array(list, dtype)</c>):
        ///     numpy's arena grows the same way and every offset lands where numpy puts it.
        /// </summary>
        [TestMethod]
        public void Export_CellsEqualNumpysOwnArrayFromTheSameValues()
        {
            PyExec(Helpers);
            NDArray source = NumSharpSource(np.dtypes.StringDType((object)null));
            using (Gil())
            {
                using PyObject x = source.ToNumpyCopy();
                Scope.Set("x", x);
            }

            PyBool("x.tobytes() == np.array(x.tolist(), dtype=x.dtype).tobytes()").Should().BeTrue();
            PyExec("del x");
        }

        /// <summary>
        ///     NumSharp → numpy → NumSharp returns the same descriptor, values and cells (the round trip of an exact copy).
        /// </summary>
        /// <remarks>
        ///     The descriptor of the round trip holds a NEW NA object: each crossing builds the other side's object from the
        ///     value (a C# <see cref="Half"/> becomes a new <c>np.float16</c> and back). For every kind but the
        ///     <c>np.float16</c> NaN the two descriptors are equal; that one is equal only to itself, because
        ///     <c>na_eq_cmp</c> compares a non-<c>float</c> NA object by identity before <c>==</c> (and NaN != NaN). numpy
        ///     itself keeps the identity only because its copy shares the NA object; the repr pins the parameters.
        /// </remarks>
        [TestMethod]
        public void RoundTrip_NumSharpToNumpyToNumSharp_IsIdentical()
        {
            foreach (var (naName, _, descr) in NAKinds())
            {
                NDArray source = NumSharpSource(descr(true));
                NDArray back;
                using (Gil())
                {
                    using PyObject x = source.ToNumpyCopy();
                    back = x.ToNDArray();
                }

                using (back)
                {
                    back.dtype.ToString().Should().Be(source.dtype.ToString(), naName);
                    if (naName != "np.float16 nan")
                        back.dtype.Should().Be(source.dtype, naName);
                    Encode(back).Should().Equal(Encode(source), naName);
                    back.tobytes().Should().Equal(np.copy(source).tobytes(), naName);
                }
            }
        }

        /// <summary>
        ///     A missing-data object with no C# form (numpy side: a user class, a tuple) and one with no Python form (NumSharp
        ///     side: a decimal, an arbitrary object) refuses the crossing with guidance; nothing is leased or leaked.
        /// </summary>
        [TestMethod]
        public void ForeignNAObjects_AreRefusedBothWays()
        {
            PyExec(Helpers +
                   "class _Other:\n" +
                   "    def __repr__(self): return 'Other()'\n" +
                   "    def __str__(self): return 'other'\n" +
                   "other_na = np.array(['a'], dtype=StringDType(na_object=_Other()))\n" +
                   "tuple_na = np.array(['a'], dtype=StringDType(na_object=(1,)))");
            ((Action)(() => ImportOf("other_na").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*na_object Other() (_Other) has no C# form*");
            ((Action)(() => ImportOf("tuple_na").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*na_object (1,) (tuple) has no C# form*");

            foreach (object na in new object[] { 1.5m, new object(), new[] { 1, 2 } })
            {
                NDArray nd = np.array(new object[] { "a" }, np.dtypes.StringDType(na));
                ((Action)(() => nd.ToNumpyCopy().Dispose()))
                    .Should().Throw<NotSupportedException>().WithMessage("*has no Python form*", na.GetType().Name);
            }
        }

        /// <summary>
        ///     An object array whose every element is a <c>str</c> (or <c>np.str_</c>) — any rank, layout or size — copies as a
        ///     NumSharp <c>StringDType()</c> array (owner's decision 2026-10-13: text that cannot be viewed is copied); any other
        ///     element is refused with the first offender named, a lone surrogate with numpy's UTF-8 error, and a view is
        ///     always refused.
        /// </summary>
        [TestMethod]
        public void ObjectArrays_OfStr_Copy_AnythingElseRefused()
        {
            PyExec(Helpers +
                   "o1 = np.array(['a', 'a medium text over fifteen bytes', np.str_('np-str'), ''], dtype=object)\n" +
                   "o2 = np.array([['x', 'y'], ['z', 'w']], dtype=object).T\n" +
                   "o3 = o1[::-2]\n" +
                   "o0 = np.array('solo', dtype=object)\n" +
                   "oe = np.empty((0, 3), dtype=object)");
            foreach (string name in new[] { "o1", "o2", "o3", "o0", "oe" })
            {
                using NDArray copy = ImportOf(name);
                copy.dtype.Should().Be(np.dtypes.StringDType(), name);
                copy.Shape.Dimensions.Should().Equal(PyDims(name), name);
                Encode(copy).Should().Equal(NumpyEncode($"{name}.astype(StringDType())"), name);
                ((Action)(() => ViewOf(name, allowReadonly: true).Dispose()))
                    .Should().Throw<NotSupportedException>().WithMessage("*object dtype*", name);
            }

            PyExec("bad_none = np.array(['a', None], dtype=object)\n" +
                   "bad_nan = np.array([['a', 'b'], ['c', float('nan')]], dtype=object)\n" +
                   "bad_int = np.array(['a', 'b', 5], dtype=object)\n" +
                   "bad_bytes = np.array(['a', b'x'], dtype=object)\n" +
                   "bad_surrogate = np.array(['a', '\\ud800'], dtype=object)\n" +
                   "dicts = np.array([{}, {}], dtype=object)");
            ((Action)(() => ImportOf("bad_none").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*NoneType at flat index 1 (None)*");
            ((Action)(() => ImportOf("bad_nan").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*float at flat index 3 (nan)*");
            ((Action)(() => ImportOf("bad_int").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*int at flat index 2 (5)*");
            ((Action)(() => ImportOf("bad_bytes").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*bytes at flat index 1 (b'x')*");
            ((Action)(() => ImportOf("bad_surrogate").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*only str elements*UTF-8*surrogates not allowed*");
            ((Action)(() => ImportOf("dicts").Dispose())).Should().Throw<NotSupportedException>().WithMessage("*dict at flat index 0*");
        }

        /// <summary>
        ///     The codec over <c>StringDType</c>: decode is a copy in Auto and Copy and declined in View; encode is a numpy
        ///     <c>StringDType</c> copy in Auto and Copy and declined in View (pythonnet then falls back to wrapping the CLR
        ///     object, as for any array the codec cannot express).
        /// </summary>
        [TestMethod]
        public void Codec_Modes_StringDType()
        {
            var auto = new NumpyCodec();
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View, EncodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy, EncodeMode = NumpyCodecMode.Copy });
            PyExec(Helpers + "ct = np.array(['a', None, 'a long text over fifteen bytes'], dtype=StringDType(na_object=None))");
            using (Gil())
            {
                using PyObject ct = Scope.Get("ct");
                viewOnly.TryDecode(ct, out NDArray declined).Should().BeFalse("a StringDType array has no view");
                declined.Should().BeNull();
                foreach (var codec in new[] { auto, copyOnly })
                {
                    codec.TryDecode(ct, out NDArray copy).Should().BeTrue();
                    using (copy)
                    {
                        copy.dtype.Should().Be(np.dtypes.StringDType((object)null));
                        copy.GetValue(1).Should().BeNull("NA stays NA");
                        copy.GetValue(2).Should().Be("a long text over fifteen bytes");
                    }
                }

                NDArray ns = np.array(new object[] { "x", null }, np.dtypes.StringDType((object)null));
                viewOnly.TryEncode(ns).Should().BeNull("View declines what only a copy can express");
                foreach (var codec in new[] { auto, copyOnly })
                {
                    using PyObject encoded = codec.TryEncode(ns);
                    encoded.Should().NotBeNull();
                    using PyObject roundTrip = Python.np.with("repr(a.dtype) + ' ' + repr(a.tolist())", ("a", encoded));
                    roundTrip.As<string>().Should().Be("StringDType(na_object=None) ['x', None]");
                }
            }
        }

        /// <summary>
        ///     A <c>StringDType</c> copy holds no Python lease and no NumSharp export pin: the numpy source can die before the
        ///     NumSharp copy is read, and the numpy copy outlives the disposed NumSharp source.
        /// </summary>
        [TestMethod]
        public void Copies_AreIndependentOfTheSourcesLifetime()
        {
            int imports = NDArrayPythonInterop.LiveImports, exports = NDArrayPythonInterop.LiveExports;
            PyExec(Helpers + "doomed = np.array(['a', 'a long text over fifteen bytes'], dtype=StringDType())");
            using NDArray copy = ImportOf("doomed");
            PyExec("del doomed\nimport gc; gc.collect()");
            copy.GetValue(1).Should().Be("a long text over fifteen bytes");

            NDArray source = np.array(new object[] { "b", "another long text over fifteen bytes" }, np.dtypes.StringDType());
            ExportCopyTo("survivor", source);
            source.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            PyStr("survivor[1]").Should().Be("another long text over fifteen bytes");
            NDArrayPythonInterop.LiveImports.Should().Be(imports, "a copy leases nothing");
            NDArrayPythonInterop.LiveExports.Should().Be(exports, "a copy pins nothing");
            PyExec("del survivor");
        }

        /// <summary>
        ///     An allocation failure in the middle of an import (NumSharp's fault hook, the n-th arena allocation of the
        ///     thread) raises the out-of-memory error and releases the half-built copy: no allocator is left alive.
        /// </summary>
        [TestMethod]
        public void FaultInjection_DuringImport_ReleasesTheHalfBuiltCopy()
        {
            PyExec(Helpers + "many = np.array(['x' * (20 + i) for i in range(12)], dtype=StringDType())");
            long allocators = VStringAllocator.LiveAllocators;
            for (int successes = 0; successes < 12; successes += 3)
            {
                VStringAllocator.FailAllocationAfter(successes);
                try
                {
                    ((Action)(() => ImportOf("many").Dispose()))
                        .Should().Throw<OutOfMemoryException>().WithMessage("*Failed to allocate string while copying a numpy StringDType array*", $"after {successes} successes");
                }
                finally
                {
                    VStringAllocator.DisarmFaultInjection();
                }

                GC.Collect();
                GC.WaitForPendingFinalizers();
                VStringAllocator.LiveAllocators.Should().Be(allocators, $"after {successes} successes: the half-built copy is released");
            }

            using NDArray ok = ImportOf("many");
            ok.GetValue(11).Should().Be(new string('x', 31), "the hook is disarmed: the next import succeeds");
        }
    }
}
