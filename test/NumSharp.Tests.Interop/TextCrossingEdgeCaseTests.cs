using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Strings;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Edge cases of the text crossing (strings phase 1, P1.8) against live NumPy 2.4.2, beyond the per-layout matrices of
    ///     <see cref="TextFixedWidthCrossingTests"/> and <see cref="TextStringDTypeCrossingTests"/>: cell histories (heap
    ///     rewrites, NA and <c>''</c> over arena strings, a short string over an arena cell), UTF-8 sizes at every cell and
    ///     prefix boundary, more missing-data objects, ranks up to 6 with composed views, empty and read-only sources, an
    ///     arena past 64 KiB, seeded random differentials, the codec end to end, concurrent threads, and the non-text import
    ///     routes the P1.8 routing change touched.
    /// </summary>
    /// <remarks>
    ///     Every <c>StringDType</c> comparison holds the copy to numpy's own copy CELL FOR CELL (<c>tobytes()</c>): numpy
    ///     duplicates each element with <c>NpyString_dup</c>, and so does the bridge, so even the leftover bytes behind a short
    ///     text (the old arena offset of a cell that held a longer string) must survive.
    /// </remarks>
    [TestClass]
    public class TextCrossingEdgeCaseTests : InteropTestBase
    {
        /// <summary>
        ///     The Python helpers: the value encoding (<c>_enc</c>: <c>N:</c> for a None / NaN missing value, <c>s:</c> + UTF-8 hex
        ///     for a text, <c>o:</c> + repr otherwise) and the seeded random arrays and views of the differential tests.
        /// </summary>
        private const string Helpers = """
            import random
            from numpy.dtypes import StringDType
            def _enc_one(x):
                if x is None: return 'N:None'
                if isinstance(x, (float, np.floating)) and x != x: return 'N:nan'
                if isinstance(x, str): return 's:' + x.encode('utf-8').hex()
                return 'o:' + repr(x)
            def _enc(a):
                return [_enc_one(x) for x in (a.ravel().tolist() if a.ndim else [a.item()])]
            _ALPH = ['a', 'Z', '0', ' ', '\x00', 'é', '漢', '😀', '\n']
            _SIZES = [0, 1, 14, 15, 16, 17, 254, 255, 256, 257, 600]
            def _mk(n, unit):
                s = ''
                while len((s + unit).encode('utf-8')) <= n:
                    s += unit
                return s + 'x' * (n - len(s.encode('utf-8')))
            def _rtext(rng):
                n = rng.choice(_SIZES) if rng.random() < 0.6 else rng.randint(0, 120)
                out, k = [], 0
                while k < n:
                    c = rng.choice([c for c in _ALPH if len(c.encode('utf-8')) <= n - k])
                    out.append(c)
                    k += len(c.encode('utf-8'))
                return ''.join(out)
            _NA = [None, float('nan'), 'NA', '', 5, True, b'x', 2 ** 70, -0.0]
            def _rshape(rng):
                dims, total = [], 1
                for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4, 5])):
                    d = rng.choice([0, 1, 2, 3]) if rng.random() < 0.08 else rng.choice([1, 2, 3, 4])
                    if total * max(d, 1) > 120:
                        d = 1
                    dims.append(d)
                    total *= max(d, 1)
                return tuple(dims)
            def _rview(rng, v):
                for _ in range(rng.randint(0, 3)):
                    if v.ndim == 0:
                        break
                    k = rng.random()
                    if k < 0.4:
                        v = v[tuple(slice(rng.randrange(d) if (d and rng.random() < 0.3) else None, None, rng.choice([1, 2, -1, -2])) for d in v.shape) + (Ellipsis,)]
                    elif k < 0.6:
                        p = list(range(v.ndim)); rng.shuffle(p); v = v.transpose(p)
                    elif k < 0.75 and v.ndim < 6:
                        i = [slice(None)] * v.ndim; i.insert(rng.randint(0, v.ndim), None); v = v[tuple(i)]
                    elif k < 0.85 and v.ndim < 6:
                        v = np.broadcast_to(v, (rng.randint(1, 3),) + v.shape)
                    else:
                        ax = rng.randrange(v.ndim)
                        if v.shape[ax]:
                            i = [slice(None)] * v.ndim; i[ax] = rng.randrange(v.shape[ax]); v = v[tuple(i) + (Ellipsis,)]
                return v
            def _t_case(seed):
                rng = random.Random(seed)
                has = rng.random() < 0.8
                na = rng.choice(_NA)
                coerce = rng.random() < 0.7
                dt = StringDType(na_object=na, coerce=coerce) if has else StringDType(coerce=coerce)
                shape = _rshape(rng)
                n = int(np.prod(shape)) if shape else 1
                a = np.empty(n, dtype=dt)
                for i in range(n):
                    a[i] = dt.na_object if (has and rng.random() < 0.12) else _rtext(rng)
                for _ in range(rng.randint(0, max(1, n // 2))):
                    if not n:
                        break
                    i, r = rng.randrange(n), rng.random()
                    if r < 0.3: a[i] = _rtext(rng)
                    elif r < 0.45 and has: a[i] = dt.na_object
                    elif r < 0.6: a[i] = ''
                    elif r < 0.8: a[i] = 'H' * rng.choice([300, 700])
                    else: a[i] = 'sh%d' % rng.randrange(100)
                return _rview(rng, a.reshape(shape))
            def _f_case(seed):
                rng = random.Random(seed)
                kind = rng.choice(['U', 'S'])
                width = rng.choice([1, 2, 3, 8, 33])
                order = rng.choice(['<', '<', '>']) if kind == 'U' else '|'
                shape = _rshape(rng)
                n = int(np.prod(shape)) if shape else 1
                if kind == 'U':
                    raw = np.array([rng.choice([0, 0x41, 0xE9, 0x6F22, 0x1F600, 0x10FFFF, 0xD800, 0x110000]) if rng.random() < 0.3 else rng.randint(0x20, 0x7E) for _ in range(n * width)], dtype='<u4')
                    a = raw.view('<U%d' % width).astype('%sU%d' % (order, width))
                else:
                    a = np.frombuffer(bytes(rng.randrange(256) for _ in range(n * width)), dtype=np.uint8).view('S%d' % width).copy()
                v = _rview(rng, a.reshape(shape))
                if rng.random() < 0.2:
                    v.flags.writeable = False
                return v
            """;

        /// <summary>
        ///     NumSharp's side of <c>_enc</c>: each element in C order as <c>N:None</c> / <c>N:nan</c> (a None or NaN missing
        ///     value), <c>s:</c> + the UTF-8 hex of a text, or <c>o:</c> + Python's repr of any other missing-value object.
        /// </summary>
        /// <param name="nd">A NumSharp <c>StringDType</c> array of any layout.</param>
        /// <returns>The encoded values.</returns>
        private static List<string> Encode(NDArray nd)
        {
            var result = new List<string>();
            if (nd.size == 0)
                return result;
            using NDArray flat = nd.ndim == 0 ? nd.reshape(1) : nd.flatten();
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

        /// <summary>A Python list of str (an <c>_enc(...)</c> result) as a C# list.</summary>
        /// <param name="expr">A Python expression in the test scope.</param>
        /// <returns>The strings.</returns>
        private List<string> PyStrings(string expr)
        {
            using (Gil())
            {
                using PyObject value = Scope.Eval(expr);
                using var list = new PyList(value);
                var result = new List<string>();
                for (int i = 0; i < list.Length(); i++)
                {
                    using PyObject item = list[i];
                    result.Add(item.As<string>());
                }

                return result;
            }
        }

        /// <summary>The bytes a Python expression evaluates to.</summary>
        /// <param name="expr">A Python expression in the test scope (a <c>bytes</c> value).</param>
        /// <returns>The bytes.</returns>
        private byte[] PyBytes(string expr)
        {
            using (Gil())
            {
                using PyObject value = Scope.Eval(expr);
                return value.As<byte[]>();
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
        ///     Imports the numpy <c>StringDType</c> expression and holds the copy to numpy's own C-order copy: the shape, the
        ///     values with NA identity, and the cells; then exports it back and holds the numpy result to the same values and
        ///     cells.
        /// </summary>
        /// <param name="expr">A numpy <c>StringDType</c> array expression.</param>
        /// <param name="because">The case name for the assertion messages.</param>
        private void AssertImportsAsNumpysCopyAndBack(string expr, string because)
        {
            PyExec($"_src = {expr}\n_npcopy = np.array(_src, order='C', copy=True)");
            using NDArray copy = ImportOf("_src");
            copy.Shape.Dimensions.Should().Equal(PyDims("_src"), because);
            Encode(copy).Should().Equal(PyStrings("_enc(_src)"), because);
            byte[] cells = PyBytes("_npcopy.tobytes()");
            copy.tobytes().Should().Equal(cells, $"{because}: the cells numpy's own copy builds");
            using (Gil())
            {
                using PyObject back = copy.ToNumpyCopy();
                Scope.Set("_back", back);
            }

            PyStr("repr(_back.dtype)").Should().Be(copy.dtype.ToString(), because);
            PyBool("_enc(_back) == _enc(_src)").Should().BeTrue($"{because}: the values come back");
            PyBytes("_back.tobytes()").Should().Equal(cells, $"{because}: the export duplicates the same cells again");
            PyExec("del _src, _npcopy, _back");
        }

        /// <summary>
        ///     Exports a NumSharp <c>StringDType</c> array (any layout) and holds the numpy copy to NumSharp's own values and to
        ///     NumSharp's C-order copy cell for cell; then imports the numpy array back and holds it to the same.
        /// </summary>
        /// <param name="view">The NumSharp array.</param>
        /// <param name="because">The case name for the assertion messages.</param>
        private void AssertExportsAsNumSharpsCopyAndBack(NDArray view, string because)
        {
            using (Gil())
            {
                using PyObject x = view.ToNumpyCopy();
                Scope.Set("_x", x);
            }

            PyStr("repr(_x.dtype)").Should().Be(view.dtype.ToString(), because);
            PyDims("_x").Should().Equal(view.Shape.Dimensions, because);
            List<string> values = Encode(view);
            PyStrings("_enc(_x)").Should().Equal(values, because);
            using NDArray nsCopy = np.copy(view, order: 'C');
            byte[] cells = nsCopy.tobytes();
            PyBytes("_x.tobytes()").Should().Equal(cells, $"{because}: the cells NumSharp's own C-order copy builds");
            using NDArray back = ImportOf("_x");
            Encode(back).Should().Equal(values, $"{because}: the round trip");
            back.tobytes().Should().Equal(cells, $"{because}: the round trip's cells");
            PyExec("del _x");
        }

        /// <summary>
        ///     Cells with a history cross exactly: an arena string set to <c>''</c> (an arena cell of size 0), an arena string set
        ///     to NA (the NA keeps the allocation's metadata), a short string rewritten long and an arena string rewritten longer
        ///     (heap strings), an arena string rewritten shorter (its slot reused), a heap string set to NA and to <c>''</c>, NA
        ///     set back to text, and a SHORT text written over an arena cell past the 64 KiB offset, whose cell keeps the high
        ///     bytes of the old offset behind the text. The import equals numpy's own copy cell for cell, the export duplicates
        ///     the same cells, and the same history written by NumSharp exports to those cells too.
        /// </summary>
        [TestMethod]
        public void StringDType_CellHistories_CrossAsNumpysOwnCopy()
        {
            PyExec(Helpers);
            PyExec("""
                dt = StringDType(na_object=None)
                filler = ['%05d-filler-text-of-thirty-bytes' % i for i in range(3000)]
                h = np.array(filler + [''] * 9, dtype=dt)
                n = len(filler)
                h[n + 0] = 'an arena string of medium size'; h[n + 0] = ''
                h[n + 1] = 'an arena string that becomes NA'; h[n + 1] = None
                h[n + 2] = 'short'; h[n + 2] = 'now a long string, longer than the short one, ' * 3
                h[n + 3] = 'arena text of 20 b.'; h[n + 3] = 'L' * 300
                h[n + 4] = 'an arena string of thirty bytes'; h[n + 4] = 'shorter, still medium'
                h[n + 5] = 'H' * 400; h[n + 5] = 'H' * 500; h[n + 5] = None
                h[n + 6] = 'J' * 400; h[n + 6] = 'J' * 600; h[n + 6] = ''
                h[n + 7] = 'an arena string, then NA, then text'; h[n + 7] = None; h[n + 7] = 'back, fits the old slot'
                h[n + 8] = 'a medium string past the 64 KiB offset'; h[n + 8] = 'ab'
                """);

            // The case under test exists: numpy's own copy moves the short cell as its 16 raw bytes, so the old arena
            // offset's third byte (the offset is past 65,535) stays behind the two text bytes.
            byte[] numpyCopy = PyBytes("np.array(h, copy=True).tobytes()");
            numpyCopy[3008 * 16 + 2].Should().NotBe(0, "the short cell keeps the high bytes of its old arena offset");

            AssertImportsAsNumpysCopyAndBack("h", "numpy's history");

            // The same history written by NumSharp: its export (NumSharp's cells duplicated into numpy's allocator) is numpy's
            // own copy of numpy's history, cell for cell.
            var values = Enumerable.Range(0, 3000).Select(i => (object)$"{i:D5}-filler-text-of-thirty-bytes")
                                   .Concat(Enumerable.Repeat((object)"", 9)).ToArray();
            using NDArray m = np.array(values, np.dtypes.StringDType((object)null));
            const int n = 3000;
            void Set(int offset, params object[] writes)
            {
                foreach (object w in writes)
                    m.SetValue(w, n + offset);
            }

            Set(0, "an arena string of medium size", "");
            Set(1, "an arena string that becomes NA", null);
            Set(2, "short", string.Concat(Enumerable.Repeat("now a long string, longer than the short one, ", 3)));
            Set(3, "arena text of 20 b.", new string('L', 300));
            Set(4, "an arena string of thirty bytes", "shorter, still medium");
            Set(5, new string('H', 400), new string('H', 500), null);
            Set(6, new string('J', 400), new string('J', 600), "");
            Set(7, "an arena string, then NA, then text", null, "back, fits the old slot");
            Set(8, "a medium string past the 64 KiB offset", "ab");
            using (Gil())
            {
                using PyObject mx = m.ToNumpyCopy();
                Scope.Set("mx", mx);
            }

            PyBool("_enc(mx) == _enc(h)").Should().BeTrue("NumSharp's history reads as numpy's");
            PyBytes("mx.tobytes()").Should().Equal(numpyCopy, "NumSharp's history exports to the cells of numpy's own copy of numpy's history");
            PyExec("del mx, h");
        }

        /// <summary>
        ///     Texts whose UTF-8 sizes sit on every cell and prefix boundary — 0, 1, 14, 15 (the last short size), 16 (the first
        ///     arena size), 17, 254, 255 (the last one-byte prefix), 256 (the first eight-byte prefix), 257, 1,000 and 70,000
        ///     bytes — built from 1-, 2-, 3- and 4-byte characters, plus NUL runs at those sizes: both directions, values and
        ///     cells; a NumSharp-born array exports to the cells numpy's own <c>np.array(list, dtype)</c> builds.
        /// </summary>
        [TestMethod]
        public void StringDType_SizeBoundaries_CrossExactly()
        {
            PyExec(Helpers);
            PyExec("""
                sizes = [0, 1, 14, 15, 16, 17, 254, 255, 256, 257, 1000, 70000]
                vals = [_mk(n, u) for n in sizes for u in ['a', 'é', '漢', '😀']]
                vals += ['\x00', 'abc\x00\x00', '\x00' * 15, '\x00' * 16, 'x' * 14 + '\x00', 'é' * 7 + '\x00', '\x00' * 255, '\x00' * 256]
                sized = [len(v.encode('utf-8')) for v in vals[:48]] == [n for n in sizes for _ in range(4)]
                b = np.array(vals, dtype=StringDType())
                """);
            PyBool("sized").Should().BeTrue("the generator built exact UTF-8 sizes");

            AssertImportsAsNumpysCopyAndBack("b", "numpy-born boundaries");
            AssertImportsAsNumpysCopyAndBack("b[::-1]", "reversed boundaries");

            List<string> texts;
            using (Gil())
            {
                using PyObject vals = Scope.Get("vals");
                using var list = new PyList(vals);
                texts = Enumerable.Range(0, (int)list.Length()).Select(i => { using PyObject s = list[i]; return s.As<string>(); }).ToList();
            }

            using NDArray nsb = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType());
            using (Gil())
            {
                using PyObject x = nsb.ToNumpyCopy();
                Scope.Set("x", x);
            }

            PyBool("_enc(x) == _enc(b)").Should().BeTrue("the NumSharp-born values are numpy's");
            PyBool("x.tobytes() == b.tobytes()").Should().BeTrue("the export builds the cells np.array(list, dtype) builds");
            AssertExportsAsNumSharpsCopyAndBack(nsb["::-3"], "NumSharp-born, stepped");
            PyExec("del x, b, vals");
        }

        /// <summary>
        ///     More missing-data objects cross both ways with their identity: a Python int past int64 (<see cref="BigInteger"/>),
        ///     a negative int, False, -0.0, inf, empty bytes, a non-BMP str, and the numpy scalars that map to their Python base
        ///     type (<c>np.bytes_</c>, <c>np.str_</c>, <c>np.complex128</c>; the repr loses the <c>np.</c> prefix), and a complex
        ///     NaN (equal only to itself, as in numpy). The NA cells are made by numpy's own T → T cast (numpy's setitem would
        ///     CAST an <c>np.bytes_</c> / <c>np.str_</c> value into text). Numeric numpy scalars with no C# form
        ///     (<c>np.int64</c>, <c>np.float32</c>) and an arbitrary object (Ellipsis) are refused.
        /// </summary>
        /// <remarks>
        ///     The <c>np.str_</c> case failed before this test (pythonnet's string conversion refuses a <c>str</c> subclass:
        ///     <c>'numpy.str_' value cannot be converted to System.String</c>); the bridge now reads it through <c>str()</c>.
        /// </remarks>
        [TestMethod]
        public void StringDType_MoreNAObjects_CrossBothWays()
        {
            PyExec(Helpers);
            PyExec("""
                _base = np.array(['a', None, 'a long string over fifteen bytes'], dtype=StringDType(na_object=None))
                _kinds = {
                    'big': 2 ** 70, 'neg': -7, 'false': False, 'negzero': -0.0, 'inf': float('inf'), 'emptybytes': b'',
                    'unicode': 'N/A😀', 'npbytes': np.bytes_(b'x'), 'npstr': np.str_('NA'), 'npcomplex': np.complex128(1 + 2j),
                    'complexnan': complex(float('nan'), 0),
                }
                na_arrays = {k: _base.astype(StringDType(na_object=v)) for k, v in _kinds.items()}
                refused = {k: _base.astype(StringDType(na_object=v)) for k, v in {'int64': np.int64(5), 'float32': np.float32('nan'), 'ellipsis': Ellipsis}.items()}
                """);
            var kinds = new (string Key, string Repr, Func<object, bool> IsNA)[]
            {
                ("big", "StringDType(na_object=1180591620717411303424)", o => o is BigInteger b && b == BigInteger.Pow(2, 70)),
                ("neg", "StringDType(na_object=-7)", o => o is long l && l == -7),
                ("false", "StringDType(na_object=False)", o => o is bool v && !v),
                ("negzero", "StringDType(na_object=-0.0)", o => o is double d && d == 0 && double.IsNegative(d)),
                ("inf", "StringDType(na_object=inf)", o => o is double d && double.IsPositiveInfinity(d)),
                ("emptybytes", "StringDType(na_object=b'')", o => o is byte[] b && b.Length == 0),
                ("unicode", "StringDType(na_object='N/A" + char.ConvertFromUtf32(0x1F600) + "')", o => o is string s && s == "N/A" + char.ConvertFromUtf32(0x1F600)),
                ("npbytes", "StringDType(na_object=b'x')", o => o is byte[] b && b.Length == 1 && b[0] == (byte)'x'),
                ("npstr", "StringDType(na_object='NA')", o => o is string s && s == "NA"),
                ("npcomplex", "StringDType(na_object=(1+2j))", o => o is Complex c && c == new Complex(1, 2)),
                ("complexnan", "StringDType(na_object=(nan+0j))", o => o is Complex c && double.IsNaN(c.Real) && c.Imaginary == 0),
            };
            foreach (var (key, repr, isNA) in kinds)
            {
                using NDArray copy = ImportOf($"na_arrays['{key}']");
                copy.dtype.ToString().Should().Be(repr, key);
                copy.dtype.HasNAObject.Should().BeTrue(key);
                isNA(copy.dtype.na_object).Should().BeTrue($"{key}: the NA object's C# form");
                isNA(copy.GetValue(1)).Should().BeTrue($"{key}: the NA cell reads as the NA object");
                copy.GetValue(2).Should().Be("a long string over fifteen bytes", key);
                copy.tobytes().Should().Equal(PyBytes($"np.array(na_arrays['{key}'], copy=True).tobytes()"), $"{key}: numpy's own copy's cells");
                using (Gil())
                {
                    using PyObject back = copy.ToNumpyCopy();
                    Scope.Set("back", back);
                }

                PyStr("repr(back.dtype)").Should().Be(repr, $"{key}: the descriptor numpy gets back");
                PyBool("back[1] is back.dtype.na_object").Should().BeTrue($"{key}: the NA cell is numpy's NA");
                PyStr("back[0] + '|' + back[2]").Should().Be("a|a long string over fifteen bytes", key);
                PyExec("del back");
            }

            foreach (string key in new[] { "int64", "float32", "ellipsis" })
            {
                ((Action)(() => ImportOf($"refused['{key}']").Dispose()))
                    .Should().Throw<NotSupportedException>().WithMessage("*has no C# form*", key);
            }

            PyExec("del na_arrays, refused");
        }

        /// <summary>
        ///     Missing-data objects born in NumSharp with C# types an import never produces — a 32-bit int, a short, a byte, a
        ///     ulong past <see cref="long.MaxValue"/>, a float, a <see cref="Half"/> and a char — export with their Python form
        ///     (an <c>int</c>, a <c>float</c>, an <c>np.float16</c>, a one-character <c>str</c>), the cell storing the value is
        ///     numpy's NA, and the numpy copy imports back with an equal descriptor (the C# type may widen: an int returns as a
        ///     long, a ulong as a <see cref="BigInteger"/>) and the same cells.
        /// </summary>
        [TestMethod]
        public void StringDType_NumSharpBornNAObjects_ExportWithTheirPythonForm()
        {
            var kinds = new (object NA, string Repr, string PythonType)[]
            {
                (5, "StringDType(na_object=5)", "int"),
                ((short)-3, "StringDType(na_object=-3)", "int"),
                ((byte)200, "StringDType(na_object=200)", "int"),
                (ulong.MaxValue, "StringDType(na_object=18446744073709551615)", "int"),
                (1.5f, "StringDType(na_object=1.5)", "float"),
                ((Half)1.5, "StringDType(na_object=np.float16(1.5))", "float16"),
                ('x', "StringDType(na_object='x')", "str"),
            };
            foreach (var (na, repr, pythonType) in kinds)
            {
                DType descr = np.dtypes.StringDType(na);
                descr.ToString().Should().Be(repr, $"{na.GetType().Name}: NumSharp's own repr");
                using NDArray m = np.array(new object[] { "a", na, "a long string over fifteen bytes" }, descr);
                using (Gil())
                {
                    using PyObject x = m.ToNumpyCopy();
                    Scope.Set("x", x);
                }

                PyStr("repr(x.dtype)").Should().Be(repr, repr);
                PyStr("type(x.dtype.na_object).__name__").Should().Be(pythonType, $"{repr}: the Python form");
                PyBool("x[1] is x.dtype.na_object").Should().BeTrue($"{repr}: storing the NA value made the cell numpy's NA");
                PyStr("x[0] + '|' + x[2]").Should().Be("a|a long string over fifteen bytes", repr);
                using (NDArray back = ImportOf("x"))
                {
                    back.dtype.Should().Be(descr, $"{repr}: the descriptor comes back equal");
                    back.tobytes().Should().Equal(m.tobytes(), $"{repr}: and so do the cells");
                }

                PyExec("del x");
            }
        }

        /// <summary>
        ///     Ranks 5 and 6 with composed views — a permutation with a new axis and reversed steps, an integer index with
        ///     negative steps, a 6-D broadcast, an empty middle axis, a reshape-transpose with a leading new axis — import as
        ///     numpy's own copy; and the same views taken in NumSharp read the same values as numpy's and export as NumSharp's
        ///     own copy.
        /// </summary>
        [TestMethod]
        public void StringDType_HighRankAndComposedViews_BothWays()
        {
            PyExec(Helpers);
            PyExec("""
                _v = [('%d-' % i) * (i % 7) + ('long tail text number %d' % i if i % 5 == 0 else '') for i in range(120)]
                hr = np.array(_v, dtype=StringDType(na_object='NA')).reshape(2, 3, 1, 4, 5)
                hr[0, 1, 0, 2, 3] = hr.dtype.na_object
                hr[1, 2, 0, 0, 0] = hr.dtype.na_object
                views = {
                    'perm+newaxis': hr.transpose(4, 2, 0, 3, 1)[::-1, :, None, ::2],
                    'int+steps': hr[1, ::-1, :, 1::2, ::-3],
                    'bcast6': np.broadcast_to(hr[:, :1], (3,) + hr[:, :1].shape),
                    'empty': hr[:, :, :, 1:1],
                    'reshape-T': hr.reshape(6, 20)[::-2].T[None],
                }
                """);
            foreach (string key in new[] { "perm+newaxis", "int+steps", "bcast6", "empty", "reshape-T" })
                AssertImportsAsNumpysCopyAndBack($"views['{key}']", key);

            // numpy's [::-1, :, None, ::2] steps the axis AFTER the new one (the transpose's third axis), so the mirror steps
            // that axis first and inserts the new axis at position 2 afterwards.
            using NDArray x = ImportOf("hr");
            var mirrors = new (string Key, NDArray View)[]
            {
                ("perm+newaxis", np.expand_dims(np.transpose(x, new[] { 4, 2, 0, 3, 1 })["::-1, :, ::2"], 2)),
                ("int+steps", x["1, ::-1, :, 1::2, ::-3"]),
                ("bcast6", np.broadcast_to(x[":, :1"], new Shape(3, 2, 1, 1, 4, 5))),
                ("empty", x[":, :, :, 1:1"]),
                ("reshape-T", np.expand_dims(x.reshape(6, 20)["::-2"].T, 0)),
            };
            foreach (var (key, view) in mirrors)
            {
                Encode(view).Should().Equal(PyStrings($"_enc(views['{key}'])"), $"{key}: NumSharp's view reads numpy's values");
                AssertExportsAsNumSharpsCopyAndBack(view, $"NumSharp {key}");
            }

            PyExec("del hr, views");
        }

        /// <summary>
        ///     Empty arrays of every text kind — <c>StringDType</c> with an NA object, <c>U5</c>, <c>S3</c> — in shapes (0,), (3,
        ///     0), (0, 3, 2), (2, 0, 0, 1), a stepped empty, an empty slice of a non-empty array and an empty broadcast cross
        ///     with their shape and descriptor in both directions; fixed-width ones also as views.
        /// </summary>
        [TestMethod]
        public void Text_EmptyArrays_EveryShapeAndKind_BothWays()
        {
            PyExec(Helpers);
            PyExec("""
                _dts = {'T': StringDType(na_object=None), 'U5': np.dtype('U5'), 'S3': np.dtype('S3')}
                empties = {}
                for k, d in _dts.items():
                    empties[k] = [np.empty(s, dtype=d) for s in [(0,), (3, 0), (0, 3, 2), (2, 0, 0, 1)]]
                    empties[k] += [np.empty((4, 3), dtype=d)[::2, 1:1], np.zeros((4, 3), dtype=d)[1:1], np.broadcast_to(np.empty((0,), dtype=d), (2, 0))]
                """);
            foreach (string kind in new[] { "T", "U5", "S3" })
            {
                for (int i = 0; i < 7; i++)
                {
                    string expr = $"empties['{kind}'][{i}]";
                    string cell = $"{kind} #{i}";
                    using (NDArray copy = ImportOf(expr))
                    {
                        copy.size.Should().Be(0, cell);
                        copy.Shape.Dimensions.Should().Equal(PyDims(expr), cell);
                        copy.dtype.ToString().Should().Be(kind == "T" ? "StringDType(na_object=None)" : PyStr($"str({expr}.dtype)"), cell);
                        copy.Shape.IsWriteable.Should().BeTrue($"{cell}: a copy is writeable");
                        using (Gil())
                        {
                            using PyObject back = kind == "T" ? copy.ToNumpyCopy() : copy.ToNumpy();
                            Scope.Set("back", back);
                        }

                        PyStr("repr(back.dtype) + ' ' + str(back.shape)").Should().Be(PyStr($"repr({expr}.dtype) + ' ' + str({expr}.shape)"), cell);
                        PyExec("del back");
                    }

                    if (kind == "T")
                    {
                        ((Action)(() => ViewOf(expr, allowReadonly: true).Dispose()))
                            .Should().Throw<NotSupportedException>().WithMessage("*has no buffer form*", cell);
                        continue;
                    }

                    using NDArray view = ViewOf(expr, allowReadonly: true);
                    view.Shape.Dimensions.Should().Equal(PyDims(expr), cell);
                    view.Shape.IsWriteable.Should().Be(PyBool($"{expr}.flags.writeable"), $"{cell}: the source's writeability");
                }
            }

            PyExec("del empties");
        }

        /// <summary>
        ///     An EMPTY read-only source keeps its writeability across the bridge, like a non-empty one (fixed with this test:
        ///     the empty early-returns built a writeable array). Import: the default view refuses with the read-only text, the
        ///     opt-in view is non-writeable, the copy is writeable — for a read-only numpy array, an empty broadcast, an empty
        ///     <c>np.frombuffer(b'')</c> and an empty slice of a read-only memoryview. Export: an empty NumSharp broadcast is a
        ///     read-only numpy array, an empty plain array a writeable one.
        /// </summary>
        [TestMethod]
        public void EmptyReadOnlySources_KeepTheirWriteability_BothWays()
        {
            PyExec("ro = np.zeros((0, 3), dtype='f8'); ro.flags.writeable = False\n" +
                   "rob = np.broadcast_to(np.empty((0,), dtype='U3'), (2, 0))\n" +
                   "rfb = np.frombuffer(b'', dtype=np.uint8)\n" +
                   "rmv = memoryview(b'abcdef')[2:2]");
            foreach (string name in new[] { "ro", "rob", "rfb", "rmv" })
            {
                ((Action)(() => ViewOf(name).Dispose()))
                    .Should().Throw<InvalidOperationException>().WithMessage("*read-only*", name);
                using (NDArray view = ViewOf(name, allowReadonly: true))
                {
                    view.size.Should().Be(0, name);
                    view.Shape.IsWriteable.Should().BeFalse($"{name}: the opt-in view of a read-only source is non-writeable");
                }

                using NDArray copy = ImportOf(name);
                copy.Shape.IsWriteable.Should().BeTrue($"{name}: a copy is writeable");
            }

            PyExec("rw = np.zeros((0, 3), dtype='f8')");
            using (NDArray writable = ViewOf("rw"))
                writable.Shape.IsWriteable.Should().BeTrue("an empty writeable source views writeable");

            using NDArray emptyF8 = np.zeros(new Shape(0), NPTypeCode.Double);
            using NDArray emptyU3 = np.zeros(new Shape(0), np.dtype("U3"));
            foreach (var (name, source, writeable) in new (string, NDArray, bool)[]
                     {
                         ("broadcast f8", np.broadcast_to(emptyF8, new Shape(2, 0)), false),
                         ("broadcast U3", np.broadcast_to(emptyU3, new Shape(2, 0)), false),
                         ("plain f8", emptyF8, true),
                         ("plain U3", emptyU3, true),
                     })
            {
                source.Shape.IsWriteable.Should().Be(writeable, name);
                using (Gil())
                {
                    using PyObject x = source.ToNumpy();
                    Scope.Set("x", x);
                }

                PyBool("x.flags.writeable").Should().Be(writeable, $"{name}: numpy's view keeps NumSharp's writeability");
                PyExec("del x");
            }
        }

        /// <summary>
        ///     A read-only numpy <c>StringDType</c> array imports as a writeable copy, and a broadcast one duplicates every
        ///     occurrence (each long string gets its own arena slot, as in numpy's own copy).
        /// </summary>
        [TestMethod]
        public void StringDType_ReadOnlyAndBroadcastSources_Import()
        {
            PyExec(Helpers);
            PyExec("""
                r = np.array(['x', None, 'a long text in the arena, thirty', 'y' * 300], dtype=StringDType(na_object=None))
                r.flags.writeable = False
                rb = np.broadcast_to(r, (3, 4))
                """);
            using (NDArray copy = ImportOf("r"))
            {
                copy.Shape.IsWriteable.Should().BeTrue("a copy is writeable");
                copy.SetValue("written", 0);
                copy.GetValue(0).Should().Be("written");
                PyStr("r[0]").Should().Be("x", "the numpy source is untouched");
            }

            AssertImportsAsNumpysCopyAndBack("r", "read-only");
            AssertImportsAsNumpysCopyAndBack("rb", "broadcast");
            AssertImportsAsNumpysCopyAndBack("rb.T[::-1]", "broadcast, transposed, reversed");
            PyExec("del r, rb");
        }

        /// <summary>
        ///     A 100,000-element array (short, medium and long texts, an NA every tenth element) crosses both ways: the arena
        ///     grows past 64 KiB, so offsets use more than two bytes, and every cell still equals numpy's own copy.
        /// </summary>
        [TestMethod]
        public void StringDType_LargeArray_ArenaPastSixtyFourKilobytes()
        {
            PyExec(Helpers);
            PyExec("""
                big = np.array([None if i % 10 == 0 else ('%d' % i if i % 3 == 0 else ('text %08d ' % i) * (1 + i % 4)) for i in range(100000)],
                               dtype=StringDType(na_object=None))
                """);
            AssertImportsAsNumpysCopyAndBack("big", "100K elements");
            AssertImportsAsNumpysCopyAndBack("big[::-7]", "100K elements, stepped back");
            PyExec("del big");
        }

        /// <summary>
        ///     The seeded random differential over <c>StringDType</c>: random shapes up to 5-D, random NA objects and
        ///     <c>coerce</c>, cells with random histories, and random views (steps, reversals, transposes, new axes, broadcasts,
        ///     integer indexing) import as numpy's own copy and export back to the same cells.
        /// </summary>
        [TestMethod]
        public void StringDType_RandomizedDifferential()
        {
            PyExec(Helpers);
            for (int seed = 0; seed < 120; seed++)
                AssertImportsAsNumpysCopyAndBack($"_t_case({seed})", $"seed {seed}");
        }

        /// <summary>
        ///     The seeded random differential over fixed-width text: random <c>U</c> (both byte orders) and <c>S</c> arrays of
        ///     random widths, contents (NULs, astral and invalid code points, every byte value) and views. The copy has numpy's
        ///     native-order bytes; the view does too, with the source's writeability, and shares a write — except a big-endian
        ///     <c>U</c> array, which only copies.
        /// </summary>
        [TestMethod]
        public void FixedWidthText_RandomizedDifferential()
        {
            PyExec(Helpers);
            for (int seed = 0; seed < 120; seed++)
            {
                string cell = $"seed {seed}";
                PyExec($"f = _f_case({seed})\n" +
                       "fn = np.ascontiguousarray(f, dtype=f.dtype.newbyteorder('=') if f.dtype.kind == 'U' else f.dtype)");
                byte[] expected = PyBytes("fn.tobytes()");
                using (NDArray copy = ImportOf("f"))
                {
                    copy.dtype.str.Should().Be(PyStr("fn.dtype.str"), cell);
                    copy.tobytes().Should().Equal(expected, cell);
                }

                if (PyBool("f.dtype.byteorder == '>'"))
                {
                    ((Action)(() => ViewOf("f", allowReadonly: true).Dispose()))
                        .Should().Throw<NotSupportedException>().WithMessage("*big-endian*", cell);
                    continue;
                }

                using NDArray view = ViewOf("f", allowReadonly: true);
                view.tobytes().Should().Equal(expected, cell);
                view.Shape.IsWriteable.Should().Be(PyBool("f.flags.writeable"), cell);
                if (view.size == 0 || !view.Shape.IsWriteable)
                    continue;
                PyExec("f[(0,) * f.ndim] = f.dtype.type('Q' if f.dtype.kind == 'U' else b'Q')\n" +
                       "fn = np.ascontiguousarray(f)");
                view.tobytes().Should().Equal(PyBytes("fn.tobytes()"), $"{cell}: numpy's write shows through the view");
            }

            PyExec("del f, fn");
        }

        /// <summary>
        ///     Text through pythonnet's own conversion pipeline with the codec registered: a NumSharp <c>U</c> array passed to a
        ///     Python function arrives as numpy's <c>U</c> array, and numpy's <c>np.strings.upper</c> result decodes back as a
        ///     fixed-width array; a NumSharp <c>StringDType</c> array with a NaN missing value arrives as a numpy
        ///     <c>StringDType</c> copy, and the result of numpy's <c>np.strings.add</c> ufunc, which carries a NaN-like NA through,
        ///     decodes back with the NA intact; an object array of <c>str</c> decodes as <c>StringDType()</c>.
        /// </summary>
        /// <remarks>
        ///     The <c>StringDType</c> half uses <c>add</c>, not <c>upper</c>: numpy 2.4.2's <c>np.strings.upper</c> calls
        ///     <c>str.upper</c> on every element and raises on any NA that is not a <c>str</c> (probed), and its ufuncs raise
        ///     on an NA that is neither NaN-like nor a <c>str</c> (<c>None</c>).
        /// </remarks>
        [TestMethod]
        public void Codec_EndToEnd_TextThroughThePythonnetPipeline()
        {
            CodecTests.EnsureCodec();
            PyExec("def shout(a):\n    return np.strings.upper(a)\n" +
                   "def exclaim(a):\n    return np.strings.add(a, '!')\n" +
                   "def describe(a):\n    return repr(a.dtype) + ' ' + str(a.tolist())");

            using NDArray u = np.array(new[] { "ab", "cde" }).astype(np.dtype("U3"));
            using NDArray t = np.array(new object[] { "short", double.NaN, "a long text over fifteen bytes" }, np.dtypes.StringDType(double.NaN));
            using (Gil())
            {
                dynamic shout = Scope.Get("shout");
                dynamic exclaim = Scope.Get("exclaim");
                dynamic describe = Scope.Get("describe");

                ((string)describe(u)).Should().Be("dtype('<U3') ['ab', 'cde']", "a U array arrives as numpy's U3");
                using (PyObject upperU = shout(u))
                using (NDArray decoded = upperU.As<NDArray>())
                {
                    decoded.dtype.Should().Be(np.dtype("U3"));
                    decoded.GetValue(1).Should().Be("CDE");
                }

                ((string)describe(t)).Should().Be("StringDType(na_object=nan) ['short', nan, 'a long text over fifteen bytes']",
                    "a StringDType array arrives as numpy's StringDType copy, NA included");
                using (PyObject exclaimed = exclaim(t))
                using (NDArray decoded = exclaimed.As<NDArray>())
                {
                    decoded.dtype.Should().Be(np.dtypes.StringDType(double.NaN));
                    decoded.GetValue(0).Should().Be("short!");
                    (decoded.GetValue(1) is double na && double.IsNaN(na)).Should().BeTrue("the NA stays NA through np.strings.add and back");
                    decoded.GetValue(2).Should().Be("a long text over fifteen bytes!");
                }

                using (PyObject objects = Scope.Eval("np.array(['x', 'a long text over fifteen bytes'], dtype=object)"))
                using (NDArray decoded = objects.As<NDArray>())
                {
                    decoded.dtype.Should().Be(np.dtypes.StringDType());
                    decoded.GetValue(1).Should().Be("a long text over fifteen bytes");
                }
            }
        }

        /// <summary>
        ///     Eight threads cross text at once — a <c>StringDType</c> copy out (with a NaN missing value), numpy's
        ///     <c>np.strings.add</c> ufunc, a copy back, and a fixed-width view out and back — each under its own GIL
        ///     acquisition: every value is right, the NA survives, and no export, import or allocator is left behind.
        /// </summary>
        [TestMethod]
        public void Concurrency_ManyThreadsCrossText()
        {
            long allocators = VStringAllocator.LiveAllocators;
            PyObject add;
            using (Gil())
                add = Scope.Eval("np.strings.add");
            var errors = new ConcurrentQueue<string>();
            try
            {
                Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, thread =>
                {
                    for (int i = 0; i < 25; i++)
                    {
                        string longText = new string((char)('a' + thread), 20 + i);
                        using NDArray t = np.array(new object[] { $"t{thread}-{i}", double.NaN, longText }, np.dtypes.StringDType(double.NaN));
                        using NDArray u = np.array(new[] { $"u{thread}{i}" }).astype(np.dtype("U8"));
                        using (Py.GIL())
                        {
                            using PyObject suffix = new PyString("!");
                            using PyObject p = t.ToNumpyCopy();
                            using PyObject added = add.Invoke(p, suffix);
                            using NDArray back = added.ToNDArray();
                            if ((string)back.GetValue(0) != $"t{thread}-{i}!" || !(back.GetValue(1) is double na && double.IsNaN(na)) ||
                                (string)back.GetValue(2) != longText + "!")
                                errors.Enqueue($"thread {thread} iteration {i}: StringDType values");

                            using PyObject pu = u.ToNumpy();
                            using NDArray vu = pu.AsNDArray();
                            if ((string)vu.GetValue(0) != $"u{thread}{i}")
                                errors.Enqueue($"thread {thread} iteration {i}: U view");
                        }
                    }
                });
            }
            finally
            {
                using (Gil())
                    add.Dispose();
            }

            errors.Should().BeEmpty();
            // At most the baseline, not exactly it: an allocator another test left to the finalizer may be collected meanwhile.
            WaitFor(() => { Pump(); return VStringAllocator.LiveAllocators <= allocators; }).Should().BeTrue("every StringDType copy was released");
        }

        /// <summary>
        ///     The non-text import routes the P1.8 routing change touched (a failed <c>memoryview</c> is now caught inside Python,
        ///     the built-in adapters skip the exact numpy and builtin exporter types): ndarray subclasses (a user subclass,
        ///     <c>np.matrix</c>, a masked array's data) still view; an object that exposes only <c>__array_interface__</c> views
        ///     through it and refuses the copy with the buffer text; a released memoryview and an object that merely carries a
        ///     StringDType <c>dtype</c> attribute are refused with Python's own error; a StringDType ndarray SUBCLASS copies.
        /// </summary>
        [TestMethod]
        public void NonTextImports_AfterTheRoutingChange()
        {
            // The Python types are made with type(...): an instance attribute is enough for numpy's and the bridge's
            // __array_interface__ lookup, and the iface object needs no reference to its source (_strided stays in scope).
            PyExec(Helpers);
            PyExec("""
                Sub = type('Sub', (np.ndarray,), {})
                sub = np.arange(6.0).view(Sub)
                mat = np.matrix([[1.0, 2.0], [3.0, 4.0]])
                masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
                _strided = np.arange(8, dtype='i4')[::2]
                iface = type('IfaceOnly', (), {})()
                iface.__array_interface__ = _strided.__array_interface__
                released = memoryview(b'abc')
                released.release()
                fake = type('FakeT', (), {'dtype': np.dtype('T')})()
                tsub = np.array(['a', 'a long text over fifteen bytes'], dtype=StringDType()).view(Sub)
                """);
            using (NDArray v = ViewOf("sub"))
                ReadAt<double>(v, 5).Should().Be(5.0, "an ndarray subclass views");
            using (NDArray v = ViewOf("mat"))
                v.Shape.Dimensions.Should().Equal(new long[] { 2, 2 }, "np.matrix views as its 2-D data");
            using (NDArray v = ViewOf("masked"))
                ReadAt<double>(v, 2).Should().Be(3.0, "a masked array views its data");

            using (NDArray v = ViewOf("iface"))
            {
                v.Shape.Dimensions.Should().Equal(new long[] { 4 });
                ReadAt<int>(v, 1).Should().Be(2, "the __array_interface__ route reads the strided data");
                PyExec("_strided[1] = 77");
                ReadAt<int>(v, 1).Should().Be(77, "and shares it");
            }

            ((Action)(() => ImportOf("iface").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("the object does not export a PEP 3118 buffer*");
            ((Action)(() => ViewOf("released", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*released memoryview*");
            ((Action)(() => ImportOf("fake").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("the object does not export a PEP 3118 buffer*",
                    "a StringDType dtype attribute is not a numpy array: the isinstance guard stops the C API copy");

            using (NDArray copy = ImportOf("tsub"))
                copy.GetValue(1).Should().Be("a long text over fifteen bytes", "a StringDType ndarray subclass copies");

            PyExec("del sub, mat, masked, iface, _strided, released, fake, tsub");
        }

        // ============================================================================================ round 2

        /// <summary>
        ///     An object array of <c>str</c> SUBCLASS instances copies like any array of <c>str</c> (fixed with this test: it
        ///     was refused with a false UTF-8 message, because numpy's <c>coerce=False</c> cast takes only an exact
        ///     <c>str</c> and <c>np.str_</c>). The copy holds what numpy's coercing cast stores — <c>str(x)</c>, so a
        ///     subclass that overrides <c>__str__</c> crosses as that text, as in <c>np.array(arr, dtype=StringDType())</c>.
        ///     A non-<c>str</c> element after a subclass is still named; a subclass whose <c>__str__</c> raises is refused
        ///     with Python's error.
        /// </summary>
        [TestMethod]
        public void ObjectArrays_OfStrSubclasses_CopyAsNumpysCoercingCast()
        {
            PyExec(Helpers);
            PyExec("""
                Plain = type('Plain', (str,), {})
                Loud = type('Loud', (str,), {'__str__': lambda self: 'LOUD:' + str.__str__(self)})
                def _boom(self):
                    raise RuntimeError('no text for you')
                Broken = type('Broken', (str,), {'__str__': _boom})
                subs = np.array([Plain('ab'), Loud('cd'), np.str_('e\x00f'), 'a medium text over fifteen bytes', Plain('')], dtype=object)
                subs2d = np.array([[Plain('x'), 'y'], [Loud('z'), 'w']], dtype=object).T
                mixed = np.array([Plain('ok'), None], dtype=object)
                broken = np.array(['ok', Broken('b')], dtype=object)
                """);
            foreach (string name in new[] { "subs", "subs2d" })
            {
                using NDArray copy = ImportOf(name);
                copy.dtype.Should().Be(np.dtypes.StringDType(), name);
                copy.Shape.Dimensions.Should().Equal(PyDims(name), name);
                Encode(copy).Should().Equal(PyStrings($"_enc({name}.astype(StringDType()))"), $"{name}: numpy's coercing cast");
            }

            using (NDArray copy = ImportOf("subs"))
                copy.GetValue(1).Should().Be("LOUD:cd", "the __str__ override is what numpy's cast stores");
            ((Action)(() => ImportOf("mixed").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*NoneType at flat index 1 (None), which is not a str*");
            ((Action)(() => ImportOf("broken").Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("*only str elements, but numpy cannot store one of them as text*no text for you*");
            PyExec("del Plain, Loud, Broken, subs, subs2d, mixed, broken");
        }

        /// <summary>
        ///     A cell its own library cannot load stops the copy with a clear exception and leaves nothing behind: on import, a
        ///     numpy cell that claims an arena string in an array without an arena (<c>NpyString_load</c> fails); on export, the
        ///     same NumSharp cell (NumSharp's load fails). numpy's allocator is released (numpy keeps working on the array, so
        ///     its lock was not left held), no allocator / export / import leaks, and once the cell is restored both copies
        ///     succeed.
        /// </summary>
        [TestMethod]
        public unsafe void StringDType_CorruptCells_FailCleanly_BothWays()
        {
            long allocators = VStringAllocator.LiveAllocators;
            PyExec(Helpers);
            PyExec("cr = np.array(['a', 'b', 'c'], dtype=StringDType())");
            byte* numpyCell = (byte*)PyLong("cr.__array_interface__['data'][0]") + 16;
            byte[] saved = new ReadOnlySpan<byte>(numpyCell, 16).ToArray();
            CorruptToArenaString(numpyCell);
            try
            {
                ((Action)(() => ImportOf("cr").Dispose()))
                    .Should().Throw<InvalidOperationException>().WithMessage("numpy could not load a StringDType element*corrupt cell*");
            }
            finally
            {
                saved.CopyTo(new Span<byte>(numpyCell, 16));
            }

            PyStr("repr(np.strings.add(cr, '!'))").Should().Be("array(['a!', 'b!', 'c!'], dtype=StringDType())",
                "numpy's allocator was released: numpy acquires it again");
            using (NDArray copy = ImportOf("cr"))
                copy.GetValue(1).Should().Be("b");

            using NDArray m = np.array(new object[] { "x", "y", "z" }, np.dtypes.StringDType());
            byte* nsCell = (byte*)m.Storage.Address + 16;
            saved = new ReadOnlySpan<byte>(nsCell, 16).ToArray();
            CorruptToArenaString(nsCell);
            try
            {
                ((Action)(() =>
                {
                    using (Gil())
                    using (PyObject x = m.ToNumpyCopy())
                    {
                    }
                })).Should().Throw<InvalidOperationException>().WithMessage("NumSharp could not load a StringDType element*corrupt cell*");
            }
            finally
            {
                saved.CopyTo(new Span<byte>(nsCell, 16));
            }

            using (Gil())
            using (PyObject x = m.ToNumpyCopy())
                x.Repr().Should().Be("array(['x', 'y', 'z'], dtype=StringDType())");
            PyExec("del cr");
            WaitFor(() => { Pump(); return VStringAllocator.LiveAllocators <= allocators + 1; })
                .Should().BeTrue("only m's allocator is alive: the failed copies left none");
        }

        /// <summary>
        ///     Writes the header of a medium ARENA string (offset 0, size 20, flags <c>INITIALIZED</c>) over a 16-byte cell —
        ///     in an array whose strings are all short, so it has no arena, the load of that cell fails.
        /// </summary>
        /// <param name="cell">The cell's first byte.</param>
        private static unsafe void CorruptToArenaString(byte* cell)
        {
            new Span<byte>(cell, 16).Clear();
            cell[8] = 20;     // the size, little-endian bytes 8..14
            cell[15] = 0x40;  // INITIALIZED, inside the arena, medium
        }

        /// <summary>
        ///     Cells that need no arena and cells that need a huge one: <c>np.empty(5, T)</c> (all-zero cells, read as
        ///     <c>''</c>), 1,000 NA cells, 1,000 empty strings, and one 16 MiB string (an eight-byte size prefix far past any
        ///     other case) — each import equals numpy's own copy cell for cell, and each comes back.
        /// </summary>
        [TestMethod]
        public void StringDType_UninitializedEmptyNaAndHugeStrings()
        {
            PyExec(Helpers);
            PyExec("""
                ue = np.empty(5, dtype=StringDType(na_object=None))
                an = np.array([None] * 1000, dtype=StringDType(na_object=None))
                ae = np.array([''] * 1000, dtype=StringDType())
                big1 = np.array(['q' * (16 << 20), 's'], dtype=StringDType())
                """);
            using (NDArray copy = ImportOf("ue"))
                Encode(copy).Should().Equal(Enumerable.Repeat("s:", 5), "np.empty's all-zero cells read as ''");
            foreach (string name in new[] { "ue", "an", "ae", "big1" })
                AssertImportsAsNumpysCopyAndBack(name, name);
            using (NDArray copy = ImportOf("big1"))
                ((string)copy.GetValue(0)).Length.Should().Be(16 << 20);
            PyExec("del ue, an, ae, big1");
        }

        /// <summary>
        ///     numpy's dimension limit: 64-dimensional <c>StringDType</c> arrays (and a transposed, reversed view of one) copy
        ///     both ways, a 64-dimensional <c>U</c> array views both ways; a 65-dimensional NumSharp array of any dtype is
        ///     refused by numpy itself (its own text) with nothing exported or leaked.
        /// </summary>
        [TestMethod]
        public void Text_MaximumRank_AndTooManyDimensions()
        {
            PyExec(Helpers);
            PyExec("""
                d64 = np.array(['x', 'a long text over fifteen bytes', None], dtype=StringDType(na_object=None)).reshape((1,) * 63 + (3,))
                u64 = np.array(['ab', 'c', 'def'], dtype='U3').reshape((3,) + (1,) * 63)
                """);
            AssertImportsAsNumpysCopyAndBack("d64", "64-D");
            AssertImportsAsNumpysCopyAndBack("d64.T[::-1]", "64-D transposed and reversed");
            using (NDArray view = ViewOf("u64"))
            {
                view.ndim.Should().Be(64);
                view.GetValue(Enumerable.Repeat(0L, 64).Select((_, i) => i == 0 ? 2L : 0L).ToArray()).Should().Be("def");
                using (Gil())
                using (PyObject back = view.ToNumpy())
                {
                    Scope.Set("u64b", back);
                }
            }

            PyBool("u64b.ndim == 64 and np.shares_memory(u64b, u64) and (u64b == u64).all()").Should().BeTrue("the view comes back as the same memory");
            PyExec("del u64b");

            var shape65 = new Shape(Enumerable.Repeat(1L, 65).ToArray());
            using NDArray t65 = np.array(new object[] { "x" }, np.dtypes.StringDType()).reshape(shape65);
            using NDArray u65 = np.array(new[] { "xy" }).astype(np.dtype("U2")).reshape(shape65);
            using NDArray f65 = np.zeros(shape65, NPTypeCode.Double);
            foreach (var (name, nd) in new[] { ("T", t65), ("U2", u65), ("float64", f65) })
            {
                ((Action)(() =>
                {
                    using (Gil())
                    using (PyObject x = name == "T" ? nd.ToNumpyCopy() : nd.ToNumpy())
                    {
                    }
                })).Should().Throw<PythonException>().WithMessage("*maximum supported dimension for an ndarray is currently 64, found 65*", name);
            }

            PyExec("del d64, u64");
        }

        /// <summary>
        ///     A TEXT cell equal to a string missing-data object is not missing: numpy's <c>U → T</c> cast does not check for
        ///     the NA object, so <c>np.array(['NA', 'b'], 'U2').astype(StringDType(na_object='NA'))</c> holds the text
        ///     <c>'NA'</c>, while a store of <c>'NA'</c> makes a missing cell. Both kinds cross as they are, and the cells equal
        ///     numpy's own copy.
        /// </summary>
        [TestMethod]
        public void StringDType_TextEqualToStringNA_StaysText()
        {
            PyExec(Helpers);
            PyExec("""
                ut = np.array(['NA', 'b', 'NA'], dtype='U2').astype(StringDType(na_object='NA'))
                ut[2] = 'NA'
                ut_flags = [ut[i] is ut.dtype.na_object for i in range(3)]
                """);
            PyStr("str(ut_flags)").Should().Be("[False, False, True]", "the cast keeps text, the store makes NA");
            AssertImportsAsNumpysCopyAndBack("ut", "text vs NA");
            using NDArray copy = ImportOf("ut");
            copy.tobytes().AsSpan(15, 1)[0].Should().NotBe(copy.tobytes()[47], "cell 0 is text, cell 2 is NA");
            using (Gil())
            using (PyObject back = copy.ToNumpyCopy())
            {
                Scope.Set("utb", back);
            }

            PyStr("str([utb[i] is utb.dtype.na_object for i in range(3)])").Should().Be("[False, False, True]");
            PyExec("del ut, ut_flags, utb");
        }

        /// <summary>
        ///     The 4300-digit boundary of a Python int missing-data object, live: numpy refuses to build
        ///     <c>StringDType(na_object=10**4300)</c> (it takes the object's <c>str()</c>, which CPython refuses past 4300
        ///     digits) and NumSharp refuses with the same ValueError text, so neither side can hold one; a 4300-digit object
        ///     crosses both ways as a <see cref="BigInteger"/> with its NA cells.
        /// </summary>
        [TestMethod]
        public void PythonIntNA_4300DigitBoundary_MatchesNumpy()
        {
            PyExec(Helpers);
            PyExec("""
                ok = 10**4300 - 1
                try:
                    StringDType(na_object=10**4300)
                    refusal = None
                except ValueError as e:
                    refusal = str(e)
                bnd = np.array(['a', ok, 'a long text over fifteen bytes'], dtype=StringDType(na_object=ok))
                """);
            string numpyRefusal = PyStr("refusal");
            numpyRefusal.Should().StartWith("Exceeds the limit (4300 digits) for integer string conversion");
            ((Action)(() => np.dtypes.StringDType(BigInteger.Pow(10, 4300)))).Should().Throw<ValueError>().WithMessage(numpyRefusal);

            BigInteger ok = BigInteger.Pow(10, 4300) - 1;
            using NDArray copy = ImportOf("bnd");
            copy.dtype.na_object.Should().Be(ok);
            copy.GetValue(1).Should().Be(ok, "the NA cell reads as the NA object");
            copy.tobytes().Should().Equal(PyBytes("np.array(bnd, copy=True).tobytes()"));
            using (Gil())
            using (PyObject back = copy.ToNumpyCopy())
            {
                Scope.Set("bndb", back);
            }

            PyBool("bndb.dtype.na_object == ok and bndb[1] is bndb.dtype.na_object and bndb[2] == bnd[2]").Should().BeTrue();
            PyExec("del ok, refusal, bnd, bndb");
        }

        /// <summary>
        ///     <c>FromArrayLike</c> over text: numpy's <c>np.asarray</c> makes a list of <c>str</c> a <c>U</c> array, a nested
        ///     list a 2-D one and a Python <c>str</c> a 0-d <c>U</c>, all copied; a list with <c>None</c> becomes an object
        ///     array, refused with the element named.
        /// </summary>
        [TestMethod]
        public void FromArrayLike_OfText()
        {
            using (Gil())
            {
                using PyObject list = Scope.Eval("['a', 'bc', 'def']");
                using (NDArray a = list.FromArrayLike())
                {
                    a.dtype.Should().Be(np.dtype("U3"));
                    a.GetValue(2).Should().Be("def");
                }

                using PyObject nested = Scope.Eval("[['a', 'bb'], ['c', 'dd']]");
                using (NDArray a = nested.FromArrayLike())
                {
                    a.Shape.Dimensions.Should().Equal(new long[] { 2, 2 });
                    a.GetValue(1L, 1L).Should().Be("dd");
                }

                using PyObject text = Scope.Eval("'hello'");
                using (NDArray a = text.FromArrayLike())
                {
                    a.ndim.Should().Be(0);
                    a.dtype.Should().Be(np.dtype("U5"));
                    a.GetValue().Should().Be("hello");
                }

                using PyObject withNone = Scope.Eval("['a', None]");
                ((Action)(() => withNone.FromArrayLike().Dispose()))
                    .Should().Throw<NotSupportedException>().WithMessage("*NoneType at flat index 1 (None)*");
            }
        }
    }
}
