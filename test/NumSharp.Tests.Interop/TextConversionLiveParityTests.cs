using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     End-to-end text parity through the pythonnet bridge for everything around the text operations: text ↔ number
    ///     casts (every parse rule numpy applies, element by element, and every number's text form), printing
    ///     (<c>repr</c> / <c>str</c> / <c>array2string</c>), <c>.npy</c> files written by one side and read by the other,
    ///     writes into numpy-born text through NumSharp's view of it, and the numeric neighbours a text operand meets
    ///     (reductions, products — the OpenBLAS backend is enabled in this process — <c>linalg</c>, arithmetic ufuncs) and
    ///     must refuse with numpy's class and text.
    /// </summary>
    /// <remarks>
    ///     Each family runs through <see cref="TextLiveParityHarness"/>: numpy builds the input, NumSharp computes over its
    ///     view or copy, NumSharp's result crosses back, and numpy's own answer over the same input is the expected one.
    /// </remarks>
    [TestClass]
    public class TextConversionLiveParityTests : TextLiveParityHarness
    {
        /// <summary>
        ///     The parse sources: 24 texts per text class that exercise numpy's text → number rules — signs, surrounding
        ///     spaces, underscores, hexadecimal and binary prefixes (refused), exponents, <c>nan</c> / <c>inf</c>, the
        ///     int64 edge and one past it, underflow and overflow of a float, words, the empty text, a non-ASCII decimal
        ///     digit (<c>٣</c>), a vulgar fraction (<c>½</c>), <c>True</c>; the NA source holds an NA in place of the
        ///     empty text. Non-ASCII characters are built with <c>chr</c>, so the source survives any re-encoding.
        /// </summary>
        private const string ParseSourcesPython = """
            _N = ['0', '1', '-7', '+42', ' 12 ', '1_000', '0x10', '1e3', '1.5', 'nan', '-inf', 'inf',
                  '9223372036854775807', '9223372036854775808', '-0', '0.1', '1e-400', '1e400', 'abc', '',
                  chr(0x663), chr(0xbd), 'True', '0b1']
            num_U = np.array(_N, dtype='U24')
            num_S = np.array([s.encode('utf-8') for s in _N], dtype='S24')
            num_T = np.array(_N, dtype=StringDType())
            num_TN = np.array(_N, dtype=StringDType(na_object=None))
            num_TN[19] = None

            def _t2c(x):
                # numpy 2.4.2's StringDType -> complex cast wrote the parsed REAL part into the imaginary part too
                # (stringdtype/casts.cpp string_to_complex_float: npy_csetimagfunc(out, complex_value.real)), so '3' was
                # 3+3j; the cast means complex(text), which NumSharp gives and numpy 2.5 computes. No text here has an
                # imaginary part, so the reference is numpy's own cast with the imaginary part set to complex(text).imag =
                # +0.0, which under numpy 2.5 changes nothing.
                r = x.astype('complex128')
                r.imag = 0.0
                return r
            """;

        /// <summary>
        ///     The number sources whose text forms are compared: float64 values that need the shortest round-trip repr
        ///     (<c>0.1</c>, <c>1/3</c>), switch to exponent form (<c>1e16</c>, <c>1e-5</c>), sit at the extremes
        ///     (subnormal, max) or are special (<c>-0.0</c>, NaN, ±inf); float32 / float16 values whose own shortest repr
        ///     differs from the float64 one; the integer extremes of int64 / uint64 / int8; bool; complex128 with NaN, inf
        ///     and signed zeros in each part.
        /// </summary>
        private const string NumberSourcesPython = """
            num_f64 = np.array([0.1, 1 / 3, 1e16, 1e-5, 1e22, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1.7976931348623157e308,
                                123456789.123, 2.0 ** 53, 0.5, 100.0, 1e-7, 123456789012345678.0, -1.5e-10])
            num_f32 = np.array([0.1, 1 / 3, 16777217.0, 1e-5, 3.4028235e38, -0.0, np.nan, np.inf, 1e-45, 0.3], dtype=np.float32)
            num_f16 = np.array([0.1, 65504, 6e-8, -0.0, np.nan, np.inf, 1 / 3], dtype=np.float16)
            num_i64 = np.array([0, -1, 9223372036854775807, -9223372036854775808, 42], dtype=np.int64)
            num_u64 = np.array([0, 18446744073709551615, 7], dtype=np.uint64)
            num_i8 = np.array([-128, 127, 0], dtype=np.int8)
            num_b = np.array([True, False])
            num_c128 = np.array([1 + 2j, complex(np.nan, np.inf), complex(-0.0, -0.0), 1e300 + 1e-300j, 0.1 - 0.2j])
            """;

        /// <summary>The numeric dtypes every parse cell targets.</summary>
        private static readonly string[] ParseTargets =
            { "int8", "uint8", "int16", "int32", "int64", "uint64", "float16", "float32", "float64", "complex128", "bool" };

        /// <summary>
        ///     A parse operation: the cast to <paramref name="target"/>, where numpy's reference for a <c>StringDType</c>
        ///     input cast to complex128 is <c>_t2c</c> (numpy 2.4.2 copies the real part into the imaginary part; NumSharp
        ///     parses <c>complex(text)</c>, pinned by
        ///     <c>StringDTypeOrderTests.SortComplex_ParsesTheText_RealPartOnly</c> and the np-layer dtype plan).
        /// </summary>
        /// <param name="target">The numpy dtype name.</param>
        /// <returns>The operation.</returns>
        private static TextOp ParseOp(string target)
            => new($"astype {target}",
                cls => target == "complex128" && cls is TextClass.StringDType or TextClass.StringDTypeNA
                    ? "_t2c(x)"
                    : $"x.astype('{target}')",
                (_, x, _) => x.astype(np.dtype(target)),
                Elementwise: false);

        /// <summary>
        ///     Whether this process's numpy runs on Windows, where a C <c>long</c> is 32 bits: numpy's text → integer
        ///     overflow messages there name the C long ("Python int too large to convert to C long" for an int8 target,
        ///     "int too big to convert" for int64, "… out of bounds for long" for a StringDType → int32), where LP64
        ///     numpy (Linux, macOS — NumSharp's model of a C long, measured identical in WSL's numpy 2.4.2) names the
        ///     target ("Python integer … out of bounds for int8", "… out of bounds for int").
        /// </summary>
        /// <returns>True on Windows.</returns>
        private bool NumpyHasLlp64Long() => PyStr("__import__('sys').platform") == "win32";

        /// <summary>
        ///     Text → number, ELEMENT BY ELEMENT (a whole-array cast stops at its first refusal; one element per cell
        ///     shows numpy's rule for each text): every one of the 24 parse texts of every text class, through both routes,
        ///     cast to eleven numeric dtypes — the value (a NaN's bits included), or numpy's exception class and text
        ///     (<c>ValueError</c> for an unparsable text, <c>OverflowError</c> past the target's range).
        /// </summary>
        /// <remarks>
        ///     On a Windows numpy the overflow TEXT of the two int64-edge texts (elements 12 and 13) into the integer
        ///     targets is the C long's (<see cref="NumpyHasLlp64Long"/>); those cells keep the class check (OverflowError)
        ///     and waive the text, which LP64 numpy — Linux and macOS CI — still compares exactly.
        /// </remarks>
        [TestMethod]
        public void TextToNumbers_EveryElementAndTarget_MatchNumpy()
        {
            DefineSources(ParseSourcesPython);
            var ops = ParseTargets.Select(ParseOp).ToArray();
            var elements = Enumerable.Range(0, 24).Select(i => ($"element {i}", $"src[{i}, ...]")).ToArray();
            var inputs = new[]
            {
                new FamilyInput("U24", TextClass.Unicode, "num_U", new[] { "view", "copy" }),
                new FamilyInput("S24", TextClass.Bytes, "num_S", new[] { "view", "copy" }),
                new FamilyInput("StringDType", TextClass.StringDType, "num_T", new[] { "copy" }),
                new FamilyInput("StringDType(na_object=None)", TextClass.StringDTypeNA, "num_TN", new[] { "copy" }),
            };
            bool llp64 = NumpyHasLlp64Long();
            RunFamily("parse", ops, inputs, elements,
                    messageGap: label => llp64 && (label.Contains(" | element 12 | ") || label.Contains(" | element 13 | ")) &&
                                         (label.Contains(" | astype int") || label.Contains(" | astype uint")))
                .Should().Be(24 * 6 * ParseTargets.Length);
        }

        /// <summary>
        ///     Text → number over whole arrays of the texts every target accepts: the same values as the element cells,
        ///     now through numpy's strided cast loop over each layout (a view's strides, a reversed or broadcast input).
        /// </summary>
        [TestMethod]
        public void TextToNumbers_WholeArrays_EveryLayout_MatchNumpy()
        {
            DefineSources(ParseSourcesPython + "\nok_U = num_U[[0, 1, 2, 3, 4, 14]]\nok_S = num_S[[0, 1, 2, 3, 4, 14]]\n" +
                          "ok_T = num_T[[0, 1, 2, 3, 4, 14]]\n");
            var layouts = new[] { ("C 1-D", "src"), ("reversed", "src[::-1]"), ("stepped", "src[1::2]"),
                                  ("broadcast", "np.broadcast_to(src[:3], (2, 3))"), ("0-d", "src[3, ...]"), ("empty", "src[:0]") };
            var inputs = new[]
            {
                new FamilyInput("U24", TextClass.Unicode, "ok_U", new[] { "view", "copy" }),
                new FamilyInput("S24", TextClass.Bytes, "ok_S", new[] { "view", "copy" }),
                new FamilyInput("StringDType", TextClass.StringDType, "ok_T", new[] { "copy" }),
            };
            var ops = new[] { "int8", "int64", "uint64", "float32", "float64", "complex128" }.Select(ParseOp).ToArray();
            RunFamily("parse arrays", ops, inputs, layouts).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Number → text: every number source (float64 / float32 / float16 / int64 / uint64 / int8 / bool /
        ///     complex128), each layout and route, cast to unsized <c>U</c> (numpy's per-type width: 32 for a float, 21
        ///     for int64, 5 for bool, 64 for complex128), to a too-narrow <c>U4</c> (truncation), to a wide <c>U40</c>,
        ///     to <c>S</c> / <c>S3</c>, and to <c>StringDType</c> — the text of each value is numpy's shortest
        ///     round-trip repr (<c>0.1</c>, <c>1e+16</c>, <c>-0.0</c>, <c>nan</c>, <c>(nan+infj)</c>).
        /// </summary>
        [TestMethod]
        public void NumbersToText_EveryDtypeLayoutAndRoute_MatchNumpy()
        {
            DefineSources(NumberSourcesPython);
            string[] sources = { "num_f64", "num_f32", "num_f16", "num_i64", "num_u64", "num_i8", "num_b", "num_c128" };
            var inputs = sources.Select(s => new FamilyInput(s, TextClass.Unicode, s, new[] { "view", "copy" }));
            var layouts = new[] { ("C 1-D", "src"), ("reversed", "src[::-1]"), ("0-d", "src[1, ...]"), ("empty", "src[:0]") };
            RunFamily("format", new[]
            {
                W("astype U", "x.astype('U')", (x, _) => x.astype(np.dtype("U"))),
                W("astype U4", "x.astype('U4')", (x, _) => x.astype(np.dtype("U4"))),
                W("astype U40", "x.astype('U40')", (x, _) => x.astype(np.dtype("U40"))),
                W("astype S", "x.astype('S')", (x, _) => x.astype(np.dtype("S"))),
                W("astype S3", "x.astype('S3')", (x, _) => x.astype(np.dtype("S3"))),
                W("astype StringDType()", "x.astype(StringDType())", (x, _) => x.astype(np.dtypes.StringDType())),
                W("astype StringDType(na_object=None)", "x.astype(StringDType(na_object=None))",
                    (x, _) => x.astype(np.dtypes.StringDType((object)null))),
            }, inputs, layouts).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Printing: <c>repr</c>, <c>str</c>, <c>array2string</c> and a narrow <c>max_line_width</c> of every text class
        ///     and layout — the quoting of each class (<c>'..'</c>, <c>b'..'</c>, an NA as <c>None</c>), the dtype
        ///     suffix (<c>dtype='&lt;U12'</c>, <c>dtype='|S12'</c>, <c>dtype=StringDType(na_object=None)</c>), escapes of
        ///     tabs, NULs and high bytes, line wrapping, the empty and 0-d forms, and a 1200-element array's summary
        ///     (<c>...</c>).
        /// </summary>
        [TestMethod]
        public void Printing_EveryClassAndLayout_MatchesNumpysText()
        {
            DefineSources();
            var layouts = Layouts.Concat(new[] { ("1200 elements (summarized)", "np.repeat(src, 50)") });
            RunFamily("print", new[]
            {
                W("repr", "np.array(repr(x))", (x, _) => np.array(np.array_repr(x))),
                W("str", "np.array(str(x))", (x, _) => np.array(np.array_str(x))),
                W("array2string", "np.array(np.array2string(x))", (x, _) => np.array(np.array2string(x))),
                W("array_repr max_line_width=30", "np.array(np.array_repr(x, max_line_width=30))",
                    (x, _) => np.array(np.array_repr(x, max_line_width: 30))),
            }, layouts: layouts).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     <c>.npy</c> both ways for every text class and layout: a file NumSharp's <c>np.save</c> writes holds the
        ///     bytes numpy's own <c>np.save</c> writes for the same array (fixed-width text: header, order flag and data;
        ///     <c>StringDType</c>: numpy's pickle, read back by <c>np.load(allow_pickle=True)</c> with the same descriptor
        ///     and values), and a file numpy writes reads back in NumSharp as numpy's array.
        /// </summary>
        [TestMethod]
        public void Npy_SaveAndLoad_BothDirections_MatchNumpy()
        {
            DefineSources();
            string dir = Path.Combine(Path.GetTempPath(), "ns-text-npy-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            var failures = new List<string>();
            int cells = 0;
            try
            {
                var layouts = new[] { ("C 1-D", "src"), ("C 2-D", "src.reshape(4, 6)"), ("F 2-D", "np.asfortranarray(src.reshape(4, 6))"),
                                      ("transposed", "src.reshape(4, 6).T"), ("reversed", "src[::-1]"), ("0-d", "src[7, ...]"),
                                      ("empty 2-D", "src.reshape(4, 6)[:, 2:2]") };
                foreach (TextClass cls in AllClasses)
                foreach (var (layoutName, layoutExpr) in layouts)
                foreach (string route in RoutesOf(cls))
                {
                    cells++;
                    string cell = $"npy | {cls} | {layoutName} | {route}";
                    string nsPath = Path.Combine(dir, $"ns_{cells}.npy").Replace('\\', '/');
                    string npPath = Path.Combine(dir, $"np_{cells}.npy").Replace('\\', '/');
                    // p is the harness's partner name; Judge's numpy expression ("x") reads only x.
                    PyExec($"x = {layoutExpr.Replace("src", SourceName(cls))}\np = x\nnp.save('{npPath}', x)");
                    using NDArray xs = route == "view" ? ViewOf("x", allowReadonly: true) : ImportOf("x");

                    // NumSharp writes; numpy reads NumSharp's file and must find x. Fixed-width text must also be the
                    // very bytes numpy's np.save writes for the SAME array — the array NumSharp holds, seen by numpy
                    // through the bridge: an import copy is C-contiguous by the bridge's contract (ToNDArray), so its
                    // file legitimately says fortran_order False where numpy's own F-ordered x says True. A
                    // StringDType pickle is held to its loaded content instead (the pickle's opcodes are each
                    // writer's implementation detail).
                    np.save(nsPath, xs);
                    string verdict = PyStr($"_cmp(np.load('{nsPath}', allow_pickle=True), x)");
                    if (verdict.Length > 0)
                        failures.Add($"{cell} | numpy reads NumSharp's file: {verdict}");
                    if (cls is TextClass.Unicode or TextClass.Bytes)
                    {
                        string samePath = Path.Combine(dir, $"same_{cells}.npy").Replace('\\', '/');
                        ExportTo("xv", xs);
                        PyExec($"np.save('{samePath}', xv)\ndel xv");
                        if (!File.ReadAllBytes(nsPath).SequenceEqual(File.ReadAllBytes(samePath)))
                            failures.Add($"{cell} | NumSharp's file is not byte-identical to numpy's np.save of the same array");
                    }

                    // numpy writes; NumSharp reads numpy's file and must hold x.
                    try
                    {
                        using NDArray loaded = (NDArray)np.load(npPath, allow_pickle: true);
                        string back = Judge("x", false, loaded, null);
                        if (back.Length > 0)
                            failures.Add($"{cell} | NumSharp reads numpy's file: {back}");
                    }
                    catch (Exception e)
                    {
                        failures.Add($"{cell} | NumSharp cannot read numpy's file: {e.GetType().Name}: {e.Message}");
                    }
                }
            }
            finally
            {
                PyExec("x = None");
                Directory.Delete(dir, recursive: true);
            }

            AssertNoFailures("npy", cells, failures);
        }

        /// <summary>
        ///     Writes through the bridge: NumSharp assigns into its zero-copy VIEW of a numpy-born <c>U</c> / <c>S</c> array,
        ///     numpy performs the same assignment on a twin, and the numpy array NumSharp wrote through must equal the twin
        ///     — a longer text truncated to the width, trailing NULs (stored, read back stripped), the empty text, an int /
        ///     float / bool coerced to its text, bytes into <c>U</c> and text into <c>S</c> (ASCII-encoded; non-ASCII raises
        ///     numpy's <c>UnicodeEncodeError</c> and leaves the array unchanged), a slice from a list, and a fill.
        /// </summary>
        [TestMethod]
        public void Writes_ThroughNumSharpsView_MatchNumpysAssignment()
        {
            DefineSources();
            var failures = new List<string>();
            int cells = 0;
            var assignments = new (string Name, Func<TextClass, string> Numpy, Action<TextClass, NDArray> Sharp)[]
            {
                ("long text", c => $"w[0] = {Q(c, "ABCDEFGHIJKLMNOPQRSTU")}", (c, w) => w["0"] = L(c, "ABCDEFGHIJKLMNOPQRSTU")),
                ("trailing NULs", c => $"w[0] = {Q(c, "ab\0\0")}", (c, w) => w["0"] = L(c, "ab\0\0")),
                ("empty", c => $"w[0] = {Q(c, "")}", (c, w) => w["0"] = L(c, "")),
                ("int 12", _ => "w[0] = 12", (_, w) => w["0"] = 12),
                ("float 1.5", _ => "w[0] = 1.5", (_, w) => w["0"] = 1.5),
                ("bool True", _ => "w[0] = True", (_, w) => w["0"] = true),
                ("bytes into the array", _ => "w[0] = b'by'", (_, w) => w["0"] = np.array("by", np.dtype("S"))),
                ("ASCII text into the array", _ => "w[0] = 'tx'", (_, w) => w["0"] = (NDArray)"tx"),
                ("non-ASCII text into the array", _ => $"w[0] = {PyQuote("\u00e9!")}", (_, w) => w["0"] = (NDArray)"\u00e9!"),
                ("slice from a list", c => $"w[1:3] = [{Q(c, "q")}, {Q(c, "rr")}]",
                    (c, w) => w["1:3"] = c == TextClass.Bytes ? np.array(new[] { "q", "rr" }, np.dtype("S")) : np.array(new[] { "q", "rr" })),
                ("fill", c => $"w[...] = {Q(c, "z")}", (c, w) => w["..."] = L(c, "z")),
            };
            foreach (TextClass cls in new[] { TextClass.Unicode, TextClass.Bytes })
            foreach (var (name, numpy, sharp) in assignments)
            {
                cells++;
                string cell = $"write | {cls} | {name}";
                PyExec($"w = {SourceName(cls)}.copy()\nw0 = w.copy()\ny = w.copy()");
                Exception error = null;
                using (NDArray ws = ViewOf("w"))
                {
                    try
                    {
                        sharp(cls, ws);
                    }
                    catch (Exception e)
                    {
                        error = e;
                    }
                }

                // numpy's twin performs the same assignment; numpy's raise must be NumSharp's, and a refused write
                // must leave numpy's memory (seen through NumSharp's view) as it was.
                string twin = numpy(cls).Replace("w[", "y[");
                PyExec($"try:\n    {twin}\n    _werr = None\nexcept Exception as e:\n    _werr = (','.join(c.__name__ for c in type(e).__mro__), str(e))");
                bool numpyRaised = PyBool("_werr is not None");
                string verdict;
                if (numpyRaised && error is null)
                    verdict = $"numpy raises {PyStr("_werr[0]")}: {PyStr("_werr[1]")} -- NumSharp wrote";
                else if (!numpyRaised && error is not null)
                    verdict = $"NumSharp raises {error.GetType().Name}: {error.Message} -- numpy wrote";
                else if (numpyRaised)
                    verdict = CompareErrors(PyStr("_werr[0]"), PyStr("_werr[1]"), error) + PyStr("_cmp(w, w0)");
                else
                    verdict = PyStr("_cmp(w, y)");
                if (verdict.Length > 0)
                    failures.Add($"{cell}: {verdict}");
            }

            PyExec("w = w0 = y = None");
            AssertNoFailures("write", cells, failures);
        }

        /// <summary>
        ///     The numeric neighbours a text operand meets — reductions (<c>sum</c>, an axis <c>sum</c>, <c>prod</c>,
        ///     <c>cumsum</c>, <c>mean</c>), products (<c>dot</c>, <c>matmul</c>, <c>inner</c>, <c>vdot</c>, <c>outer</c>,
        ///     <c>trace</c>; the OpenBLAS backend is enabled in this process and declines text, so the answer is
        ///     NumSharp's own), <c>linalg</c> (<c>norm</c>, <c>det</c>) and arithmetic ufuncs (<c>add</c>,
        ///     <c>multiply</c> by an int, <c>subtract</c>, <c>negative</c>, <c>isnan</c>): numpy computes some (a
        ///     <c>StringDType</c> <c>sum</c> concatenates, <c>add</c> concatenates every class, <c>multiply</c> repeats)
        ///     and refuses the rest — NumSharp must do the same, with numpy's class and text.
        /// </summary>
        [TestMethod]
        public void NumericNeighbours_OfTextOperands_MatchNumpy()
        {
            DefineSources();
            RunFamily("neighbours", new[]
            {
                W("sum", "np.sum(x)", (x, _) => np.sum(x)),
                W("sum axis=0", "np.sum(x, axis=0)", (x, _) => np.sum(x, axis: 0)),
                W("prod", "np.prod(x)", (x, _) => np.prod(x)),
                W("cumsum", "np.cumsum(x)", (x, _) => np.cumsum(x)),
                W("mean", "np.mean(x)", (x, _) => np.mean(x)),
                W("dot", "np.dot(x, p)", (x, p) => np.dot(x, p)),
                W("matmul", "np.matmul(x, p)", (x, p) => np.matmul(x, p)),
                W("inner", "np.inner(x, p)", (x, p) => np.inner(x, p)),
                W("vdot", "np.vdot(x, p)", (x, p) => np.vdot(x, p)),
                W("outer", "np.outer(x, p)", (x, p) => np.outer(x, p)),
                W("trace", "np.trace(x)", (x, _) => np.trace(x)),
                W("linalg.norm", "np.linalg.norm(x)", (x, _) => np.linalg.norm(x)),
                W("linalg.det", "np.linalg.det(x)", (x, _) => np.linalg.det(x)),
                W("linalg.inv", "np.linalg.inv(x)", (x, _) => np.linalg.inv(x)),
                W("linalg.slogdet", "np.linalg.slogdet(x)[0]", (x, _) => np.linalg.slogdet(x).sign),
                W("linalg.solve", "np.linalg.solve(x, p)", (x, p) => np.linalg.solve(x, p)),
                W("linalg.matrix_power", "np.linalg.matrix_power(x, 2)", (x, _) => np.linalg.matrix_power(x, 2)),
                W("linalg.cholesky", "np.linalg.cholesky(x)", (x, _) => np.linalg.cholesky(x)),
                W("linalg.eigvals", "np.linalg.eigvals(x)", (x, _) => np.linalg.eigvals(x)),
                W("linalg.svd", "np.linalg.svd(x)[1]", (x, _) => np.linalg.svd(x).S),
                W("linalg.qr", "np.linalg.qr(x)[1]", (x, _) => np.linalg.qr(x).R),
                W("linalg.pinv", "np.linalg.pinv(x)", (x, _) => np.linalg.pinv(x)),
                E2("np.add", "np.add(x, p)", (x, p) => np.add(x, p)),
                E("np.multiply by 2", "np.multiply(x, 2)", x => np.multiply(x, 2)),
                E2("np.subtract", "np.subtract(x, p)", (x, p) => np.subtract(x, p)),
                E("np.negative", "np.negative(x)", x => np.negative(x)),
                E("np.isnan", "np.isnan(x)", x => np.isnan(x)),
            }).Should().BeGreaterThan(0);
        }
    }
}
