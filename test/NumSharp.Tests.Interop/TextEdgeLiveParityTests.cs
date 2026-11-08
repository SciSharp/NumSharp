using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The EDGES of end-to-end text parity through the pythonnet bridge, each against LIVE numpy over the same memory
    ///     (<see cref="TextLiveParityHarness"/>): the export direction (NumSharp-born layouts — the split child's offset,
    ///     the broadcast, the F order — handed to numpy as views), operands of two different text classes, widths or NA
    ///     objects (numpy's promotion and its refusals), broadcasting shapes, the ufunc keywords (<c>out=</c>,
    ///     <c>where=</c>, <c>dtype=</c>, and the identity of the object returned for <c>out=</c>), every
    ///     <c>StringDType</c> NA kind (a NaN-like NA, a string NA, <c>None</c>, <c>coerce=False</c>), <c>numpy.char</c>
    ///     and its chararray (the trailing-whitespace comparisons), <c>partition</c> / <c>mod</c> / <c>translate</c>,
    ///     fused <c>np.evaluate</c> text expressions against numpy's composition, and numpy's own functions called with
    ///     NumSharp arrays through the registered codec.
    /// </summary>
    /// <remarks>
    ///     Each test is one family: it runs every cell and fails once with the list of every disagreement.
    /// </remarks>
    [TestClass]
    public class TextEdgeLiveParityTests : TextLiveParityHarness
    {
        // ------------------------------------------------------------------------------------------------------------
        // The export direction
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Elementwise operations in the EXPORT direction: NumSharp owns the text, makes each of its own layouts
        ///     (<see cref="TextLiveParityHarness.NumSharpLayouts"/>: the <c>np.split</c> child at a non-zero offset, the
        ///     broadcast, F order, the transpose, a step to the last element, reversed rows, 0-d, empty), hands it to numpy
        ///     (a view of a fixed-width array, a copy of a <c>StringDType</c> one), and numpy's answer over what it was handed
        ///     is held to NumSharp's over its own array — so a wrong stride, offset or flag in the export fails a cell.
        /// </summary>
        [TestMethod]
        public void ExportDirection_ElementwiseOps_MatchNumpy()
        {
            DefineSources();
            RunFamilyFromNumSharp("export elementwise", new[]
            {
                E("upper", "np.strings.upper(x)", x => np.strings.upper(x)),
                E("swapcase", "np.strings.swapcase(x)", x => np.strings.swapcase(x)),
                E("isalpha", "np.strings.isalpha(x)", x => np.strings.isalpha(x)),
                E("isnumeric", "np.strings.isnumeric(x)", x => np.strings.isnumeric(x)),
                E("str_len", "np.strings.str_len(x)", x => np.strings.str_len(x)),
                E("strip", "np.strings.strip(x)", x => np.strings.strip(x)),
                E("center 14", "np.strings.center(x, 14)", x => np.strings.center(x, 14)),
                Lit("find 'l'", "np.strings.find(x, {0})", "l", (x, s) => np.strings.find(x, s)),
                Lit("count ''", "np.strings.count(x, {0})", "", (x, s) => np.strings.count(x, s)),
                Lit2("replace l LL", "np.strings.replace(x, {0}, {1})", "l", "LL", (x, a, b) => np.strings.replace(x, a, b)),
                E("slice ::-1", "np.strings.slice(x, None, None, -1)", x => np.strings.slice(x, null, null, -1)),
                E2("add x p", "np.strings.add(x, p)", (x, p) => np.strings.add(x, p)),
                E("multiply 2", "np.strings.multiply(x, 2)", x => np.strings.multiply(x, 2)),
                E2("less", "x < p", (x, p) => x < p),
                E2("equal", "np.equal(x, p)", (x, p) => np.equal(x, p)),
                E("encode utf-8", "np.strings.encode(x, 'utf-8')", x => np.strings.encode(x, "utf-8")),
                E("decode latin-1", "np.strings.decode(x, 'latin-1')", x => np.strings.decode(x, "latin-1")),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Ordering, shape routines and casts in the EXPORT direction (see
        ///     <see cref="ExportDirection_ElementwiseOps_MatchNumpy"/>): sort, stable argsort, unique, searchsorted, the
        ///     extrema with keepdims, concatenate, where, take, and the text casts.
        /// </summary>
        [TestMethod]
        public void ExportDirection_OrderShapeAndCasts_MatchNumpy()
        {
            DefineSources();
            RunFamilyFromNumSharp("export order", new[]
            {
                W("sort", "np.sort(x)", (x, _) => np.sort(x)),
                W("argsort stable", "np.argsort(x, kind='stable')", (x, _) => np.argsort(x, kind: "stable")),
                W("unique", "np.unique(x)", (x, _) => np.unique(x).values),
                W("searchsorted", "np.searchsorted(np.sort(x, axis=None), x)", (x, _) => np.searchsorted(np.sort(x, axis: null), x)),
                W("max keepdims", "np.max(x, keepdims=True)", (x, _) => np.max(x, keepdims: true)),
                W("concatenate", "np.concatenate((x.ravel(), p.ravel()))", (x, p) => np.concatenate((x.ravel(), p.ravel()))),
                W("where", "np.where(np.strings.str_len(x) > 3, x, p)", (x, p) => np.where(np.strings.str_len(x) > 3, x, p)),
                W("take [0, -1]", "np.take(x, [0, -1])", (x, _) => np.take(x, np.array(new long[] { 0, -1 }))),
                W("astype U3", "x.astype('U3')", (x, _) => x.astype(np.dtype("U3"))),
                W("astype StringDType()", "_eager(x.astype(StringDType()))", (x, _) => x.astype(np.dtypes.StringDType())),
                W("copy order=F", "x.copy(order='F')", (x, _) => x.copy('F')),
            }).Should().BeGreaterThan(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Two operands: classes, NA objects, shapes
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The NA-kind sources: <c>StringDType(na_object=nan)</c> (a NaN-like NA), <c>StringDType(na_object='NA')</c> (a
        ///     string NA: it behaves as its text), each with NA cells at 4 and 17, and <c>StringDType(coerce=False)</c>.
        /// </summary>
        private const string NAKindSourcesPython = """
            src_Tnan = np.array(_U, dtype=StringDType(na_object=np.nan))
            src_Tnan[4] = src_Tnan.dtype.na_object
            src_Tnan[17] = src_Tnan.dtype.na_object
            src_Tstr = np.array(_U, dtype=StringDType(na_object='NA'))
            src_Tstr[4] = src_Tstr.dtype.na_object
            src_Tstr[17] = src_Tstr.dtype.na_object
            src_Tnc = np.array(_U, dtype=StringDType(coerce=False))
            """;

        /// <summary>The binary text operations every operand pair is put through.</summary>
        private static readonly TextOp[] PairOps =
        {
            E2("add", "np.strings.add(x, p)", (x, p) => np.strings.add(x, p)),
            E2("equal", "np.equal(x, p)", (x, p) => np.equal(x, p)),
            E2("not_equal", "x != p", (x, p) => x != p),
            E2("less", "np.less(x, p)", (x, p) => np.less(x, p)),
            E2("greater_equal", "x >= p", (x, p) => x >= p),
            E2("maximum", "np.maximum(x, p)", (x, p) => np.maximum(x, p)),
            E2("minimum", "np.minimum(x, p)", (x, p) => np.minimum(x, p)),
            E2("find", "np.strings.find(x, p)", (x, p) => np.strings.find(x, p)),
            E2("count", "np.strings.count(x, p)", (x, p) => np.strings.count(x, p)),
            E2("startswith", "np.strings.startswith(x, p)", (x, p) => np.strings.startswith(x, p)),
            E2("endswith", "np.strings.endswith(x, p)", (x, p) => np.strings.endswith(x, p)),
            W("concatenate", "np.concatenate((x.ravel(), p.ravel()))", (x, p) => np.concatenate((x.ravel(), p.ravel()))),
            W("where", "np.where(np.arange(x.size).reshape(x.shape) % 2 == 0, x, p)",
                (x, p) => np.where((np.arange(x.size).reshape(x.Shape) % 2) == 0, x, p)),
        };

        /// <summary>
        ///     Operands of two different text classes and NA objects: every ordered pair of <c>U12</c>, <c>S12</c>,
        ///     <c>StringDType()</c>, <c>StringDType(na_object=None)</c>, <c>StringDType(na_object=nan)</c> and
        ///     <c>StringDType(na_object='NA')</c> — numpy promotes <c>U</c> / <c>StringDType</c> mixes to <c>StringDType</c>
        ///     (taking the NA object of the side that has one), refuses <c>U</c> with <c>S</c> and <c>S</c> with
        ///     <c>StringDType</c> (no loop), refuses two different NA objects ("Cannot find a compatible null string value"),
        ///     and raises for a None NA in arithmetic ("Cannot add null that is not a nan-like value") — NumSharp must give
        ///     every answer and every refusal numpy gives, over views and copies.
        /// </summary>
        [TestMethod]
        public void CrossClassOperands_MatchNumpysPromotionAndRefusals()
        {
            DefineSources(NAKindSourcesPython);
            var sources = new (string Label, TextClass Class, string Source, string Route)[]
            {
                ("U", TextClass.Unicode, "src_U", "view"),
                ("S", TextClass.Bytes, "src_S", "view"),
                ("T", TextClass.StringDType, "src_T", "copy"),
                ("T(None)", TextClass.StringDTypeNA, "src_TN", "copy"),
                ("T(nan)", TextClass.StringDType, "src_Tnan", "copy"),
                ("T('NA')", TextClass.StringDType, "src_Tstr", "copy"),
            };
            var pairs = from a in sources
                        from b in sources
                        select new OperandPair($"{a.Label} with {b.Label}", a.Class, $"{a.Source}.reshape(4, 6)",
                            $"{b.Source}.reshape(4, 6)[::-1]", a.Route, b.Route);
            RunPairFamily("cross class", PairOps, pairs).Should().Be(36 * PairOps.Length);
        }

        /// <summary>
        ///     Broadcasting text operands: a 4 × 6 array against a row (6,), a column (4, 1), a 1 × 6 row, a 0-d text, a
        ///     (6,) against a (4, 1) (a fresh 4 × 6 result), a 0-d against a 0-d, and two shapes that do not broadcast (numpy's
        ///     <c>ValueError</c> text, in NumSharp's documented <c>IncorrectShapeException</c> house class) — per class.
        /// </summary>
        [TestMethod]
        public void BroadcastingShapes_MatchNumpy()
        {
            DefineSources();
            var shapes = new (string Label, string X, string P)[]
            {
                ("(4,6) with (6,)", "{s}.reshape(4, 6)", "{s}[6:12]"),
                ("(4,6) with (4,1)", "{s}.reshape(4, 6)", "{s}[::6].reshape(4, 1)"),
                ("(4,6) with (1,6)", "{s}.reshape(4, 6)", "{s}[18:].reshape(1, 6)"),
                ("(4,6) with 0-d", "{s}.reshape(4, 6)", "{s}[9, ...]"),
                ("(6,) with (4,1)", "{s}[:6]", "{s}[6::5][:4].reshape(4, 1)"),
                ("0-d with 0-d", "{s}[1, ...]", "{s}[2, ...]"),
                ("(4,6) with (4,) - no broadcast", "{s}.reshape(4, 6)", "{s}[:4]"),
            };
            var pairs = from cls in AllClasses
                        from shape in shapes
                        let route = RoutesOf(cls)[0]
                        select new OperandPair($"{cls} {shape.Label}", cls, shape.X.Replace("{s}", SourceName(cls)),
                            shape.P.Replace("{s}", SourceName(cls)), route, route);
            RunPairFamily("broadcast", PairOps.Where(op => op.Elementwise).ToArray(), pairs).Should().BeGreaterThan(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // The ufunc keywords
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The keyword helpers: <c>_mask(x)</c> (every other element, a scalar True for 0-d), <c>_kw1</c> / <c>_kw2</c> (a
        ///     ufunc call with <c>out=</c> that must return the out object itself, then the out array), and the out
        ///     factories per class for text results (a wide and a narrow fixed width, or <c>StringDType()</c>).
        /// </summary>
        private const string KeywordsPython = """
            def _mask(x):
                return (np.arange(x.size).reshape(x.shape) % 2 == 0) if x.ndim else np.True_

            def _kw1(f, x, out, **kw):
                r = f(x, out=out, **kw)
                if r is not out: raise AssertionError('the ufunc returned another object than out=')
                return out

            def _kw2(f, x, p, out, **kw):
                r = f(x, p, out=out, **kw)
                if r is not out: raise AssertionError('the ufunc returned another object than out=')
                return out

            def _text_out(x, width):
                if x.dtype.kind == 'T': return np.full(x.shape, 'Q', dtype=StringDType())
                return np.full(x.shape, 'Q' if x.dtype.kind == 'U' else b'Q', dtype=x.dtype.kind + str(width))
            """;

        /// <summary>
        ///     NumSharp's twin of <c>_mask</c>: every other element in C order (a weak scalar True for a 0-d operand).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>The where mask.</returns>
        private static NDArray Mask(NDArray x) => x.ndim == 0 ? (NDArray)true : (np.arange(x.size).reshape(x.Shape) % 2) == 0;

        /// <summary>
        ///     NumSharp's twin of <c>_text_out</c>: a text out array of the operand's class (<c>U</c> / <c>S</c> of the given
        ///     width, or <c>StringDType()</c>) filled with <c>'Q'</c>.
        /// </summary>
        /// <param name="x">The operand whose shape and class the out array takes.</param>
        /// <param name="width">The fixed width of a <c>U</c> / <c>S</c> out array.</param>
        /// <returns>The out array.</returns>
        private static NDArray TextOut(NDArray x, int width)
            => x.dtype.kind == 'T'
                ? np.full(x.Shape, "Q", np.dtypes.StringDType())
                : np.full(x.Shape, x.dtype.kind == 'U' ? "Q" : (object)new[] { (byte)'Q' }, np.dtype($"{x.dtype.kind}{width}"));

        /// <summary>
        ///     Calls a ufunc-shaped NumSharp function with an <c>out=</c> array and returns the out array — after checking
        ///     that the function returned the out OBJECT itself, as numpy's ufuncs do.
        /// </summary>
        /// <param name="call">The call, given the out array.</param>
        /// <param name="out">The out array.</param>
        /// <returns><paramref name="out"/>.</returns>
        /// <exception cref="InvalidOperationException">The function returned another object than <paramref name="out"/>.</exception>
        private static NDArray Out(Func<NDArray, NDArray> call, NDArray @out)
        {
            NDArray r = call(@out);
            if (!ReferenceEquals(r, @out))
                throw new InvalidOperationException("the ufunc returned another object than out=");
            return @out;
        }

        /// <summary>
        ///     The ufunc keywords over text: <c>str_len</c> into an int64 / int8 / float64 <c>out=</c> (numpy casts the
        ///     result under <c>same_kind</c>) with and without a <c>where=</c> mask (the masked elements keep the out array's
        ///     values), <c>dtype=int32</c> (numpy has no such loop), a predicate and the comparisons into a bool and an int64
        ///     <c>out=</c>, and <c>add</c> into a wide, a NARROW (each result cut to the width) and a <c>StringDType</c>
        ///     <c>out=</c> — the returned object must be the out array itself.
        /// </summary>
        [TestMethod]
        public void UfuncKeywords_OutWhereDtype_MatchNumpy()
        {
            DefineSources(KeywordsPython);
            RunFamily("keywords", new[]
            {
                W("str_len out int64 where", "_kw1(np.strings.str_len, x, np.full(x.shape, -1, np.int64), where=_mask(x))",
                    (x, _) => Out(o => np.strings.str_len(x, @out: o, where: Mask(x)), np.full(x.Shape, -1, np.dtype("int64")))),
                W("str_len out int8", "_kw1(np.strings.str_len, x, np.full(x.shape, -1, np.int8))",
                    (x, _) => Out(o => np.strings.str_len(x, @out: o), np.full(x.Shape, -1, np.dtype("int8")))),
                W("str_len out float64", "_kw1(np.strings.str_len, x, np.full(x.shape, -1.0))",
                    (x, _) => Out(o => np.strings.str_len(x, @out: o), np.full(x.Shape, -1.0, np.dtype("float64")))),
                W("str_len dtype=int32", "np.strings.str_len(x, dtype=np.int32)",
                    (x, _) => np.strings.str_len(x, dtype: np.dtype("int32"))),
                W("isalpha out where", "_kw1(np.strings.isalpha, x, np.zeros(x.shape, bool), where=_mask(x))",
                    (x, _) => Out(o => np.strings.isalpha(x, @out: o, where: Mask(x)), np.zeros(x.Shape, np.dtype("bool")))),
                W("equal out bool where", "_kw2(np.equal, x, p, np.zeros(x.shape, bool), where=_mask(x))",
                    (x, p) => Out(o => np.equal(x, p, @out: o, where: Mask(x)), np.zeros(x.Shape, np.dtype("bool")))),
                W("less out int64", "_kw2(np.less, x, p, np.full(x.shape, 7, np.int64))",
                    (x, p) => Out(o => np.less(x, p, @out: o), np.full(x.Shape, 7, np.dtype("int64")))),
                W("add out wide where", "_kw2(np.strings.add, x, p, _text_out(x, 30), where=_mask(x))",
                    (x, p) => Out(o => np.strings.add(x, p, @out: o, where: Mask(x)), TextOut(x, 30))),
                W("add out narrow", "_kw2(np.strings.add, x, p, _text_out(x, 2))",
                    (x, p) => Out(o => np.strings.add(x, p, @out: o), TextOut(x, 2))),
                W("add out StringDType", "_kw2(np.strings.add, x, p, np.full(x.shape, 'Q', dtype=StringDType()))",
                    (x, p) => Out(o => np.strings.add(x, p, @out: o), np.full(x.Shape, "Q", np.dtypes.StringDType()))),
            }).Should().BeGreaterThan(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // StringDType NA kinds
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Every <c>StringDType</c> NA kind through the operations whose answer depends on it: a NaN-like NA propagates
        ///     through <c>add</c>, sorts last and is NaN to <c>isnan</c>; a string NA behaves as its text; a <c>None</c> NA
        ///     raises in arithmetic, ordering and the string functions with numpy's per-function texts; <c>coerce=False</c>
        ///     changes nothing for text input — over every layout.
        /// </summary>
        [TestMethod]
        public void StringDTypeNAKinds_MatchNumpy()
        {
            DefineSources(NAKindSourcesPython);
            var inputs = new[]
            {
                new FamilyInput("na_object=None", TextClass.StringDTypeNA, "src_TN", new[] { "copy" }),
                new FamilyInput("na_object=nan", TextClass.StringDType, "src_Tnan", new[] { "copy" }),
                new FamilyInput("na_object='NA'", TextClass.StringDType, "src_Tstr", new[] { "copy" }),
                new FamilyInput("coerce=False", TextClass.StringDType, "src_Tnc", new[] { "copy" }),
            };
            RunFamily("NA kinds", new[]
            {
                E("str_len", "np.strings.str_len(x)", x => np.strings.str_len(x)),
                E("upper", "np.strings.upper(x)", x => np.strings.upper(x)),
                E("isalpha", "np.strings.isalpha(x)", x => np.strings.isalpha(x)),
                E("isnan", "np.isnan(x)", x => np.isnan(x)),
                E2("add", "np.strings.add(x, p)", (x, p) => np.strings.add(x, p)),
                E2("equal", "x == p", (x, p) => x == p),
                E2("less", "x < p", (x, p) => x < p),
                Lit("find 'l'", "np.strings.find(x, {0})", "l", (x, s) => np.strings.find(x, s)),
                Lit("startswith 'H'", "np.strings.startswith(x, {0})", "H", (x, s) => np.strings.startswith(x, s)),
                W("sort", "np.sort(x)", (x, _) => np.sort(x)),
                W("argsort stable", "np.argsort(x, kind='stable')", (x, _) => np.argsort(x, kind: "stable")),
                W("unique", "np.unique(x)", (x, _) => np.unique(x).values),
                W("max keepdims", "np.max(x, keepdims=True)", (x, _) => np.max(x, keepdims: true)),
                W("astype U12", "x.astype('U12')", (x, _) => x.astype(np.dtype("U12"))),
                W("astype float64", "x.astype('float64')", (x, _) => x.astype(np.dtype("float64"))),
                W("concatenate", "np.concatenate((x.ravel(), p.ravel()))", (x, p) => np.concatenate((x.ravel(), p.ravel()))),
                W("where", "np.where(np.arange(x.size).reshape(x.shape) % 2 == 0, x, p)",
                    (x, p) => np.where((np.arange(x.size).reshape(x.Shape) % 2) == 0, x, p)),
            }, inputs).Should().BeGreaterThan(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // numpy.char, chararray, partition / mod / translate
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>numpy.char</c> against <c>np.@char</c>: the case maps, predicates, pads and searches, and the comparisons,
        ///     which in <c>numpy.char</c> STRIP trailing whitespace first (<c>'ab ' == 'ab'</c> is True there and False for
        ///     <c>np.equal</c>), <c>compare_chararrays</c> with and without the strip, and a chararray's own comparison —
        ///     over every class, layout and route.
        /// </summary>
        [TestMethod]
        public void CharModule_AndChararray_MatchNumpy()
        {
            DefineSources();
            RunFamily("numpy.char", new[]
            {
                E("char.upper", "np.char.upper(x)", x => np.@char.upper(x)),
                E("char.title", "np.char.title(x)", x => np.@char.title(x)),
                E("char.str_len", "np.char.str_len(x)", x => np.@char.str_len(x)),
                E("char.isdigit", "np.char.isdigit(x)", x => np.@char.isdigit(x)),
                E("char.center 9", "np.char.center(x, 9)", x => np.@char.center(x, 9)),
                E("char.rstrip", "np.char.rstrip(x)", x => np.@char.rstrip(x)),
                Lit("char.count 'l'", "np.char.count(x, {0})", "l", (x, s) => np.@char.count(x, s)),
                Lit2("char.replace", "np.char.replace(x, {0}, {1})", "l", "_", (x, a, b) => np.@char.replace(x, a, b)),
                E2("char.equal (strips)", "np.char.equal(x, p)", (x, p) => np.@char.equal(x, p)),
                E2("char.not_equal", "np.char.not_equal(x, p)", (x, p) => np.@char.not_equal(x, p)),
                E2("char.less", "np.char.less(x, p)", (x, p) => np.@char.less(x, p)),
                E2("char.greater_equal", "np.char.greater_equal(x, p)", (x, p) => np.@char.greater_equal(x, p)),
                E2("compare_chararrays == strip", "np.char.compare_chararrays(x, p, '==', True)",
                    (x, p) => np.@char.compare_chararrays(x, p, "==", true)),
                E2("compare_chararrays < no strip", "np.char.compare_chararrays(x, p, '<', False)",
                    (x, p) => np.@char.compare_chararrays(x, p, "<", false)),
                E2("char.add", "np.char.add(x, p)", (x, p) => np.@char.add(x, p)),
                E("char.multiply 2", "np.char.multiply(x, 2)", x => np.@char.multiply(x, 2)),
                W("chararray == chararray", "np.char.array(x) == np.char.array(p)", (x, p) => np.@char.array(x) == np.@char.array(p)),
            }, classes: new[] { TextClass.Unicode, TextClass.Bytes, TextClass.StringDType }).Should().BeGreaterThan(0);
        }

        /// <summary>The format and separator sources: printf-style texts and texts with zero, one and several separators.</summary>
        private const string FormatSourcesPython = """
            fmt_U = np.array(['%s!', '[%5s]', '%-4s|', '%.2s', '%r', '%%s %s', 'x%sy', '%10s', '%s%%', '<%-6s>', '%s', '%.0s'] * 2,
                             dtype='U12')
            fmt_S = fmt_U.astype('S12')
            sep_U = np.array(['a-b-c', 'xyz', '-', '', '--', 'a-', '-b', 'no sep here', 'é-è', 'x-y', '-a-', 'q'] * 2,
                             dtype='U12')
            sep_S = np.array([s.encode('latin-1') for s in sep_U.tolist()], dtype='S12')
            sep_T = sep_U.astype(StringDType())

            def _w1(a):
                # numpy's partition gives a ZERO-WIDTH part ('<U' / '|S', itemsize 0) when every element's part is empty;
                # NumSharp's has width 1 - the documented [Misaligned] divergence (.claude/rules/strings-methods.md).
                return a.astype(a.dtype.kind + '1') if a.dtype.kind in 'US' and a.dtype.itemsize == 0 else a
            """;

        /// <summary>
        ///     <c>partition</c> / <c>rpartition</c> (all three parts: zero, one and several separators, an empty text, a
        ///     separator at an end), <c>mod</c> over printf-style texts (<c>%s</c>, a width, left alignment, a precision,
        ///     <c>%r</c>, an escaped <c>%%</c>) and <c>translate</c> (a <c>U</c> dict table mapping and deleting; an <c>S</c>
        ///     256-byte table with deleted bytes) — over the layouts of a 24-element source.
        /// </summary>
        [TestMethod]
        public void PartitionModTranslate_MatchNumpy()
        {
            DefineSources(FormatSourcesPython);
            var layouts = new[] { ("C 1-D", "src"), ("C 2-D", "src.reshape(4, 6)"), ("reversed", "src[::-1]"), ("stepped", "src[1::3]"),
                                  ("broadcast", "np.broadcast_to(src[:6], (2, 6))"), ("0-d", "src[3, ...]"), ("empty", "src[:0]") };
            var sepInputs = new[]
            {
                new FamilyInput("U", TextClass.Unicode, "sep_U", new[] { "view", "copy" }),
                new FamilyInput("S", TextClass.Bytes, "sep_S", new[] { "view", "copy" }),
                new FamilyInput("StringDType", TextClass.StringDType, "sep_T", new[] { "copy" }),
            };
            RunFamily("partition", new[]
            {
                Lit("partition before", "_w1(np.strings.partition(x, {0})[0])", "-", (x, s) => np.strings.partition(x, s).Before),
                Lit("partition separator", "_w1(np.strings.partition(x, {0})[1])", "-", (x, s) => np.strings.partition(x, s).Separator),
                Lit("partition after", "_w1(np.strings.partition(x, {0})[2])", "-", (x, s) => np.strings.partition(x, s).After),
                Lit("rpartition before", "_w1(np.strings.rpartition(x, {0})[0])", "-", (x, s) => np.strings.rpartition(x, s).Before),
                Lit("rpartition after", "_w1(np.strings.rpartition(x, {0})[2])", "-", (x, s) => np.strings.rpartition(x, s).After),
                Lit("partition '' (empty separator)", "_w1(np.strings.partition(x, {0})[0])", "", (x, s) => np.strings.partition(x, s).Before),
                new TextOp("translate", cls => cls == TextClass.Bytes
                        ? "np.strings.translate(x, bytes(range(256)).replace(b'-', b'+'), b'a')"
                        : "np.strings.translate(x, {ord('-'): '+', ord('a'): None, ord('x'): 'XX'})",
                    (cls, x, _) => cls == TextClass.Bytes
                        ? np.strings.translate(x, TranslateBytesTable(), new[] { (byte)'a' })
                        : np.strings.translate(x, new Dictionary<int, object> { ['-'] = "+", ['a'] = null, ['x'] = "XX" })),
            }, sepInputs, layouts).Should().BeGreaterThan(0);

            var fmtInputs = new[]
            {
                new FamilyInput("U", TextClass.Unicode, "fmt_U", new[] { "view", "copy" }),
                new FamilyInput("S", TextClass.Bytes, "fmt_S", new[] { "view", "copy" }),
            };
            RunFamily("mod", new[]
            {
                Lit("mod text", "np.strings.mod(x, {0})", "abc", (x, s) => np.strings.mod(x, s)),
                W("mod int", "np.strings.mod(x, 42)", (x, _) => np.strings.mod(x, 42)),
                W("mod per-element", "np.strings.mod(x, np.arange(x.size).reshape(x.shape))",
                    (x, _) => np.strings.mod(x, np.arange(x.size).reshape(x.Shape))),
            }, fmtInputs, layouts).Should().BeGreaterThan(0);
        }

        /// <summary>The 256-byte <c>S</c> translation table of the test: identity except <c>'-'</c> → <c>'+'</c>.</summary>
        /// <returns>The table.</returns>
        private static byte[] TranslateBytesTable()
        {
            byte[] table = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
            table['-'] = (byte)'+';
            return table;
        }

        // ------------------------------------------------------------------------------------------------------------
        // np.evaluate over text
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Fused <c>np.evaluate</c> text expressions against numpy's own composition of the same functions: comparisons
        ///     with a literal and with an array, a logical combination of a predicate and a length test, the
        ///     <c>numpy.strings</c> factories (<c>StrLen</c>, <c>Find</c>, <c>StartsWith</c>), the truth reductions
        ///     (<c>Any</c>, <c>CountNonzero</c>), and the text RESULTS (<c>Where</c> with text branches, <c>Add</c>,
        ///     <c>Multiply</c>, <c>Maximum</c>, a <c>Cast</c> to a narrower <c>U</c>) — every class, layout and route.
        /// </summary>
        /// <remarks>
        ///     <c>NDExpr.Multiply</c> is the <c>np.multiply</c> UFUNC, not <c>np.strings.multiply</c>: numpy refuses a
        ///     fixed-width text there without <c>out=</c> ("The 'out' kwarg is necessary when using the string multiply ufunc
        ///     directly…") and computes a <c>StringDType</c> one, so that is the twin.
        /// </remarks>
        [TestMethod]
        public void Evaluate_TextExpressions_MatchNumpysComposition()
        {
            DefineSources();
            RunFamily("evaluate", new[]
            {
                Lit("Equal literal", "x == {0}", "hello", (x, s) => np.evaluate(NDExpr.Equal(x, s))),
                E2("Less array", "x < p", (x, p) => np.evaluate(NDExpr.Less(x, p))),
                Lit("And(StartsWith, StrLen > 3)", "np.logical_and(np.strings.startswith(x, {0}), np.strings.str_len(x) > 3)", "H",
                    (x, s) => np.evaluate(NDExpr.LogicalAnd(NDExpr.StartsWith(x, s), NDExpr.Greater(NDExpr.StrLen(x), 3)))),
                E("StrLen * 2 + 1", "np.strings.str_len(x) * 2 + 1", x => np.evaluate(NDExpr.Add(NDExpr.Multiply(NDExpr.StrLen(x), 2), 1))),
                Lit("Find", "np.strings.find(x, {0})", "l", (x, s) => np.evaluate(NDExpr.Find(x, s))),
                W("Any(Equal '')", "np.any(x == '' if x.dtype.kind != 'S' else x == b'')",
                    (x, _) => np.evaluate(NDExpr.Any(NDExpr.Equal(x, x.dtype.kind == 'S' ? np.array("", np.dtype("S")) : (NDArray)"")))),
                W("CountNonzero(IsAlpha)", "np.count_nonzero(np.strings.isalpha(x))", (x, _) => np.evaluate(NDExpr.CountNonzero(NDExpr.IsAlpha(x)))),
                E2("Where text branches", "np.where(np.strings.str_len(x) > 3, x, p)",
                    (x, p) => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.StrLen(x), 3), x, p))),
                Lit("Add literal", "np.strings.add(x, {0})", "!", (x, s) => np.evaluate(NDExpr.Add(x, s))),
                E("Multiply 2 (the np.multiply ufunc)", "np.multiply(x, 2)", x => np.evaluate(NDExpr.Multiply(x, 2))),
                E2("Maximum", "np.maximum(x, p)", (x, p) => np.evaluate(NDExpr.Maximum(x, p))),
                W("Cast U3", "x.astype('U3')", (x, _) => np.evaluate(NDExpr.Cast(x, np.dtype("U3")))),
            }).Should().BeGreaterThan(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // numpy's functions called with NumSharp arrays through the codec
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The way a program USES the bridge: with the codec registered, numpy's own functions are called with NumSharp
        ///     text arrays as arguments (pythonnet encodes each through the codec: a view of a fixed-width array, a copy of
        ///     a <c>StringDType</c> one) and the result is decoded back into an NDArray. For every class, NumSharp-made layout
        ///     and function, numpy's result over the encoded argument equals NumSharp's own computation, and the decoded
        ///     NDArray equals numpy's result — both directions of the codec in one cell.
        /// </summary>
        /// <remarks>
        ///     A call over a 0-d argument returns a SCALAR: a numpy scalar (<c>np.str_</c>, <c>np.bytes_</c>, <c>np.int64</c>)
        ///     exports a buffer and decodes as a 0-d array, while a <c>StringDType</c> call returns a plain Python
        ///     <c>str</c>, which the codec deliberately does not decode (a <c>str</c> is not an array-like,
        ///     <c>NumpyCodecOptions.DecodeArrayLike</c>): that refusal is pinned, and NumSharp's result is still held to
        ///     numpy's.
        /// </remarks>
        [TestMethod]
        public void CodecCalls_NumpyFunctionsOverNumSharpText_MatchNumSharp()
        {
            DefineSources();
            NDArrayPythonInterop.RegisterCodec();
            var calls = new (string Name, Func<dynamic, NDArray, dynamic> Numpy, Func<NDArray, NDArray> Sharp)[]
            {
                ("strings.upper", (npy, x) => npy.strings.upper(x), x => np.strings.upper(x)),
                ("strings.str_len", (npy, x) => npy.strings.str_len(x), x => np.strings.str_len(x)),
                ("strings.add(x, x)", (npy, x) => npy.strings.add(x, x), x => np.strings.add(x, x)),
                ("char.center(x, 9)", (npy, x) => npy.@char.center(x, 9), x => np.@char.center(x, 9)),
                ("sort", (npy, x) => npy.sort(x, null), x => np.sort(x, axis: null)),
                ("unique", (npy, x) => npy.unique(x), x => np.unique(x).values),
            };
            var failures = new List<string>();
            int cells = 0;
            foreach (TextClass cls in AllClasses)
            {
                using NDArray b = ImportOf(SourceName(cls));
                foreach (var (layoutName, make) in NumSharpLayouts.Where(l => l.Name is "C 2-D" or "F 2-D" or "reversed" or "split child (offset view)" or "0-d"))
                {
                    NDArray xs = make(b);
                    foreach (var (name, numpy, sharp) in calls)
                    {
                        cells++;
                        string cell = $"codec | {cls} | {layoutName} | {name}";
                        NDArray computed;
                        try
                        {
                            computed = sharp(xs);
                        }
                        catch (Exception e)
                        {
                            // A refusal (a None NA in a string function): numpy's call over the encoded argument must
                            // refuse too — its exception surfaces as a PythonException through pythonnet.
                            using (Gil())
                            {
                                dynamic npyRefusal = Py.Import("numpy");
                                try
                                {
                                    using PyObject unexpected = numpy(npyRefusal, xs);
                                    failures.Add($"{cell}: NumSharp raises {e.GetType().Name}: {e.Message} -- numpy computes");
                                }
                                catch (PythonException)
                                {
                                    // Both refuse: the class and text of this refusal are the NA-kinds family's subject.
                                }
                            }

                            continue;
                        }

                        using NDArray mine = computed;
                        using (Gil())
                        {
                            dynamic npy = Py.Import("numpy");
                            using PyObject result = numpy(npy, xs);
                            Scope.Set("_ref", result);

                            // NumSharp's own computation against numpy's result over the encoded argument.
                            using (PyObject mineOut = mine.dtype.kind == 'T' ? mine.ToNumpyCopy() : mine.ToNumpy())
                            {
                                Scope.Set("_mine", mineOut);
                                using PyObject diff = Scope.Eval("_cmp(_mine, _ref)");
                                if (diff.As<string>().Length > 0)
                                    failures.Add($"{cell}: NumSharp vs numpy: {diff.As<string>()}");
                            }

                            // The decode direction. A StringDType call over a 0-d argument returns a plain Python str,
                            // which the codec deliberately refuses (NumpyCodecOptions.DecodeArrayLike: a str is not an
                            // array-like) - pinned; a numpy scalar (np.str_, np.bytes_, np.int64: a buffer exporter)
                            // and an ndarray decode, and must read back as numpy's result.
                            if (PyBool("isinstance(_ref, (str, bytes)) and not isinstance(_ref, np.generic)"))
                            {
                                new Action(() => result.As<NDArray>()).Should().Throw<InvalidCastException>(cell);
                            }
                            else
                            {
                                using NDArray decoded = result.As<NDArray>();
                                using PyObject decodedOut = decoded.dtype.kind == 'T' ? decoded.ToNumpyCopy() : decoded.ToNumpy();
                                Scope.Set("_dec", decodedOut);
                                using PyObject decDiff = Scope.Eval("_cmp(_dec, _ref)");
                                if (decDiff.As<string>().Length > 0)
                                    failures.Add($"{cell}: the decoded result vs numpy's: {decDiff.As<string>()}");
                                Scope.Exec("del _dec");
                            }

                            Scope.Exec("del _ref, _mine");
                        }
                    }

                    if (!ReferenceEquals(xs, b))
                        xs.Dispose();
                }
            }

            AssertNoFailures("codec", cells, failures);
        }
    }
}
