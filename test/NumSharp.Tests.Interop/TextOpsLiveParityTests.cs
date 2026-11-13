using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     End-to-end text parity through the pythonnet bridge, one family of <c>numpy.strings</c> / text ufunc operations
    ///     per test: numpy builds the text, the bridge carries it into NumSharp (a zero-copy view of a fixed-width array, an
    ///     exact copy of a <c>StringDType</c> one), NumSharp computes, the bridge carries the result back, and LIVE numpy's
    ///     answer over the same memory is the expected one — dtype, shape and bytes, or the exception class and text.
    /// </summary>
    /// <remarks>
    ///     <para>The crossing itself (bytes, layouts, descriptors, lifetimes) is
    ///     <see cref="TextFixedWidthCrossingTests"/>' / <see cref="TextStringDTypeCrossingTests"/>' subject; this class
    ///     holds what a program does with the text once it crossed. Every family runs the four classes
    ///     (<see cref="TextClass"/>) × the twelve layouts (<see cref="TextLiveParityHarness.Layouts"/>) × the routes the
    ///     class has, so an operation is proved over numpy-born memory in every shape and stride numpy hands over, the
    ///     broadcast (stride 0, read-only) and the step that ends at the buffer's last element included.</para>
    ///     <para>Where numpy refuses (a bytes array has no <c>isnumeric</c>, an <c>index</c> that finds nothing, an ASCII
    ///     encode of non-ASCII text), the cell asserts NumSharp refuses with numpy's class and message.</para>
    /// </remarks>
    [TestClass]
    public class TextOpsLiveParityTests : TextLiveParityHarness
    {
        /// <summary>
        ///     Case mapping — <c>upper</c>, <c>lower</c>, <c>capitalize</c>, <c>title</c>, <c>swapcase</c> — over texts whose
        ///     mapping changes their length (<c>ß</c> → <c>SS</c>, <c>İ</c> → <c>i̇</c>), needs context (a final sigma)
        ///     or has a titlecase form (<c>ǅ</c>); bytes map ASCII only and leave high bytes alone.
        /// </summary>
        [TestMethod]
        public void CaseMapping_EveryClassLayoutAndRoute_MatchesNumpy()
        {
            DefineSources();
            RunFamily("case", new[]
            {
                E("upper", "np.strings.upper(x)", x => np.strings.upper(x)),
                E("lower", "np.strings.lower(x)", x => np.strings.lower(x)),
                E("capitalize", "np.strings.capitalize(x)", x => np.strings.capitalize(x)),
                E("title", "np.strings.title(x)", x => np.strings.title(x)),
                E("swapcase", "np.strings.swapcase(x)", x => np.strings.swapcase(x)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     The character-class predicates and <c>str_len</c>: <c>isnumeric</c> / <c>isdecimal</c> are refused for bytes
        ///     by numpy (no loop), numeric-only (<c>Ⅻ</c>), decimal (<c>٣</c>) and digit-only (<c>²</c>) characters split
        ///     the three numeric predicates, and the empty string is false for every predicate; <c>str_len</c> counts code
        ///     points (an astral character is one) and an NA answers numpy's NA rule.
        /// </summary>
        [TestMethod]
        public void PredicatesAndLength_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("predicates", new[]
            {
                E("isalpha", "np.strings.isalpha(x)", x => np.strings.isalpha(x)),
                E("isdigit", "np.strings.isdigit(x)", x => np.strings.isdigit(x)),
                E("isnumeric", "np.strings.isnumeric(x)", x => np.strings.isnumeric(x)),
                E("isdecimal", "np.strings.isdecimal(x)", x => np.strings.isdecimal(x)),
                E("isspace", "np.strings.isspace(x)", x => np.strings.isspace(x)),
                E("isalnum", "np.strings.isalnum(x)", x => np.strings.isalnum(x)),
                E("islower", "np.strings.islower(x)", x => np.strings.islower(x)),
                E("isupper", "np.strings.isupper(x)", x => np.strings.isupper(x)),
                E("istitle", "np.strings.istitle(x)", x => np.strings.istitle(x)),
                E("str_len", "np.strings.str_len(x)", x => np.strings.str_len(x)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Whitespace and padding: the three strips with and without a character set, <c>center</c> / <c>ljust</c> /
        ///     <c>rjust</c> wider and narrower than the texts (a fixed-width result widens to the requested width), a
        ///     multi-character fill refused, <c>zfill</c> after a sign, and <c>expandtabs</c> with the default and an odd
        ///     tab size.
        /// </summary>
        [TestMethod]
        public void StripPadAndTabs_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("strip/pad", new[]
            {
                E("strip", "np.strings.strip(x)", x => np.strings.strip(x)),
                E("lstrip", "np.strings.lstrip(x)", x => np.strings.lstrip(x)),
                E("rstrip", "np.strings.rstrip(x)", x => np.strings.rstrip(x)),
                Lit("strip ' x'", "np.strings.strip(x, {0})", " x", (x, c) => np.strings.strip(x, c)),
                Lit("lstrip tab+space", "np.strings.lstrip(x, {0})", "\t ", (x, c) => np.strings.lstrip(x, c)),
                Lit("rstrip 'od '", "np.strings.rstrip(x, {0})", "od ", (x, c) => np.strings.rstrip(x, c)),
                E("center 14", "np.strings.center(x, 14)", x => np.strings.center(x, 14)),
                Lit("center 14 '*'", "np.strings.center(x, 14, {0})", "*", (x, c) => np.strings.center(x, 14, c)),
                E("center 3", "np.strings.center(x, 3)", x => np.strings.center(x, 3)),
                Lit("center 9 '-='", "np.strings.center(x, 9, {0})", "-=", (x, c) => np.strings.center(x, 9, c)),
                Lit("ljust 9 '-'", "np.strings.ljust(x, 9, {0})", "-", (x, c) => np.strings.ljust(x, 9, c)),
                E("rjust 13", "np.strings.rjust(x, 13)", x => np.strings.rjust(x, 13)),
                E("rjust -1", "np.strings.rjust(x, -1)", x => np.strings.rjust(x, -1)),
                E("zfill 8", "np.strings.zfill(x, 8)", x => np.strings.zfill(x, 8)),
                E("expandtabs", "np.strings.expandtabs(x)", x => np.strings.expandtabs(x)),
                E("expandtabs 3", "np.strings.expandtabs(x, 3)", x => np.strings.expandtabs(x, 3)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Searching: <c>find</c> / <c>rfind</c> / <c>count</c> / <c>index</c> / <c>rindex</c> / <c>startswith</c> /
        ///     <c>endswith</c> with a literal, the empty substring (it matches at every position, <c>count</c> gives length
        ///     + 1), a start past the end, a negative end, and an ARRAY of substrings (the reversed input, elementwise);
        ///     <c>rindex</c> over texts that lack the substring raises numpy's <c>ValueError</c>.
        /// </summary>
        [TestMethod]
        public void Searching_EveryClassLayoutAndRoute_MatchesNumpy()
        {
            DefineSources();
            RunFamily("search", new[]
            {
                Lit("find 'l'", "np.strings.find(x, {0})", "l", (x, s) => np.strings.find(x, s)),
                Lit("rfind 'l'", "np.strings.rfind(x, {0})", "l", (x, s) => np.strings.rfind(x, s)),
                Lit("count 'l'", "np.strings.count(x, {0})", "l", (x, s) => np.strings.count(x, s)),
                Lit("count ''", "np.strings.count(x, {0})", "", (x, s) => np.strings.count(x, s)),
                Lit("find 'o' 2 -1", "np.strings.find(x, {0}, 2, -1)", "o", (x, s) => np.strings.find(x, s, 2, -1)),
                Lit("rfind '' 3", "np.strings.rfind(x, {0}, 3)", "", (x, s) => np.strings.rfind(x, s, 3)),
                Lit("count ' ' 1 6", "np.strings.count(x, {0}, 1, 6)", " ", (x, s) => np.strings.count(x, s, 1, 6)),
                Lit("index ''", "np.strings.index(x, {0})", "", (x, s) => np.strings.index(x, s)),
                Lit("rindex 'l'", "np.strings.rindex(x, {0})", "l", (x, s) => np.strings.rindex(x, s)),
                Lit("startswith 'He'", "np.strings.startswith(x, {0})", "He", (x, s) => np.strings.startswith(x, s)),
                Lit("endswith 'd'", "np.strings.endswith(x, {0})", "d", (x, s) => np.strings.endswith(x, s)),
                Lit("startswith '' 30", "np.strings.startswith(x, {0}, 30)", "", (x, s) => np.strings.startswith(x, s, 30)),
                Lit("endswith 'o' 0 5", "np.strings.endswith(x, {0}, 0, 5)", "o", (x, s) => np.strings.endswith(x, s, 0, 5)),
                E2("find partner", "np.strings.find(x, p)", (x, p) => np.strings.find(x, p)),
                E2("count partner", "np.strings.count(x, p)", (x, p) => np.strings.count(x, p)),
                E2("startswith partner", "np.strings.startswith(x, p)", (x, p) => np.strings.startswith(x, p)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Rewriting: <c>replace</c> (with a count, with an empty old string that inserts between characters, with a
        ///     replacement longer than the match, so a fixed-width result widens), <c>slice</c> (forward, reversed,
        ///     negative start, stepped), <c>add</c> (two arrays, array + literal, literal + array) and <c>multiply</c>
        ///     (positive, zero and negative counts, and an elementwise count array).
        /// </summary>
        [TestMethod]
        public void ReplaceSliceAddMultiply_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("rewrite", new[]
            {
                Lit2("replace l LL", "np.strings.replace(x, {0}, {1})", "l", "LL", (x, a, b) => np.strings.replace(x, a, b)),
                Lit2("replace l '' 1", "np.strings.replace(x, {0}, {1}, 1)", "l", "", (x, a, b) => np.strings.replace(x, a, b, 1)),
                Lit2("replace '' -", "np.strings.replace(x, {0}, {1})", "", "-", (x, a, b) => np.strings.replace(x, a, b)),
                Lit2("replace xx y 2", "np.strings.replace(x, {0}, {1}, 2)", "xx", "y", (x, a, b) => np.strings.replace(x, a, b, 2)),
                E("slice 1:4", "np.strings.slice(x, 1, 4)", x => np.strings.slice(x, 1, 4)),
                E("slice ::-1", "np.strings.slice(x, None, None, -1)", x => np.strings.slice(x, null, null, -1)),
                E("slice -3:", "np.strings.slice(x, -3, None)", x => np.strings.slice(x, -3, null)),
                E("slice ::2", "np.strings.slice(x, None, None, 2)", x => np.strings.slice(x, null, null, 2)),
                E2("add x p", "np.strings.add(x, p)", (x, p) => np.strings.add(x, p)),
                Lit("add x '!?'", "np.strings.add(x, {0})", "!?", (x, s) => np.strings.add(x, s)),
                Lit("add '>' x", "np.strings.add({0}, x)", ">", (x, s) => np.strings.add(s, x)),
                E("multiply 3", "np.strings.multiply(x, 3)", x => np.strings.multiply(x, 3)),
                E("multiply 0", "np.strings.multiply(x, 0)", x => np.strings.multiply(x, 0)),
                E("multiply -2", "np.strings.multiply(x, -2)", x => np.strings.multiply(x, -2)),
                E2("multiply by partner length", "np.strings.multiply(x, np.strings.str_len(p) % 3)",
                    (x, p) => np.strings.multiply(x, np.strings.str_len(p) % 3)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Codecs: <c>encode</c> (UTF-8, UTF-16 with its BOM, ASCII with <c>replace</c>, strict ASCII refusing non-ASCII
        ///     text with numpy's <c>UnicodeEncodeError</c>) and <c>decode</c> (Latin-1, UTF-8 with <c>replace</c>, strict
        ///     UTF-8 refusing the high bytes with <c>UnicodeDecodeError</c>, ASCII with <c>ignore</c>); each refused by
        ///     numpy for the class it does not take.
        /// </summary>
        [TestMethod]
        public void EncodeDecode_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("codec", new[]
            {
                E("encode utf-8", "np.strings.encode(x, 'utf-8')", x => np.strings.encode(x, "utf-8")),
                E("encode utf-16", "np.strings.encode(x, 'utf-16')", x => np.strings.encode(x, "utf-16")),
                E("encode ascii replace", "np.strings.encode(x, 'ascii', 'replace')", x => np.strings.encode(x, "ascii", "replace")),
                E("encode ascii", "np.strings.encode(x, 'ascii')", x => np.strings.encode(x, "ascii")),
                E("decode latin-1", "np.strings.decode(x, 'latin-1')", x => np.strings.decode(x, "latin-1")),
                E("decode utf-8 replace", "np.strings.decode(x, 'utf-8', 'replace')", x => np.strings.decode(x, "utf-8", "replace")),
                E("decode utf-8", "np.strings.decode(x, 'utf-8')", x => np.strings.decode(x, "utf-8")),
                E("decode ascii ignore", "np.strings.decode(x, 'ascii', 'ignore')", x => np.strings.decode(x, "ascii", "ignore")),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Comparison: the six comparison ufuncs and the six C# operators against numpy's operators, array to array (the
        ///     reversed input) and array to literal, plus <c>np.maximum</c> / <c>np.minimum</c> — all code-point (or byte)
        ///     order, an NA ordered as numpy orders it.
        /// </summary>
        [TestMethod]
        public void Comparisons_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("compare", new[]
            {
                E2("np.equal", "np.equal(x, p)", (x, p) => np.equal(x, p)),
                E2("np.not_equal", "np.not_equal(x, p)", (x, p) => np.not_equal(x, p)),
                E2("np.less", "np.less(x, p)", (x, p) => np.less(x, p)),
                E2("np.less_equal", "np.less_equal(x, p)", (x, p) => np.less_equal(x, p)),
                E2("np.greater", "np.greater(x, p)", (x, p) => np.greater(x, p)),
                E2("np.greater_equal", "np.greater_equal(x, p)", (x, p) => np.greater_equal(x, p)),
                E2("==", "x == p", (x, p) => x == p),
                E2("!=", "x != p", (x, p) => x != p),
                E2("<", "x < p", (x, p) => x < p),
                E2("<=", "x <= p", (x, p) => x <= p),
                E2(">", "x > p", (x, p) => x > p),
                E2(">=", "x >= p", (x, p) => x >= p),
                Lit("== literal", "x == {0}", "hello", (x, s) => x == s),
                Lit("< literal", "x < {0}", "M", (x, s) => x < s),
                Lit("np.strings.greater_equal literal", "np.strings.greater_equal(x, {0})", "hello",
                    (x, s) => np.strings.greater_equal(x, s)),
                E2("np.maximum", "np.maximum(x, p)", (x, p) => np.maximum(x, p)),
                E2("np.minimum", "np.minimum(x, p)", (x, p) => np.minimum(x, p)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Ordering and set routines over text: <c>sort</c> (last axis, flattened, first axis), stable <c>argsort</c>,
        ///     the four <c>unique</c> results, <c>searchsorted</c> into the sorted text, <c>max</c> / <c>min</c> with
        ///     <c>keepdims</c>, <c>argmax</c> / <c>argmin</c> — on a 0-d input the axis errors, on an empty one the
        ///     empty-reduction errors, each with numpy's class and text (the AxisError compared without .NET's
        ///     <c>(Parameter 'axis')</c> decoration, see <see cref="TextLiveParityHarness.NumpyFacingMessage"/>).
        /// </summary>
        /// <remarks>
        ///     The flattened <c>argsort</c> asks numpy over <c>np.ascontiguousarray(x)</c>: numpy 2.4.2's
        ///     <c>argsort(axis=None)</c> of a NON-contiguous <c>StringDType</c> array raises
        ///     <c>MemoryError: Failed to load string in string comparison</c> (transposed) or returns a wrong permutation
        ///     (reversed rows), where the same call over the contiguous copy — the definition of the flattened argsort —
        ///     is right; 2.5.4 still returns wrong permutations for transposed, reversed and stepped views of arena texts
        ///     (probed). NumSharp gives the contiguous copy's answer for every layout;
        ///     <c>StringDTypeOrderTests.ArgSort_NonContiguousView_AxisNone_SortsLikeItsCopy</c> (<c>[Misaligned]</c>, the unit
        ///     suite) pins it.
        /// </remarks>
        [TestMethod]
        public void SortUniqueSearchAndExtrema_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("order", new[]
            {
                W("sort", "np.sort(x)", (x, _) => np.sort(x)),
                W("sort axis=None", "np.sort(x, axis=None)", (x, _) => np.sort(x, axis: null)),
                W("sort axis=0", "np.sort(x, axis=0)", (x, _) => np.sort(x, axis: 0)),
                W("argsort stable", "np.argsort(x, kind='stable')", (x, _) => np.argsort(x, kind: "stable")),
                W("argsort axis=None stable", "np.argsort(np.ascontiguousarray(x), axis=None, kind='stable')",
                    (x, _) => np.argsort(x, axis: null, kind: "stable")),
                W("unique", "np.unique(x)", (x, _) => np.unique(x).values),
                W("unique counts", "np.unique(x, return_counts=True)[1]", (x, _) => np.unique(x, return_counts: true).counts),
                W("unique index", "np.unique(x, return_index=True)[1]", (x, _) => np.unique(x, return_index: true).indices),
                W("unique inverse", "np.unique(x, return_inverse=True)[1]", (x, _) => np.unique(x, return_inverse: true).inverse_indices),
                W("searchsorted", "np.searchsorted(np.sort(x, axis=None), x)", (x, _) => np.searchsorted(np.sort(x, axis: null), x)),
                W("searchsorted right", "np.searchsorted(np.sort(x, axis=None), x, side='right')",
                    (x, _) => np.searchsorted(np.sort(x, axis: null), x, side: "right")),
                W("max keepdims", "np.max(x, keepdims=True)", (x, _) => np.max(x, keepdims: true)),
                W("min keepdims", "np.min(x, keepdims=True)", (x, _) => np.min(x, keepdims: true)),
                W("max axis=-1", "np.max(x, axis=-1)", (x, _) => np.max(x, axis: -1)),
                W("argmax", "np.argmax(x)", (x, _) => np.argmax(x, axis: (int?)null)),
                W("argmin", "np.argmin(x)", (x, _) => np.argmin(x, axis: (int?)null)),
                W("argmax axis=0", "np.argmax(x, axis=0)", (x, _) => np.argmax(x, axis: (int?)0)),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Shape and selection routines over text: <c>concatenate</c>, <c>stack</c>, <c>where</c> with a mask computed
        ///     from the text, <c>repeat</c>, a boolean mask, <c>take</c> (an empty input refuses with numpy's
        ///     <c>IndexError</c>), <c>flip</c>, <c>full_like</c>, <c>zeros_like</c> and a copy in F order — every one keeps
        ///     the text's dtype (width, or descriptor with its NA object) as numpy does.
        /// </summary>
        [TestMethod]
        public void ShapeAndSelection_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("shape", new[]
            {
                W("concatenate", "np.concatenate((x.ravel(), p.ravel()))", (x, p) => np.concatenate((x.ravel(), p.ravel()))),
                W("stack", "np.stack((x, p))", (x, p) => np.stack(new[] { x, p })),
                W("where", "np.where(np.strings.str_len(x) > 3, x, p)", (x, p) => np.where(np.strings.str_len(x) > 3, x, p)),
                W("repeat 2", "np.repeat(x, 2)", (x, _) => np.repeat(x, 2)),
                W("boolean mask", "x[np.strings.str_len(x) > 3]", (x, _) => x[np.strings.str_len(x) > 3]),
                W("take [0, -1]", "np.take(x, [0, -1])", (x, _) => np.take(x, np.array(new long[] { 0, -1 }))),
                W("flip", "np.flip(x)", (x, _) => np.flip(x)),
                W("full_like 'zz'", "np.full_like(x, 'zz')", (x, _) => np.full_like(x, "zz")),
                W("zeros_like", "np.zeros_like(x)", (x, _) => np.zeros_like(x)),
                W("copy order=F", "x.copy(order='F')", (x, _) => x.copy('F')),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     Casts between the text classes: to a narrower <c>U</c> (truncation), a wider one, to <c>S</c> (strict ASCII:
        ///     non-ASCII text raises numpy's <c>UnicodeEncodeError</c>), to a narrow <c>S</c>, to <c>StringDType</c> with and
        ///     without an NA object, to unsized <c>U</c> (numpy keeps or derives the width) and to <c>bool</c> (numpy's
        ///     nonzero-length rule for every text class).
        /// </summary>
        /// <remarks>
        ///     <para>numpy's <c>S → StringDType</c> cast copies bytes that are not UTF-8 unchecked and raises
        ///     <c>UnicodeDecodeError</c> only when an element is READ; NumSharp raises the same text at the cast (the parity
        ///     journal's documented owner's call). numpy's cast is therefore followed by a read of every element
        ///     (<c>_eager</c>), which moves numpy's error to the cast — the cell then holds NumSharp's eager refusal to
        ///     numpy's text. <c>StringDTypeCastTests.NumPyBug_SToT_InvalidUtf8_RaisesAtTheCast</c> (the unit suite) pins
        ///     the difference.</para>
        ///     <para>A cast to <c>S</c> that meets unencodable text names the FIRST such element of the cast's MEMORY walk
        ///     on both sides, so an F-ordered input names another element than its C-ordered copy; the copy route's twin is
        ///     therefore numpy's C copy (<c>RunFamily</c>: a copy is C-contiguous by the
        ///     bridge's contract),
        ///     and <c>TextCastErrorOrderTests</c> (the unit suite) guards the walk.</para>
        /// </remarks>
        [TestMethod]
        public void TextToTextCasts_EveryClassLayoutAndRoute_MatchNumpy()
        {
            DefineSources();
            RunFamily("text cast", new[]
            {
                W("astype U3", "x.astype('U3')", (x, _) => x.astype(np.dtype("U3"))),
                W("astype U30", "x.astype('U30')", (x, _) => x.astype(np.dtype("U30"))),
                W("astype S12", "x.astype('S12')", (x, _) => x.astype(np.dtype("S12"))),
                W("astype S3", "x.astype('S3')", (x, _) => x.astype(np.dtype("S3"))),
                W("astype U", "x.astype('U')", (x, _) => x.astype(np.dtype("U"))),
                W("astype S", "x.astype('S')", (x, _) => x.astype(np.dtype("S"))),
                W("astype StringDType()", "_eager(x.astype(StringDType()))", (x, _) => x.astype(np.dtypes.StringDType())),
                W("astype StringDType(na_object=None)", "_eager(x.astype(StringDType(na_object=None)))",
                    (x, _) => x.astype(np.dtypes.StringDType((object)null))),
                W("astype bool", "x.astype(bool)", (x, _) => x.astype(np.dtype("bool"))),
            }).Should().BeGreaterThan(0);
        }

        /// <summary>
        ///     The cell matrix is not vacuous: one family's run over the standard classes, layouts and routes visits
        ///     (2 + 2 + 1 + 1) × 12 crossings per operation — a regression that dropped a class or a route from
        ///     <see cref="TextLiveParityHarness.RunFamily"/> would shrink it.
        /// </summary>
        [TestMethod]
        public void Matrix_VisitsEveryClassLayoutAndRoute()
        {
            DefineSources();
            int crossings = AllClasses.Sum(c => RoutesOf(c).Length) * Layouts.Length;
            crossings.Should().Be(72);
            RunFamily("matrix", new[] { E("str_len", "np.strings.str_len(x)", x => np.strings.str_len(x)) })
                .Should().Be(crossings, "one operation runs once per class × layout × route");
        }
    }
}
