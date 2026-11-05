using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The four text classes the end-to-end parity families run over. Each is built by numpy (numpy-born memory) and
    ///     crosses the bridge before NumSharp computes anything.
    /// </summary>
    public enum TextClass
    {
        /// <summary>Fixed-width <c>U12</c> (UCS-4 code points): crosses as a zero-copy view and as a copy.</summary>
        Unicode,

        /// <summary>Fixed-width <c>S12</c> (bytes, high bytes and embedded NULs included): view and copy.</summary>
        Bytes,

        /// <summary>NEP 55 <c>StringDType()</c> without an NA object: crosses as an exact copy only.</summary>
        StringDType,

        /// <summary><c>StringDType(na_object=None)</c> holding two NA cells: an exact copy only.</summary>
        StringDTypeNA,
    }

    /// <summary>
    ///     One operation of a parity family: the numpy expression and the NumSharp computation of the same thing.
    /// </summary>
    /// <param name="Name">The cell label printed in a failure.</param>
    /// <param name="Numpy">The numpy expression over <c>x</c> (the input) and <c>p</c> (its partner: <c>x[::-1]</c>, or
    ///     <c>x</c> itself when 0-d) for a text class — the class decides whether a literal is <c>'..'</c> or
    ///     <c>b'..'</c>.</param>
    /// <param name="Sharp">NumSharp's computation over the imported input and partner, for the same class.</param>
    /// <param name="Elementwise">Whether the operation maps elements one to one. numpy returns a SCALAR for a 0-d input
    ///     of an elementwise ufunc where NumSharp returns a 0-d array, so for a 0-d input numpy computes over
    ///     <c>x[None]</c> and the result is taken back as <c>r[0, ...]</c> — a 0-d array of numpy's result dtype. A
    ///     non-elementwise operation (sort, unique, a reduction) runs on the 0-d input as it is, errors included.</param>
    public sealed record TextOp(string Name, Func<TextClass, string> Numpy, Func<TextClass, NDArray, NDArray, NDArray> Sharp,
        bool Elementwise = true);

    /// <summary>
    ///     The machinery the end-to-end text parity tests share: the Python helpers that compute and compare numpy's
    ///     answer, the numpy-born source data, the layouts and routes every input crosses through, and the cell runner
    ///     that holds NumSharp's result (carried back over the bridge) against numpy's.
    /// </summary>
    /// <remarks>
    ///     <para><b>What a cell proves.</b> numpy builds the input; the bridge hands it to NumSharp (a view of numpy's own
    ///     memory, or a copy); NumSharp computes; the bridge hands NumSharp's result back to numpy (a zero-copy view of a
    ///     fixed-width or numeric result, an exact copy of a <c>StringDType</c> one); and numpy compares it with what numpy
    ///     itself computes over the same input. Every hop of the bridge and the NumSharp operation are therefore inside
    ///     one assertion: a wrong byte anywhere along the way is a failed cell.</para>
    ///     <para><b>Equality.</b> A fixed-width or numeric result must have numpy's dtype string, shape and C-order bytes
    ///     (the NUL padding, the code-point width, a NaN's sign and payload); a <c>StringDType</c> result numpy's
    ///     descriptor (<c>repr</c>: the NA object and <c>coerce</c>), shape and values with NA identity (an NA is not
    ///     equal to a text that reads the same). When numpy raises, NumSharp must raise the exception class numpy's
    ///     most specific builtin class names (<see cref="KnownPythonErrors"/>) with numpy's message text.</para>
    ///     <para><b>Reporting.</b> A family runs every cell and fails once, listing every disagreement — a red run shows
    ///     the whole gap rather than its first cell.</para>
    /// </remarks>
    public abstract class TextLiveParityHarness : InteropTestBase
    {
        /// <summary>
        ///     The Python half of the harness: value tokens, the result description and comparison, and the guarded
        ///     numpy evaluation. Defined once per test in the test's own scope (each test gets a fresh scope).
        /// </summary>
        /// <remarks>
        ///     <c>_tok</c> encodes one element so NA kinds stay distinct from text (<c>N:None</c>, <c>N:nan</c>) and text
        ///     compares by its UTF-8 bytes with surrogates kept (<c>surrogatepass</c>: a U array may hold a lone surrogate
        ///     that a strict encode would refuse). <c>_np_run</c> returns <c>('ok', result)</c> or
        ///     <c>('err', mro, message)</c> where <c>mro</c> is the exception's class names, most derived first.
        /// </remarks>
        protected const string HarnessPython = """
            import math, os, tempfile
            from numpy.dtypes import StringDType

            def _tok(v):
                if v is None: return 'N:None'
                if isinstance(v, (float, np.floating)) and v != v: return 'N:nan'
                if isinstance(v, str): return 's:' + v.encode('utf-8', 'surrogatepass').hex()
                return 'o:' + repr(v)

            def _desc(a):
                if a.dtype.kind == 'T':
                    vals = a.ravel().tolist() if a.ndim else [a.item()]
                    return (repr(a.dtype), a.shape, tuple(_tok(v) for v in vals))
                return (a.dtype.str, a.shape, np.ascontiguousarray(a).tobytes())

            def _arr(r, like):
                if isinstance(r, np.ndarray): return r
                if like is not None and like.dtype.kind == 'T' and (r is None or isinstance(r, (str, float))):
                    return np.array(r, dtype=like.dtype)
                if like is not None and isinstance(r, (np.str_, np.bytes_)) and like.dtype.kind == np.asarray(r).dtype.kind:
                    # numpy hands a 0-d result back as a SCALAR, whose width is its own text's: the array it stands for
                    # has the width NumSharp's 0-d result reports (the value is still compared, padding included).
                    return np.asarray(r, dtype=like.dtype)
                return np.asarray(r)

            def _elem(a, i):
                return a.ravel()[i:i + 1] if a.ndim else a

            def _show(a, i):
                e = _elem(a, i)
                v = e.ravel()[0] if e.size else None
                s = repr(v)
                if e.dtype.kind in 'fc': s += ' (' + e.tobytes().hex() + ')'
                return s

            def _cmp(ns, ref):
                ref = _arr(ref, ns)
                dn, dr = _desc(ns), _desc(ref)
                if dn == dr: return ''
                out = []
                if dn[0] != dr[0]: out.append('dtype ' + str(dn[0]) + ' vs numpy ' + str(dr[0]))
                if dn[1] != dr[1]: out.append('shape ' + str(dn[1]) + ' vs numpy ' + str(dr[1]))
                if not out:
                    for i in range(ns.size):
                        if _desc(_elem(ns, i))[2] != _desc(_elem(ref, i))[2]:
                            out.append('element ' + str(i) + ': ' + _show(ns, i) + ' vs numpy ' + _show(ref, i))
                            break
                    else:
                        out.append('bytes differ')
                return '; '.join(out)

            def _np_run(code, x, p, zero_d):
                try:
                    if zero_d:
                        r = eval(code, globals(), {'x': x[None], 'p': p[None]})
                        r = r[0, ...]
                    else:
                        r = eval(code, globals(), {'x': x, 'p': p})
                    return ('ok', r)
                except Exception as e:
                    return ('err', ','.join(c.__name__ for c in type(e).__mro__), str(e))

            def _eager(r):
                # numpy's bytes -> StringDType cast copies bytes unchecked and raises UnicodeDecodeError only when an
                # element is read; reading every element (C order) moves that error to the cast, where NumSharp raises.
                if r.dtype.kind == 'T': r.tolist()
                return r

            def _npdesc(r):
                if isinstance(r, tuple): return 'a tuple of ' + str(len(r))
                a = _arr(r, None)
                return repr(a.dtype) + ' ' + str(a.shape)
            """;

        /// <summary>
        ///     The numpy-born sources: 24 texts per class. The texts cover case mapping that changes length or needs
        ///     context (<c>ß</c>, <c>İ</c>, a final sigma, the titlecase digraph <c>ǅ</c>), numeric classes (<c>Ⅻ</c>
        ///     numeric only, <c>٣</c> decimal, <c>²</c> digit), whitespace runs and tabs, an astral emoji, a combining mark,
        ///     an embedded NUL, the empty string and a full-width element; the bytes add high bytes that are not UTF-8,
        ///     a leading NUL and CR/LF. The <c>StringDType</c> sources add a text past the 15-byte short-string cell (an
        ///     arena string) and, for the NA source, two NA cells.
        /// </summary>
        protected const string SourcesPython = """
            _U = ['hello', 'Hello World', 'HELLO', 'hi there yo', '  padded  ', '\tTab\tbed', '123', '4.5e6', '-0',
                  '\u01c5ungla', '\u00df', '\u0130stanbul', '\u03a3\u038a\u03a3\u03a5\u03a6\u039f\u03a3', '\u216b',
                  '\u0663', '\u00b2', '\U0001F600 smile', '', 'a\x00b', 'MiXeD cAsE', 'x' * 12, '  ', 'Title Case',
                  'e\u0301']
            src_U = np.array(_U, dtype='U12')
            src_S = np.array([b'hello', b'Hello World', b'HELLO', b'hi there yo', b'  padded  ', b'\tTab\tbed', b'123',
                              b'4.5e6', b'-0', b'\x80\xffhigh', b'\xe9t\xe9', b'a\x00b', b'', b'MiXeD cAsE', b'x' * 12,
                              b'  ', b'Title Case', b'\x00lead', b'UPPER lower', b'12ab', b'\r\n', b'a,b,c', b'zz',
                              b'007'], dtype='S12')
            src_T = np.array(_U, dtype=StringDType())
            src_T[20] = 'x' * 40
            src_TN = np.array(_U, dtype=StringDType(na_object=None))
            src_TN[20] = 'a long text past the sixteen byte cell'
            src_TN[4] = None
            src_TN[17] = None
            """;

        /// <summary>
        ///     The numpy layouts of a 24-element source every family crosses: contiguous (C and F), transposed, a step that
        ///     ends at the buffer's LAST element (an over-read past the view would leave the buffer), reversed, reversed
        ///     rows with an inner slice, a broadcast (read-only, stride 0), an offset slice, 0-d, and two empty shapes.
        /// </summary>
        protected static readonly (string Name, string Expr)[] Layouts =
        {
            ("C 1-D", "src"),
            ("C 2-D", "src.reshape(4, 6)"),
            ("F 2-D", "np.asfortranarray(src.reshape(4, 6))"),
            ("transposed", "src.reshape(4, 6).T"),
            ("stepped to the end", "src[2::3]"),
            ("reversed", "src[::-1]"),
            ("reversed rows", "src.reshape(4, 6)[::-1, 1:5]"),
            ("broadcast", "np.broadcast_to(src[:6], (3, 6))"),
            ("offset slice", "src[13:]"),
            ("0-d", "src[7, ...]"),
            ("empty", "src[3:3]"),
            ("empty 2-D", "src.reshape(4, 6)[:, 2:2]"),
        };

        /// <summary>Every text class, in the order the families run them.</summary>
        protected static readonly TextClass[] AllClasses =
            { TextClass.Unicode, TextClass.Bytes, TextClass.StringDType, TextClass.StringDTypeNA };

        /// <summary>
        ///     numpy's builtin exception classes a NumSharp exception is matched by NAME, most specific first. numpy's own
        ///     private subclasses (<c>_UFuncNoLoopError</c>, <c>UFuncTypeError</c>) are skipped, so their builtin base
        ///     (<c>TypeError</c>) is the class NumSharp must raise.
        /// </summary>
        protected static readonly string[] KnownPythonErrors =
        {
            "DTypePromotionError", "AxisError", "UnicodeEncodeError", "UnicodeDecodeError", "UnicodeError", "IndexError",
            "KeyError", "OverflowError", "ZeroDivisionError", "NotImplementedError", "TypeError", "ValueError",
            "RuntimeError", "MemoryError",
        };

        /// <summary>The numpy source variable of a text class.</summary>
        /// <param name="cls">The class.</param>
        /// <returns><c>src_U</c>, <c>src_S</c>, <c>src_T</c> or <c>src_TN</c>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown class.</exception>
        protected static string SourceName(TextClass cls) => cls switch
        {
            TextClass.Unicode => "src_U",
            TextClass.Bytes => "src_S",
            TextClass.StringDType => "src_T",
            TextClass.StringDTypeNA => "src_TN",
            _ => throw new ArgumentOutOfRangeException(nameof(cls)),
        };

        /// <summary>
        ///     How a class can cross into NumSharp: a fixed-width array as a zero-copy VIEW of numpy's memory and as a COPY;
        ///     a <c>StringDType</c> array only as a copy (numpy keeps its text outside the buffer, in an allocator the
        ///     descriptor owns, so no view exists).
        /// </summary>
        /// <param name="cls">The class.</param>
        /// <returns><c>"view"</c> and <c>"copy"</c>, or <c>"copy"</c> alone.</returns>
        protected static string[] RoutesOf(TextClass cls)
            => cls is TextClass.Unicode or TextClass.Bytes ? new[] { "view", "copy" } : new[] { "copy" };

        /// <summary>
        ///     A Python literal of a text parameter for a class: <c>b'..'</c> for bytes, <c>'..'</c> otherwise (a
        ///     <c>StringDType</c> operand takes a Python <c>str</c>, which numpy promotes to the operand's descriptor).
        /// </summary>
        /// <param name="cls">The class the literal is an operand for.</param>
        /// <param name="text">The literal's text; non-printable and non-ASCII characters are escaped.</param>
        /// <returns>The literal.</returns>
        protected static string Q(TextClass cls, string text) => (cls == TextClass.Bytes ? "b" : "") + PyQuote(text);

        /// <summary>
        ///     The NumSharp twin of <see cref="Q"/>: a 0-d <c>S</c> array for bytes, the implicit Python-<c>str</c> literal
        ///     otherwise (a weak scalar, as numpy treats a Python <c>str</c>).
        /// </summary>
        /// <param name="cls">The class the literal is an operand for.</param>
        /// <param name="text">The literal's text (ASCII for bytes).</param>
        /// <returns>The 0-d literal array.</returns>
        protected static NDArray L(TextClass cls, string text)
            => cls == TextClass.Bytes ? np.array(text, np.dtype("S")) : (NDArray)text;

        /// <summary>A single-quoted Python string literal of a C# string, escaping what Python source cannot carry raw.</summary>
        /// <param name="text">The text.</param>
        /// <returns>The literal.</returns>
        protected static string PyQuote(string text)
        {
            var sb = new StringBuilder("'");
            foreach (char c in text)
            {
                // Everything outside printable ASCII is escaped, so the literal reads the same in any source encoding
                // and a surrogate half (one UTF-16 unit) is written as Python's \uXXXX rather than as a broken pair.
                if (c == '\\' || c == '\'')
                    sb.Append('\\').Append(c);
                else if (c >= 0x20 && c < 0x7F)
                    sb.Append(c);
                else if (c <= 0xFF)
                    sb.Append("\\x").Append(((int)c).ToString("x2"));
                else
                    sb.Append("\\u").Append(((int)c).ToString("x4"));
            }

            return sb.Append('\'').ToString();
        }

        /// <summary>An elementwise operation whose numpy expression needs no class-specific literal.</summary>
        /// <param name="name">The cell label.</param>
        /// <param name="numpy">The numpy expression over <c>x</c> / <c>p</c>.</param>
        /// <param name="sharp">NumSharp's computation over the input.</param>
        /// <returns>The operation.</returns>
        protected static TextOp E(string name, string numpy, Func<NDArray, NDArray> sharp)
            => new(name, _ => numpy, (_, x, _) => sharp(x));

        /// <summary>An elementwise operation over the input and its partner (<c>x[::-1]</c>).</summary>
        /// <param name="name">The cell label.</param>
        /// <param name="numpy">The numpy expression over <c>x</c> and <c>p</c>.</param>
        /// <param name="sharp">NumSharp's computation over the input and the partner.</param>
        /// <returns>The operation.</returns>
        protected static TextOp E2(string name, string numpy, Func<NDArray, NDArray, NDArray> sharp)
            => new(name, _ => numpy, (_, x, p) => sharp(x, p));

        /// <summary>A non-elementwise operation (it runs on a 0-d input as it is).</summary>
        /// <param name="name">The cell label.</param>
        /// <param name="numpy">The numpy expression over <c>x</c> / <c>p</c>.</param>
        /// <param name="sharp">NumSharp's computation over the input and the partner.</param>
        /// <returns>The operation.</returns>
        protected static TextOp W(string name, string numpy, Func<NDArray, NDArray, NDArray> sharp)
            => new(name, _ => numpy, (_, x, p) => sharp(x, p), Elementwise: false);

        /// <summary>
        ///     An elementwise operation with one text literal, written once: <paramref name="numpyFormat"/>'s <c>{0}</c> is
        ///     the class's Python literal and <paramref name="sharp"/> receives the class's NumSharp literal.
        /// </summary>
        /// <param name="name">The cell label.</param>
        /// <param name="numpyFormat">The numpy expression with <c>{0}</c> for the literal.</param>
        /// <param name="literal">The literal's text.</param>
        /// <param name="sharp">NumSharp's computation over the input and the literal.</param>
        /// <returns>The operation.</returns>
        protected static TextOp Lit(string name, string numpyFormat, string literal, Func<NDArray, NDArray, NDArray> sharp)
            => new(name, cls => string.Format(numpyFormat, Q(cls, literal)), (cls, x, _) => sharp(x, L(cls, literal)));

        /// <summary>An elementwise operation with two text literals (<c>{0}</c> and <c>{1}</c>).</summary>
        /// <param name="name">The cell label.</param>
        /// <param name="numpyFormat">The numpy expression with <c>{0}</c> and <c>{1}</c>.</param>
        /// <param name="first">The first literal's text.</param>
        /// <param name="second">The second literal's text.</param>
        /// <param name="sharp">NumSharp's computation over the input and both literals.</param>
        /// <returns>The operation.</returns>
        protected static TextOp Lit2(string name, string numpyFormat, string first, string second,
            Func<NDArray, NDArray, NDArray, NDArray> sharp)
            => new(name, cls => string.Format(numpyFormat, Q(cls, first), Q(cls, second)),
                (cls, x, _) => sharp(x, L(cls, first), L(cls, second)));

        /// <summary>Defines the harness helpers and every numpy source in the test's scope.</summary>
        /// <param name="extraPython">Further Python (more sources) to run after the standard ones, or null.</param>
        protected void DefineSources(string extraPython = null)
        {
            PyExec(HarnessPython);
            PyExec(SourcesPython);
            if (extraPython != null)
                PyExec(extraPython);
        }

        /// <summary>
        ///     One input a family runs over: a label for the failure lines, the class whose literals its operations take,
        ///     the numpy source variable the layouts slice, and the routes it crosses by.
        /// </summary>
        /// <param name="Label">The input's label (a text class's name, or a numeric source's).</param>
        /// <param name="Class">The text class whose literals the operations use (a numeric source takes
        ///     <see cref="TextClass.Unicode"/>'s, which its operations do not read).</param>
        /// <param name="Source">The numpy variable the layout expressions slice (<c>src</c> is replaced by it).</param>
        /// <param name="Routes"><c>"view"</c> and/or <c>"copy"</c>.</param>
        protected readonly record struct FamilyInput(string Label, TextClass Class, string Source, string[] Routes);

        /// <summary>The standard inputs of a set of text classes: each class's source and routes.</summary>
        /// <param name="classes">The classes; null = <see cref="AllClasses"/>.</param>
        /// <param name="sourceOf">Maps a class to its numpy source; null = <see cref="SourceName"/>.</param>
        /// <returns>One input per class.</returns>
        protected static IEnumerable<FamilyInput> InputsOf(IEnumerable<TextClass> classes = null,
            Func<TextClass, string> sourceOf = null)
            => (classes ?? AllClasses).Select(c => new FamilyInput(c.ToString(), c, (sourceOf ?? SourceName)(c), RoutesOf(c)));

        /// <summary>
        ///     Runs every operation over every class × layout × route and fails with the list of every disagreement.
        /// </summary>
        /// <param name="family">The family's label (the first field of each failure line).</param>
        /// <param name="ops">The operations.</param>
        /// <param name="classes">The classes to run; null = <see cref="AllClasses"/>.</param>
        /// <param name="layouts">The layouts to run; null = <see cref="Layouts"/>.</param>
        /// <param name="sourceOf">Maps a class to its numpy source; null = <see cref="SourceName"/>.</param>
        /// <param name="messageGap">Cells (by their failure label) whose exception TEXT is a pinned, known divergence:
        ///     the class must still match, only the message comparison is waived — the pin lives in an
        ///     <c>[OpenBugs]</c> / <c>[Misaligned]</c> test that names it. Null = every message is compared.</param>
        /// <returns>The number of cells run (so a caller can assert the matrix is not vacuous).</returns>
        protected int RunFamily(string family, IReadOnlyList<TextOp> ops, IEnumerable<TextClass> classes = null,
            IEnumerable<(string Name, string Expr)> layouts = null, Func<TextClass, string> sourceOf = null,
            Func<string, bool> messageGap = null)
            => RunFamily(family, ops, InputsOf(classes, sourceOf), layouts, messageGap);

        /// <summary>
        ///     Runs every operation over every input × layout × route and fails with the list of every disagreement.
        /// </summary>
        /// <param name="family">The family's label (the first field of each failure line).</param>
        /// <param name="ops">The operations.</param>
        /// <param name="inputs">The inputs (text classes or numeric sources).</param>
        /// <param name="layouts">The layouts to run; null = <see cref="Layouts"/>.</param>
        /// <param name="messageGap">Cells whose exception text is a pinned divergence (see the other overload).</param>
        /// <returns>The number of cells run.</returns>
        /// <remarks>
        ///     The input crosses ONCE per input × layout × route; every operation of the family reads that one import, so
        ///     an operation that wrote into its input would show in the cells after it — none of these do (they all
        ///     return new arrays or views), which is part of what a passing run proves.
        /// </remarks>
        protected int RunFamily(string family, IReadOnlyList<TextOp> ops, IEnumerable<FamilyInput> inputs,
            IEnumerable<(string Name, string Expr)> layouts = null, Func<string, bool> messageGap = null)
        {
            var failures = new List<string>();
            int cells = 0;
            foreach (FamilyInput input in inputs)
            foreach (var (layoutName, layoutExpr) in layouts ?? Layouts)
            foreach (string route in input.Routes)
            {
                PyExec($"x = {layoutExpr.Replace("src", input.Source)}");
                bool zeroD = PyBool("x.ndim == 0");

                // The import: a view of numpy's memory (read-only layouts too: the bridge marks the view
                // non-writeable) or an owning copy. The partner is derived on the NumSharp side, so NumSharp's own
                // slicing of the import is part of the cell.
                using NDArray xs = route == "view" ? ViewOf("x", allowReadonly: true) : ImportOf("x");

                // numpy's twin is the array NumSharp holds. A view IS numpy's x; an import copy is C-contiguous by the
                // bridge's contract (ToNDArray), so numpy computes over x's C-ordered copy — the same values in the
                // same memory order. The order is observable: a cast that raises names the first failing element of
                // its MEMORY walk (an F-ordered x and its C copy name different elements), and a result keeps its
                // input's layout.
                if (route == "copy")
                    PyExec("x = x.copy(order='C')");
                PyExec("p = x[::-1] if x.ndim else x");
                NDArray ps = xs.ndim > 0 ? xs["::-1"] : xs;
                foreach (TextOp op in ops)
                {
                    cells++;
                    string label = $"{family} | {input.Label} | {layoutName} | {route} | {op.Name}";
                    string verdict = RunCell(op, input.Class, xs, ps, zeroD && op.Elementwise);

                    // A pinned text divergence waives the message only: a verdict that is anything but "same class,
                    // different text" still fails.
                    if (verdict.Length > 0 && !(messageGap != null && verdict.StartsWith(TextDiffers, StringComparison.Ordinal) &&
                                                messageGap(label)))
                        failures.Add($"{label}: {verdict}");
                }

                if (!ReferenceEquals(ps, xs))
                    ps.Dispose();
            }

            PyExec("x = p = None");
            AssertNoFailures(family, cells, failures);
            return cells;
        }

        /// <summary>The verdict prefix of a cell whose exception classes agree and whose texts do not.</summary>
        protected const string TextDiffers = "both raise ";

        /// <summary>
        ///     Fails with EVERY disagreeing cell, one per line (an assertion library's collection message shows only the
        ///     first item, which hides the shape of a gap: one layout, one class, or everything).
        /// </summary>
        /// <param name="family">The family's label.</param>
        /// <param name="cells">How many cells ran.</param>
        /// <param name="failures">The disagreeing cells.</param>
        /// <exception cref="AssertFailedException">At least one cell disagreed.</exception>
        protected static void AssertNoFailures(string family, int cells, IReadOnlyList<string> failures)
        {
            if (failures.Count == 0)
                return;
            const int shown = 400;
            Assert.Fail($"{failures.Count} of {cells} {family} cells disagree with numpy:\n" +
                        string.Join("\n", failures.Take(shown)) +
                        (failures.Count > shown ? $"\n... and {failures.Count - shown} more" : ""));
        }

        /// <summary>
        ///     One operand PAIR of a two-operand family: the two numpy expressions (bound to <c>x</c> and <c>p</c>) and how
        ///     each crosses into NumSharp.
        /// </summary>
        /// <param name="Label">The pair's label in a failure line.</param>
        /// <param name="Class">The text class whose literals the operations take (the first operand's).</param>
        /// <param name="X">The numpy expression of the first operand.</param>
        /// <param name="P">The numpy expression of the second operand (any class, any shape that may broadcast).</param>
        /// <param name="XRoute"><c>"view"</c> or <c>"copy"</c> for the first operand.</param>
        /// <param name="PRoute"><c>"view"</c> or <c>"copy"</c> for the second operand.</param>
        protected readonly record struct OperandPair(string Label, TextClass Class, string X, string P, string XRoute, string PRoute);

        /// <summary>
        ///     Runs every operation over every operand PAIR — operands of two different text classes, widths, NA objects or
        ///     shapes (broadcasting) — and fails with the list of every disagreement. The operations read the second
        ///     operand as their partner (<c>p</c> / the delegate's third argument).
        /// </summary>
        /// <param name="family">The family's label.</param>
        /// <param name="ops">The operations (each reads <c>x</c> and <c>p</c>).</param>
        /// <param name="pairs">The operand pairs.</param>
        /// <returns>The number of cells run.</returns>
        /// <remarks>
        ///     A pair's operands cross independently (a view of one next to a copy of the other is a legitimate user
        ///     situation); a copy route's numpy twin is the operand's C-ordered copy, as in <see cref="RunFamily(string, IReadOnlyList{TextOp}, IEnumerable{FamilyInput}, IEnumerable{ValueTuple{string, string}}, Func{string, bool})"/>.
        /// </remarks>
        protected int RunPairFamily(string family, IReadOnlyList<TextOp> ops, IEnumerable<OperandPair> pairs)
        {
            var failures = new List<string>();
            int cells = 0;
            foreach (OperandPair pair in pairs)
            {
                PyExec($"x = {pair.X}\np = {pair.P}");
                bool zeroD = PyBool("x.ndim == 0 and p.ndim == 0");
                using NDArray xs = pair.XRoute == "view" ? ViewOf("x", allowReadonly: true) : ImportOf("x");
                using NDArray ps = pair.PRoute == "view" ? ViewOf("p", allowReadonly: true) : ImportOf("p");
                if (pair.XRoute == "copy")
                    PyExec("x = x.copy(order='C')");
                if (pair.PRoute == "copy")
                    PyExec("p = p.copy(order='C')");
                foreach (TextOp op in ops)
                {
                    cells++;
                    string verdict = RunCell(op, pair.Class, xs, ps, zeroD && op.Elementwise);
                    if (verdict.Length > 0)
                        failures.Add($"{family} | {pair.Label} | {op.Name}: {verdict}");
                }
            }

            PyExec("x = p = None");
            AssertNoFailures(family, cells, failures);
            return cells;
        }

        /// <summary>
        ///     The layouts NumSharp itself makes of its own 24-element array (the export direction): contiguous C and F, the
        ///     transpose, a step that ends at the last element, reversed, reversed rows with an inner slice, a broadcast
        ///     (stride 0), the child of an <c>np.split</c> (a view at a non-zero offset into its parent's buffer), 0-d by
        ///     integer indexing, and two empty shapes.
        /// </summary>
        protected static readonly (string Name, Func<NDArray, NDArray> Make)[] NumSharpLayouts =
        {
            ("C 1-D", b => b),
            ("C 2-D", b => b.reshape(4, 6)),
            ("F 2-D", b => np.asfortranarray(b.reshape(4, 6))),
            ("transposed", b => b.reshape(4, 6).T),
            ("stepped to the end", b => b["2::3"]),
            ("reversed", b => b["::-1"]),
            ("reversed rows", b => b.reshape(4, 6)["::-1, 1:5"]),
            ("broadcast", b => np.broadcast_to(b["0:6"], new Shape(3, 6))),
            ("split child (offset view)", b => np.split(b, 4)[2]),
            ("0-d", b => b["7"]),
            ("empty", b => b["3:3"]),
            ("empty 2-D", b => b.reshape(4, 6)[":, 2:2"]),
        };

        /// <summary>
        ///     The EXPORT direction of a family: NumSharp owns the text (an owning copy of the class's numpy source), makes
        ///     each of <see cref="NumSharpLayouts"/> by its own slicing, and hands the result to numpy — a zero-copy view of
        ///     a fixed-width array, an exact copy of a <c>StringDType</c> one; numpy computes over what it was handed,
        ///     NumSharp over its own array, and the answers are compared as in <see cref="RunFamily(string, IReadOnlyList{TextOp}, IEnumerable{TextClass}, IEnumerable{ValueTuple{string, string}}, Func{TextClass, string}, Func{string, bool})"/>.
        /// </summary>
        /// <param name="family">The family's label.</param>
        /// <param name="ops">The operations.</param>
        /// <param name="classes">The classes; null = <see cref="AllClasses"/>.</param>
        /// <returns>The number of cells run.</returns>
        /// <remarks>
        ///     This proves the export's dtype, shape, strides and offset at once: numpy's kernels run over NumSharp's memory
        ///     as numpy reads it, so a wrong stride or a dropped offset (the split child) gives numpy other elements than
        ///     NumSharp computed over, and the cell fails.
        /// </remarks>
        protected int RunFamilyFromNumSharp(string family, IReadOnlyList<TextOp> ops, IEnumerable<TextClass> classes = null)
        {
            var failures = new List<string>();
            int cells = 0;
            foreach (TextClass cls in classes ?? AllClasses)
            {
                // NumSharp-born: an owning NumSharp copy of the class's source (no numpy memory behind it).
                using NDArray b = ImportOf(SourceName(cls));
                foreach (var (layoutName, make) in NumSharpLayouts)
                {
                    NDArray xs = make(b);
                    if (cls is TextClass.Unicode or TextClass.Bytes)
                        ExportTo("x", xs);
                    else
                        ExportCopyTo("x", xs);
                    PyExec("p = x[::-1] if x.ndim else x");
                    bool zeroD = xs.ndim == 0;
                    NDArray ps = xs.ndim > 0 ? xs["::-1"] : xs;
                    foreach (TextOp op in ops)
                    {
                        cells++;
                        string verdict = RunCell(op, cls, xs, ps, zeroD && op.Elementwise);
                        if (verdict.Length > 0)
                            failures.Add($"{family} | {cls} | {layoutName} | export | {op.Name}: {verdict}");
                    }

                    // The export holds NumSharp's buffer until numpy releases it: drop numpy's names first.
                    PyExec("x = p = None");
                    if (!ReferenceEquals(ps, xs))
                        ps.Dispose();
                    if (!ReferenceEquals(xs, b))
                        xs.Dispose();
                }
            }

            AssertNoFailures(family, cells, failures);
            return cells;
        }

        /// <summary>
        ///     Runs one cell: NumSharp computes (outside the GIL), numpy computes over the same input, and the two answers
        ///     are compared — NumSharp's result crossing back to numpy for the comparison.
        /// </summary>
        /// <param name="op">The operation.</param>
        /// <param name="cls">The input's class (picks the literals).</param>
        /// <param name="xs">NumSharp's import of <c>x</c>.</param>
        /// <param name="ps">NumSharp's partner of <paramref name="xs"/>.</param>
        /// <param name="zeroD">Whether numpy computes over <c>x[None]</c> and takes <c>[0, ...]</c> (an elementwise
        ///     operation on a 0-d input).</param>
        /// <returns>An empty string when the two agree, else what differs.</returns>
        protected string RunCell(TextOp op, TextClass cls, NDArray xs, NDArray ps, bool zeroD)
        {
            NDArray result = null;
            Exception error = null;
            try
            {
                result = op.Sharp(cls, xs, ps);
            }
            catch (Exception e)
            {
                error = e;
            }

            try
            {
                return Judge(op.Numpy(cls), zeroD, result, error);
            }
            finally
            {
                // A result that IS an operand (a no-op view routine may return its input) must not be disposed here:
                // the import is the next operation's input.
                if (result is not null && !ReferenceEquals(result, xs) && !ReferenceEquals(result, ps))
                    result.Dispose();
            }
        }

        /// <summary>
        ///     Evaluates numpy's expression over the scope's <c>x</c> / <c>p</c> and compares NumSharp's outcome with it.
        /// </summary>
        /// <param name="numpyCode">The numpy expression.</param>
        /// <param name="zeroD">Whether numpy computes over <c>x[None]</c> and takes <c>[0, ...]</c>.</param>
        /// <param name="result">NumSharp's result, or null when it raised.</param>
        /// <param name="error">NumSharp's exception, or null when it returned.</param>
        /// <returns>An empty string when the two agree, else what differs.</returns>
        protected string Judge(string numpyCode, bool zeroD, NDArray result, Exception error)
        {
            using (Gil())
            {
                using (PyObject code = new PyString(numpyCode))
                    Scope.Set("_code", code);
                Scope.Exec($"_r = _np_run(_code, x, p, {(zeroD ? "True" : "False")})");
                bool numpyOk;
                using (PyObject status = Scope.Eval("_r[0]"))
                    numpyOk = status.As<string>() == "ok";

                if (numpyOk && error is null)
                {
                    if (result is null)
                        return "NumSharp returned null";

                    // NumSharp's result crosses back: a zero-copy view of a fixed-width or numeric result (the
                    // comparison reads NumSharp's memory through the bridge), an exact copy of a StringDType one.
                    using PyObject exported = result.dtype.kind == 'T' ? result.ToNumpyCopy() : result.ToNumpy();
                    Scope.Set("_ns", exported);
                    try
                    {
                        using PyObject diff = Scope.Eval("_cmp(_ns, _r[1])");
                        return diff.As<string>();
                    }
                    finally
                    {
                        Scope.Exec("del _ns");
                    }
                }

                if (numpyOk)
                {
                    using PyObject desc = Scope.Eval("_npdesc(_r[1])");
                    return $"NumSharp raises {error.GetType().Name}: {error.Message} -- numpy returns {desc.As<string>()}";
                }

                string mro, message;
                using (PyObject m = Scope.Eval("_r[1]"))
                    mro = m.As<string>();
                using (PyObject m = Scope.Eval("_r[2]"))
                    message = m.As<string>();
                if (error is null)
                    return $"numpy raises {mro.Split(',')[0]}: {message} -- NumSharp returns {result?.dtype} {result?.Shape}";
                return CompareErrors(mro, message, error);
            }
        }

        /// <summary>
        ///     Holds NumSharp's exception to numpy's: the class numpy's most specific known builtin class names must appear
        ///     in NumSharp's exception type chain (a .NET builtin standing in for <c>OverflowError</c>,
        ///     <c>ZeroDivisionError</c>, <c>MemoryError</c>, <c>NotImplementedError</c> or <c>KeyError</c> counts, and so
        ///     do the two documented house conventions of <see cref="IsHouseConvention"/>), and the message text must be
        ///     numpy's (<see cref="NumpyFacingMessage"/>: .NET's argument decoration cut).
        /// </summary>
        /// <param name="mro">numpy's exception class names, most derived first, comma-separated.</param>
        /// <param name="message">numpy's <c>str(e)</c>.</param>
        /// <param name="error">NumSharp's exception.</param>
        /// <returns>An empty string when class and text agree, else what differs.</returns>
        protected static string CompareErrors(string mro, string message, Exception error)
        {
            string[] chain = mro.Split(',');
            string pyClass = chain.FirstOrDefault(n => KnownPythonErrors.Contains(n)) ?? chain[0];
            var names = new HashSet<string>();
            for (Type t = error.GetType(); t != null; t = t.BaseType)
                names.Add(t.Name);
            bool classMatches = names.Contains(pyClass) || pyClass switch
            {
                "OverflowError" => error is OverflowException,
                "ZeroDivisionError" => error is DivideByZeroException,
                "MemoryError" => error is OutOfMemoryException,
                "NotImplementedError" => error is NotImplementedException,
                "KeyError" => error is KeyNotFoundException,
                _ => false,
            } || IsHouseConvention(chain, error);
            if (!classMatches)
                return $"numpy raises {chain[0]} ({pyClass}): {message} -- NumSharp raises {error.GetType().Name}: {error.Message}";
            string text = NumpyFacingMessage(error);
            if (text != message)
                return $"{TextDiffers}{pyClass}, text differs -- NumSharp: \"{text}\" -- numpy: \"{message}\"";
            return "";
        }

        /// <summary>
        ///     The two documented NumSharp house conventions for an exception CLASS, where the text is still numpy's: a
        ///     <c>UFuncTypeError</c> cast refusal ("Cannot cast ufunc … with casting rule …") is an
        ///     <see cref="ArgumentException"/> (the parity journal names it "the house ArgumentException the rules name
        ///     for ufunc cast refusals"), and a shape <c>ValueError</c> is an <see cref="IncorrectShapeException"/>
        ///     ("library-wide", <c>.claude/rules/axis-arguments.md</c>).
        /// </summary>
        /// <param name="chain">numpy's exception class names, most derived first.</param>
        /// <param name="error">NumSharp's exception.</param>
        /// <returns>Whether the pair is one of the two conventions (the caller still compares the texts).</returns>
        protected static bool IsHouseConvention(string[] chain, Exception error)
            => (chain.Contains("UFuncTypeError") && error is ArgumentException) ||
               (chain.Contains("ValueError") && error is IncorrectShapeException);

        /// <summary>
        ///     The part of a NumSharp exception's message that answers numpy's <c>str(e)</c>: .NET decorates an
        ///     <see cref="ArgumentException"/>'s <c>Message</c> with <c> (Parameter 'name')</c> and an
        ///     <see cref="ArgumentOutOfRangeException"/>'s with <c>Actual value was …</c> after it — NumSharp's
        ///     <c>AxisError</c> derives from the latter, and the parity journal documents the decoration (the rules
        ///     assert axis texts with wildcards). Everything from the decoration on is cut; any other text is compared
        ///     whole.
        /// </summary>
        /// <param name="error">NumSharp's exception.</param>
        /// <returns>The message numpy's text is compared with.</returns>
        protected static string NumpyFacingMessage(Exception error)
        {
            string message = error.Message;
            if (error is ArgumentException argument && !string.IsNullOrEmpty(argument.ParamName))
            {
                int at = message.IndexOf($" (Parameter '{argument.ParamName}')", StringComparison.Ordinal);
                if (at >= 0)
                    message = message.Substring(0, at);
            }

            return message;
        }
    }
}
