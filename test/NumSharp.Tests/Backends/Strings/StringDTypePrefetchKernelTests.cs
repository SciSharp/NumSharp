using System;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The prefetching form of the <c>StringDType</c> ordering kernels (<c>&lt;</c>, <c>&lt;=</c>, <c>&gt;</c>,
    ///     <c>&gt;=</c>): for long texts in arenas past the caches the kernel asks for the texts of the pair
    ///     <see cref="NDVStringCells.TextPrefetchDistance"/> elements ahead before it compares the current one. These tests hold
    ///     that the form changes no result (against the plain kernel and against <c>NpyString_cmp</c> itself, over every cell
    ///     kind, layout, route and operand order), that it reads no cell outside the chunk it was given (fenced memory on both
    ///     sides of every walk), that an equality never takes it, and that the automatic choice follows its documented
    ///     thresholds.
    /// </summary>
    /// <remarks>
    ///     <see cref="NDVStringCells.TextPrefetchOverride"/> forces one form or the other on the test's own thread, and every
    ///     call checks <see cref="NDVStringComparison.LastKernel"/>, the kernel the driver actually ran (a choice the driver
    ///     dropped would otherwise hide behind equal values); both hooks are thread-static, so the tests run in parallel with
    ///     the rest of the suite.
    /// </remarks>
    [TestClass]
    public unsafe class StringDTypePrefetchKernelTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor (no NA object).</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor: every ordering with an NA is False.</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>The four orderings, the only comparisons with a prefetching form.</summary>
        private static readonly string[] Orderings = { "less", "less_equal", "greater", "greater_equal" };

        /// <summary>
        ///     A random text of one of the cell kinds: short (in the cell), medium (arena, one-byte size prefix), long (arena,
        ///     eight-byte prefix), empty, multi-byte UTF-8, a NUL inside, and texts sharing long prefixes (so the first eight
        ///     bytes often tie and the slow path decides).
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <returns>The text.</returns>
        private static string Text(System.Random rng) => rng.Next(8) switch
        {
            0 => "s" + rng.Next(100),
            1 => "m" + new string((char)('a' + rng.Next(3)), rng.Next(16, 60)),
            2 => "L" + new string((char)('a' + rng.Next(3)), rng.Next(300, 500)) + rng.Next(10),
            3 => "",
            4 => "é\U0001F600" + rng.Next(10),
            5 => "a\0" + rng.Next(3),
            6 => "common-prefix-" + new string('x', rng.Next(0, 40)) + rng.Next(3),
            _ => new string('z', rng.Next(1, 200)),
        };

        /// <summary>
        ///     <paramref name="n"/> random texts in a new <c>StringDType</c> array, every seventh element then overwritten by a
        ///     longer text, which outgrows its slot and goes to the heap pages (so heap cells are in the mix).
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The element count.</param>
        /// <param name="texts">The final texts, in order.</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray Texts(System.Random rng, int n, out string[] texts)
        {
            texts = Enumerable.Range(0, n).Select(_ => Text(rng)).ToArray();
            var a = np.array(texts, T);
            for (int i = 0; i < n; i += 7)
            {
                texts[i] = texts[i] + new string('h', 40 + rng.Next(600)) + i;
                a.SetValue(texts[i], i);
            }
            return a;
        }

        /// <summary>
        ///     NumPy's <c>NpyString_cmp</c>: <c>strncmp</c> over the shorter length (it stops at the first differing byte or at
        ///     a NUL both texts share), then the sizes.
        /// </summary>
        /// <param name="s1">The first text.</param>
        /// <param name="s2">The second text.</param>
        /// <returns>-1, 0 or 1.</returns>
        private static int NpyStringCmp(string s1, string s2)
        {
            byte[] a = Encoding.UTF8.GetBytes(s1), b = Encoding.UTF8.GetBytes(s2);
            int min = System.Math.Min(a.Length, b.Length);
            for (int k = 0; k < min; k++)
            {
                if (a[k] != b[k])
                    return a[k] < b[k] ? -1 : 1;
                if (a[k] == 0)
                    break;
            }
            return System.Math.Sign(a.Length.CompareTo(b.Length));
        }

        /// <summary>What an ordering answers for a <c>NpyString_cmp</c> result.</summary>
        /// <param name="ufunc">The ordering's name.</param>
        /// <param name="cmp">The comparison of the first operand against the second.</param>
        /// <returns>The bool.</returns>
        /// <exception cref="ArgumentException">Not an ordering.</exception>
        private static bool Orders(string ufunc, int cmp) => ufunc switch
        {
            "less" => cmp < 0,
            "less_equal" => cmp <= 0,
            "greater" => cmp > 0,
            "greater_equal" => cmp >= 0,
            _ => throw new ArgumentException(ufunc),
        };

        /// <summary>
        ///     The IL kernel a comparison runs for a form: the (operation, operand order) the driver maps the ufunc to
        ///     (<c>greater</c> / <c>greater_equal</c> are <c>less</c> / <c>less_equal</c> of the swapped pair), in the
        ///     prefetching form or the plain one.
        /// </summary>
        /// <param name="ufunc">The comparison's name.</param>
        /// <param name="prefetch">True for the prefetching form (orderings only).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="ArgumentException">An unknown name.</exception>
        private static NDInnerLoopFunc ExpectedKernel(string ufunc, bool prefetch) => ufunc switch
        {
            "equal" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Equal, 0),
            "not_equal" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.NotEqual, 0),
            "less" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Less, 0, prefetch),
            "less_equal" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.LessEqual, 0, prefetch),
            "greater" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Less, 1, prefetch),
            "greater_equal" => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.LessEqual, 1, prefetch),
            _ => throw new ArgumentException(ufunc),
        };

        /// <summary>
        ///     One comparison ufunc under a kernel-form override (restored afterwards): its values in C order, after checking
        ///     that the driver ran the expected kernel.
        /// </summary>
        /// <param name="ufunc">The comparison's name.</param>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <param name="form">The value of <see cref="NDVStringCells.TextPrefetchOverride"/> for the call (0 automatic,
        ///     1 prefetching, 2 plain).</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="expectPrefetch">Whether an ordering must have run the prefetching kernel; null reads it from
        ///     <paramref name="form"/> (1 yes, otherwise no). An equality always runs the plain kernel.</param>
        /// <returns>The result's values.</returns>
        /// <exception cref="ArgumentException">An unknown name.</exception>
        private static bool[] Run(string ufunc, NDArray a, NDArray b, int form, NDArray @out = null, NDArray where = null,
                                  bool? expectPrefetch = null)
        {
            NDVStringComparison.LastKernel = null;
            NDArray r;
            NDVStringCells.TextPrefetchOverride = form;
            try
            {
                r = ufunc switch
                {
                    "equal" => np.equal(a, b, @out: @out, where: where),
                    "not_equal" => np.not_equal(a, b, @out: @out, where: where),
                    "less" => np.less(a, b, @out: @out, where: where),
                    "less_equal" => np.less_equal(a, b, @out: @out, where: where),
                    "greater" => np.greater(a, b, @out: @out, where: where),
                    "greater_equal" => np.greater_equal(a, b, @out: @out, where: where),
                    _ => throw new ArgumentException(ufunc),
                };
            }
            finally
            {
                NDVStringCells.TextPrefetchOverride = 0;
            }
            try
            {
                if (ILKernelGenerator.TextKernelsAvailable && r.size > 0)
                {
                    bool ordering = ufunc is not ("equal" or "not_equal");
                    bool prefetch = ordering && (expectPrefetch ?? form == 1);
                    ReferenceEquals(NDVStringComparison.LastKernel, ExpectedKernel(ufunc, prefetch))
                        .Should().BeTrue($"np.{ufunc} under form {form} runs the {(prefetch ? "prefetching" : "plain")} kernel");
                }
                return r.ToArray<bool>();
            }
            finally
            {
                if (!ReferenceEquals(r, @out))
                    r.Dispose();
            }
        }

        /// <summary>
        ///     Runs every ordering over (<paramref name="x"/>, <paramref name="y"/>) under both forms and requires the same
        ///     values; with <paramref name="where"/> the same <c>out=</c> contents too (the masked-off slots keep the fill).
        /// </summary>
        /// <param name="x">The first operand.</param>
        /// <param name="y">The second operand.</param>
        /// <param name="what">The case, for the failure message.</param>
        /// <param name="where">A mask (an <c>out=</c> of the mask's shape is then passed, pre-filled with True), or null.</param>
        private static void AssertBothFormsAgree(NDArray x, NDArray y, string what, NDArray where = null)
        {
            foreach (var u in Orderings)
            {
                bool[] plain, pre;
                if (where is null)
                {
                    plain = Run(u, x, y, 2);
                    pre = Run(u, x, y, 1);
                }
                else
                {
                    using var o1 = np.ones(where.Shape.dimensions, NPTypeCode.Boolean);
                    using var o2 = np.ones(where.Shape.dimensions, NPTypeCode.Boolean);
                    plain = Run(u, x, y, 2, o1, where);
                    pre = Run(u, x, y, 1, o2, where);
                }
                pre.Should().Equal(plain, $"{what}: np.{u} under the prefetching kernel");
            }
        }

        /// <summary>
        ///     The orderings under the prefetching kernel give the plain kernel's values, and NpyString_cmp's, over random texts
        ///     of every cell kind (short, medium, long, heap, empty, multi-byte, NUL inside) for lengths around the prefetch
        ///     distance, through the trivial loop (contiguous, reversed, stepped, a one-element broadcast operand, a <c>U</c>
        ///     operand on either side) and the iterator (a C operand against an F one, a <c>where=</c> mask with an
        ///     <c>out=</c>); every call checks the kernel that ran.
        /// </summary>
        [TestMethod]
        public void Orderings_PrefetchingKernel_GivesThePlainKernelsValues()
        {
            foreach (int n in new[] { 1, 2, 15, 16, 17, 31, 32, 33, 64, 257, 1000 })
            {
                var rng = new System.Random(1000 + n);
                using var a = Texts(rng, n, out var ta);
                using var b = Texts(rng, n, out var tb);

                // The element reference on the contiguous pair, for both forms.
                foreach (var u in Orderings)
                foreach (int form in new[] { 1, 2 })
                    Run(u, a, b, form).Should().Equal(Enumerable.Range(0, n).Select(i => Orders(u, NpyStringCmp(ta[i], tb[i]))),
                        $"n={n} np.{u} form {form} against NpyString_cmp");

                AssertBothFormsAgree(a, b, $"n={n} contiguous");
                using (var ra = a["::-1"])
                    AssertBothFormsAgree(ra, b, $"n={n} reversed A");
                using (var rb = b["::-1"])
                    AssertBothFormsAgree(a, rb, $"n={n} reversed B");
                using (var sa = a["::2"])
                using (var sb = b["::2"])
                    AssertBothFormsAgree(sa, sb, $"n={n} stepped");
                using (var one = np.array(new[] { "L" + new string('b', 400) }, T))
                {
                    AssertBothFormsAgree(a, one, $"n={n} broadcast B");
                    AssertBothFormsAgree(one, a, $"n={n} broadcast A");
                }
                using (var u = np.array(tb))
                {
                    AssertBothFormsAgree(a, u, $"n={n} (T, U)");
                    AssertBothFormsAgree(u, a, $"n={n} (U, T)");
                }
                if (n % 4 == 0)
                {
                    using var ca = a.reshape(4, n / 4);
                    using var fb = b.reshape(n / 4, 4).T;
                    AssertBothFormsAgree(ca, fb, $"n={n} C against F (iterator)");
                }
                using (var mask = np.array(Enumerable.Range(0, n).Select(_ => rng.Next(3) != 0).ToArray()))
                    AssertBothFormsAgree(a, b, $"n={n} where=", mask);
            }
        }

        /// <summary>
        ///     An NA cell of a NaN-like descriptor makes every ordering False under both forms (the prefetch skips the NA cell,
        ///     which holds no text).
        /// </summary>
        [TestMethod]
        public void Orderings_WithNaNLikeNA_AreFalseUnderBothForms()
        {
            var rng = new System.Random(5);
            const int n = 100;
            var values = Enumerable.Range(0, n).Select(i => i % 5 == 0 ? (object)double.NaN : "L" + new string('q', 300 + rng.Next(100)) + i).ToArray();
            var others = Enumerable.Range(0, n).Select(i => i % 3 == 0 ? (object)double.NaN : "L" + new string('q', 300 + rng.Next(100)) + (n - i)).ToArray();
            using var a = np.array(values, TNaN);
            using var b = np.array(others, TNaN);
            foreach (var u in Orderings)
            foreach (int form in new[] { 1, 2 })
            {
                var got = Run(u, a, b, form);
                for (int i = 0; i < n; i++)
                    if (values[i] is double || others[i] is double)
                        got[i].Should().BeFalse($"np.{u} form {form} with an NA at {i}");
            }
            AssertBothFormsAgree(a, b, "NaN-like NA");
        }

        /// <summary>
        ///     An equality has no prefetching form: <see cref="NDVStringCells.TakesPrefetchingKernel"/> answers false for
        ///     <c>==</c> / <c>!=</c> even when a test forces the form, the kernel factory refuses the combination, and
        ///     <c>np.equal</c> / <c>np.not_equal</c> under the forcing override run the plain kernel.
        /// </summary>
        [TestMethod]
        public void Equality_HasNoPrefetchingForm()
        {
            var rng = new System.Random(9);
            using var a = Texts(rng, 64, out _);
            using var b = Texts(rng, 64, out _);
            var sa = a.Storage.VStringSlice;
            var sb = b.Storage.VStringSlice;
            NDVStringCells.TextPrefetchOverride = 1;
            try
            {
                NDVStringCells.TakesPrefetchingKernel(TextCompareOp.Equal, sa, 64, sb, 64).Should().BeFalse();
                NDVStringCells.TakesPrefetchingKernel(TextCompareOp.NotEqual, sa, 64, sb, 64).Should().BeFalse();
                NDVStringCells.TakesPrefetchingKernel(TextCompareOp.Less, sa, 64, sb, 64).Should().BeTrue();
                NDVStringCells.TakesPrefetchingKernel(TextCompareOp.LessEqual, sa, 64, sb, 64).Should().BeTrue();
            }
            finally
            {
                NDVStringCells.TextPrefetchOverride = 0;
            }
            Action equal = () => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Equal, 0, prefetch: true);
            equal.Should().Throw<ArgumentException>();
            Action notEqual = () => ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.NotEqual, 1, prefetch: true);
            notEqual.Should().Throw<ArgumentException>();
            foreach (var u in new[] { "equal", "not_equal" })
                Run(u, a, b, 1).Should().Equal(Run(u, a, b, 2), $"np.{u} forced");
        }

        /// <summary>
        ///     The prefetching kernel exists wherever the text kernels do (an emission failure would fall back to the managed
        ///     loop and hide behind equal values), it is cached, it is a different kernel from the plain one, and its body
        ///     holds the look-ahead: its IL is longer than the plain kernel's. A kernel built without the look-ahead computes
        ///     the same values and reads nothing outside the cells, so only its size tells it apart (a mutation that skipped
        ///     the emission passed every other test).
        /// </summary>
        [TestMethod]
        public void PrefetchingKernel_IsEmitted()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Text kernels are not available on this host (no dynamic code).");
            foreach (var op in new[] { TextCompareOp.Less, TextCompareOp.LessEqual })
            foreach (int aIndex in new[] { 0, 1 })
            {
                var pre = ILKernelGenerator.GetVStringCompareKernel(op, aIndex, prefetch: true);
                var plain = ILKernelGenerator.GetVStringCompareKernel(op, aIndex, prefetch: false);
                pre.Should().NotBeNull($"{op} a{aIndex}");
                plain.Should().NotBeNull();
                ReferenceEquals(pre, plain).Should().BeFalse();
                ReferenceEquals(ILKernelGenerator.GetVStringCompareKernel(op, aIndex, prefetch: true), pre).Should().BeTrue("cached");
                // A delegate of a DynamicMethod exposes the method, whose generator still reports the emitted IL length.
                pre.Method.Should().BeOfType<DynamicMethod>();
                plain.Method.Should().BeOfType<DynamicMethod>();
                ((DynamicMethod)pre.Method).GetILGenerator().ILOffset
                    .Should().BeGreaterThan(((DynamicMethod)plain.Method).GetILGenerator().ILOffset, $"{op} a{aIndex}: the look-ahead is emitted");
            }
        }

        /// <summary>
        ///     <paramref name="n"/> long texts (a fill character repeated 300 times between a marker and a random number) in a
        ///     new <c>StringDType</c> array: every text in the arena, on lines of its own.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The element count.</param>
        /// <param name="fill">The repeated character (texts of a smaller fill order first).</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray LongTexts(System.Random rng, int n, char fill)
            => np.array(Enumerable.Range(0, n).Select(_ => "L" + new string(fill, 300) + rng.Next()).ToArray(), T);

        /// <summary>
        ///     The automatic choice (<see cref="NDVStringCells.PrefetchesTexts"/>): an ordering of long texts (on average at
        ///     least <see cref="NDVStringCells.TextPrefetchMinTextBytes"/> arena bytes a cell) that touches at least
        ///     <see cref="NDVStringCells.TextPrefetchMinTexts"/> distinct texts takes the prefetching kernel — two arrays of
        ///     10,000 long texts do, two of 5,000 do not, one array of 10,000 against its own reverse touches its texts once and
        ///     does not, an array of 20,000 against a one-element operand does; medium texts (sharing lines) and short texts (in
        ///     their cells) never do; an equality never does. On a host without a prefetch instruction (non-x86) nothing does.
        ///     Held through the model and through the ufuncs (the kernel that ran).
        /// </summary>
        [TestMethod]
        public void AutomaticChoice_IsLongTextsTouchedPastL2()
        {
            bool host = Sse.IsSupported;
            var rng = new System.Random(11);
            using var la = LongTexts(rng, 10_000, 'x');
            using var lb = LongTexts(rng, 10_000, 'y');
            using var fewA = LongTexts(rng, 5_000, 'x');
            using var fewB = LongTexts(rng, 5_000, 'y');
            using var many = LongTexts(rng, 20_000, 'x');
            using var one = LongTexts(rng, 1, 'y');
            const int mediumCount = 120_000;
            using var ma = np.array(Enumerable.Range(0, mediumCount).Select(_ => "m" + new string('x', 40) + rng.Next()).ToArray(), T);
            using var mb = np.array(Enumerable.Range(0, mediumCount).Select(_ => "m" + new string('y', 40) + rng.Next()).ToArray(), T);
            const int shortCount = 30_000;
            using var sa = np.array(Enumerable.Range(0, shortCount).Select(i => "s" + (i % 1000)).ToArray(), T);
            using var sb = np.array(Enumerable.Range(0, shortCount).Select(i => "t" + (i % 1000)).ToArray(), T);

            NDVStringCells.PrefetchesTexts(la.Storage.VStringSlice, 10_000, lb.Storage.VStringSlice, 10_000).Should().Be(host, "20,000 long texts");
            NDVStringCells.PrefetchesTexts(fewA.Storage.VStringSlice, 5_000, fewB.Storage.VStringSlice, 5_000)
                .Should().BeFalse("10,000 texts: their first lines fit in L2");
            NDVStringCells.PrefetchesTexts(la.Storage.VStringSlice, 10_000, la.Storage.VStringSlice, 10_000)
                .Should().BeFalse("one buffer's 10,000 texts, counted once");
            NDVStringCells.PrefetchesTexts(many.Storage.VStringSlice, 20_000, one.Storage.VStringSlice, 1).Should().Be(host, "20,001 long texts");
            NDVStringCells.PrefetchesTexts(ma.Storage.VStringSlice, mediumCount, mb.Storage.VStringSlice, mediumCount)
                .Should().BeFalse("medium texts share lines");
            NDVStringCells.PrefetchesTexts(sa.Storage.VStringSlice, shortCount, sb.Storage.VStringSlice, shortCount)
                .Should().BeFalse("short texts live in their cells");

            Run("less", la, lb, 0, expectPrefetch: host).Should().AllBeEquivalentTo(true, "'x' orders before 'y'");
            Run("less", la, lb, 2).Should().AllBeEquivalentTo(true, "the plain form forced where the automatic choice prefetches");
            Run("greater_equal", la, lb, 0, expectPrefetch: host).Should().AllBeEquivalentTo(false);
            Run("less", fewA, fewB, 0, expectPrefetch: false).Should().AllBeEquivalentTo(true);
            using (var rev = la["::-1"])
                Run("greater", la, rev, 0, expectPrefetch: false).Length.Should().Be(10_000);
            Run("less", many, one, 0, expectPrefetch: host).Should().AllBeEquivalentTo(true);
            Run("greater", one, many, 0, expectPrefetch: host).Should().AllBeEquivalentTo(true);
            Run("less", ma, mb, 0, expectPrefetch: false).Should().AllBeEquivalentTo(true);
            Run("less", sa, sb, 0, expectPrefetch: false).Should().AllBeEquivalentTo(true);
            Run("equal", la, lb, 0).Should().AllBeEquivalentTo(false);
        }

        /// <summary>
        ///     The prefetching kernel reads no cell outside the chunk it was given: the operand's cells lie in
        ///     <see cref="FencedMemory"/> against an inaccessible page (the last cell directly below one, or the first cell
        ///     directly above one), and the operand is walked both forward and reversed, so every walk meets a fence at its
        ///     start or at its end — a read past the chunk or behind it ends the test host. Lengths around the prefetch
        ///     distance, either side of the comparison, the trivial loop, the iterator, and the one-element operand broadcast
        ///     against a long one (stride 0); the values are checked against the plain kernel too.
        /// </summary>
        [TestMethod]
        public void PrefetchingKernel_StaysInsideTheChunk()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            foreach (int n in new[] { 1, 2, 15, 16, 17, 33, 100 })
            foreach (bool atEnd in new[] { true, false })
                WithFencedArray(n, atEnd, (t, texts) =>
                {
                    var rng = new System.Random(n * 2 + (atEnd ? 1 : 0));
                    using var other = Texts(rng, n, out _);
                    using var forward = t[":"];
                    using var reversed = t["::-1"];
                    using var otherForward = other[":"];
                    using var otherReversed = other["::-1"];
                    string where = $"n={n} fence {(atEnd ? "after the last cell" : "before the first cell")}";
                    AssertBothFormsAgree(forward, otherForward, $"{where}: fenced A forward");
                    AssertBothFormsAgree(reversed, otherReversed, $"{where}: fenced A reversed");
                    AssertBothFormsAgree(otherForward, forward, $"{where}: fenced B forward");
                    AssertBothFormsAgree(otherReversed, reversed, $"{where}: fenced B reversed");
                    if (n % 4 == 0)
                    {
                        using var m = t.reshape(4, n / 4);
                        using var f = other.reshape(n / 4, 4).T;
                        AssertBothFormsAgree(m, f, $"{where}: iterator");
                    }
                    if (n == 1)
                    {
                        // The fenced cell broadcast against a long operand (its stride is 0): the prefetch of its pair ahead
                        // must not step off the one cell.
                        using var many = Texts(rng, 100, out _);
                        AssertBothFormsAgree(many, forward, $"{where}: fenced B broadcast");
                        AssertBothFormsAgree(forward, many, $"{where}: fenced A broadcast");
                    }
                    for (int i = 0; i < n; i++)
                        ((string)t.GetValue(i)).Should().Be(texts[i]);
                });
        }

        /// <summary>
        ///     Builds a <c>StringDType</c> array of <paramref name="n"/> elements whose cells lie in fenced memory (texts of
        ///     every kind, every fifth one rewritten longer so it moves to the heap pages), runs <paramref name="check"/> on it,
        ///     then frees the strings (the cells' block wraps foreign memory and does not own the allocator).
        /// </summary>
        /// <param name="n">The element count.</param>
        /// <param name="atEnd">True: the last cell ends at the upper fence; false: the first cell starts at the lower one.</param>
        /// <param name="check">The comparisons to run, given the array and its texts.</param>
        private static void WithFencedArray(int n, bool atEnd, Action<NDArray, string[]> check)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            using var fence = new FencedMemory(pages: (n * cell + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes + 1);
            byte* cells = fence.Place(n * (long)cell, atEnd);
            new Span<byte>(cells, n * cell).Clear(); // all-zero cells are empty strings
            var allocator = new VStringAllocator();
            var slice = new VStringArraySlice(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(cells, n * (long)cell)), allocator, np.dtypes.StringDType());
            try
            {
                var rng = new System.Random(77 + n);
                var texts = Enumerable.Range(0, n).Select(_ => Text(rng)).ToArray();
                for (int i = 0; i < n; i++)
                    slice.SetIndex(i, texts[i]);
                for (int i = 0; i < n; i += 5)
                {
                    texts[i] += new string('h', 50 + rng.Next(500));
                    slice.SetIndex(i, texts[i]);
                }
                var t = new NDArray(new UnmanagedStorage(slice, new Shape(n)));
                check(t, texts);
            }
            finally
            {
                allocator.ClearAndFree(cells, n);
            }
        }
    }
}
