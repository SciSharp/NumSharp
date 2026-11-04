using System;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the runnable kernel examples of <c>docs/website-src/docs/NDIter.md</c>: the Layer 1
    ///     <c>ForEach</c> delegate, the Layer 2 struct kernels (<c>ExecuteGeneric</c> with the page's <c>HypotKernel</c>,
    ///     <c>ExecuteReducing</c> with its early-exit <c>AnyNonZero</c>). Each kernel is the page's code; the tests run it on
    ///     contiguous operands (the typed fast branch) and on a strided view (the byte-pointer slow branch), so both
    ///     branches the page shows are proven to compute <c>sqrt(a*a + b*b)</c>. The state-field listings, the IL emission
    ///     walkthrough and the decision tables of the page are descriptive and not run here.
    /// </summary>
    [TestClass]
    public unsafe class NDIterDocTests
    {
        /// <summary>The page's Layer 2 kernel, verbatim: a typed fast branch when every stride is 4 bytes, a byte-pointer
        /// branch for any other stride.</summary>
        private readonly unsafe struct HypotKernel : INDInnerLoop
        {
            /// <summary>Writes <c>sqrt(a*a + b*b)</c> for <paramref name="n"/> elements of the three operands.</summary>
            /// <param name="p">The three data pointers: a, b, the output.</param>
            /// <param name="s">The three byte strides.</param>
            /// <param name="n">The inner-loop length.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Execute(void** p, long* s, long n)
            {
                if (s[0] == 4 && s[1] == 4 && s[2] == 4)
                {
                    float* a = (float*)p[0], b = (float*)p[1], c = (float*)p[2];
                    for (long i = 0; i < n; i++)
                        c[i] = MathF.Sqrt(a[i] * a[i] + b[i] * b[i]);
                    return;
                }
                long sA = s[0], sB = s[1], sC = s[2];
                byte* pA = (byte*)p[0]; byte* pB = (byte*)p[1]; byte* pC = (byte*)p[2];
                for (long i = 0; i < n; i++)
                {
                    float av = *(float*)(pA + i * sA);
                    float bv = *(float*)(pB + i * sB);
                    *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
                }
            }
        }

        /// <summary>The page's early-exit reduction, verbatim: true at the first non-zero int32, which also stops the
        /// walk.</summary>
        private readonly unsafe struct AnyNonZero : INDReducingInnerLoop<bool>
        {
            /// <summary>Scans <paramref name="n"/> int32 elements; returns false (stop) after setting the accumulator at
            /// the first non-zero one.</summary>
            /// <param name="p">The operand's data pointer.</param>
            /// <param name="s">The operand's byte stride.</param>
            /// <param name="n">The inner-loop length.</param>
            /// <param name="acc">The accumulator: true once a non-zero element was seen.</param>
            /// <returns>Whether the iteration continues.</returns>
            public bool Execute(void** p, long* s, long n, ref bool acc)
            {
                long st = s[0]; byte* pt = (byte*)p[0];
                for (long i = 0; i < n; i++)
                    if (*(int*)(pt + i * st) != 0) { acc = true; return false; }
                return true;
            }
        }

        /// <summary>The three-operand iterator of the page's examples: two read-only float32 inputs and a write-only
        /// float32 output, with an external loop and no casting.</summary>
        /// <param name="a">The first input.</param>
        /// <param name="b">The second input.</param>
        /// <param name="c">The output.</param>
        /// <returns>The iterator (the caller disposes it).</returns>
        private static NDIterRef ThreeOperands(NDArray a, NDArray b, NDArray c) =>
            NDIterRef.MultiNew(3, new[] { a, b, c },
                NDIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });

        /// <summary>Layer 1: the page's <c>ForEach</c> lambda computes the hypotenuse on contiguous operands and, through
        /// its slow branch, on a strided view.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ForEach_DelegateKernel_BothBranches()
        {
            foreach (bool strided in new[] { false, true })
            {
                var a = strided ? np.array(new[] { 3f, 0f, 5f, 0f, 8f })["::2"] : np.array(new[] { 3f, 5f, 8f });
                var b = np.array(new[] { 4f, 12f, 15f });
                var c = np.empty(new Shape(3), np.float32);
                using (var iter = ThreeOperands(a, b, c))
                    iter.ForEach((ptrs, strides, count, _) =>
                    {
                        if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4)
                        {
                            float* pa = (float*)ptrs[0], pb = (float*)ptrs[1], pc = (float*)ptrs[2];
                            for (long i = 0; i < count; i++)
                                pc[i] = MathF.Sqrt(pa[i] * pa[i] + pb[i] * pb[i]);
                            return;
                        }
                        long sA = strides[0], sB = strides[1], sC = strides[2];
                        byte* bA = (byte*)ptrs[0]; byte* bB = (byte*)ptrs[1]; byte* bC = (byte*)ptrs[2];
                        for (long i = 0; i < count; i++)
                        {
                            float av = *(float*)(bA + i * sA);
                            float bv = *(float*)(bB + i * sB);
                            *(float*)(bC + i * sC) = MathF.Sqrt(av * av + bv * bv);
                        }
                    });
                c.ToArray<float>().Should().Equal(new[] { 5f, 13f, 17f }, strided ? "the slow branch" : "the fast branch");
            }
        }

        /// <summary>Layer 2: <c>ExecuteGeneric</c> with the page's <c>HypotKernel</c> struct gives the same results as the
        /// delegate, on both branches.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ExecuteGeneric_StructKernel_BothBranches()
        {
            foreach (bool strided in new[] { false, true })
            {
                var a = strided ? np.array(new[] { 3f, 0f, 5f, 0f, 8f })["::2"] : np.array(new[] { 3f, 5f, 8f });
                var b = np.array(new[] { 4f, 12f, 15f });
                var c = np.empty(new Shape(3), np.float32);
                using (var iter = ThreeOperands(a, b, c))
                    iter.ExecuteGeneric(default(HypotKernel));
                c.ToArray<float>().Should().Equal(new[] { 5f, 13f, 17f }, strided ? "the slow branch" : "the fast branch");
            }
        }

        /// <summary>Layer 2's early exit: <c>ExecuteReducing</c> with <c>AnyNonZero</c> finds a non-zero element and
        /// answers false for an all-zero array (the initial value).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ExecuteReducing_EarlyExit()
        {
            using (var iter = NDIterRef.New(np.array(new[] { 0, 0, 7, 0 }), NDIterGlobalFlags.EXTERNAL_LOOP))
                iter.ExecuteReducing<AnyNonZero, bool>(default, false).Should().BeTrue();
            using (var iter = NDIterRef.New(np.zeros(new Shape(5), np.int32), NDIterGlobalFlags.EXTERNAL_LOOP))
                iter.ExecuteReducing<AnyNonZero, bool>(default, false).Should().BeFalse();
        }
    }
}
