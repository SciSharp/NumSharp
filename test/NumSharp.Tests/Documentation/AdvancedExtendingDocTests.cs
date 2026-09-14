using System;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples in
    ///     <c>docs/website-src/docs/advanced/extending-numsharp.md</c> ("Extending NumSharp"). Each
    ///     test mirrors one documented snippet and asserts the behaviour the page claims. The C# 14
    ///     extension-members example compiles only under the .NET 10 SDK (C# 14), so it is guarded to
    ///     the net10.0 target framework. Values were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class AdvancedExtendingDocTests
    {
        [TestMethod]
        public void TypedIteration_NditerAndChunks()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });

            double sum = 0;
            foreach (ref double x in np.nditer<double>(a)) sum += x;
            sum.Should().Be(10.0);

            double chunkSum = 0;
            foreach (Span<double> chunk in np.nditer_chunks<double>(a))
                foreach (var e in chunk) chunkSum += e;
            chunkSum.Should().Be(10.0);
        }

        [TestMethod]
        public void FusedExpressions_Evaluate()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            var b = np.array(new[] { 10.0, 20.0, 30.0, 40.0 });

            np.evaluate((NDExpr)a * b + 2.0).ToArray<double>().Should().Equal(12.0, 42.0, 92.0, 162.0);
            ((double)np.evaluate(NDExpr.Sum((NDExpr)a * b))).Should().Be(300.0);
        }

        [TestMethod]
        public void Multithreading_Toggle()
        {
            // Enable then disable so the global setting is left off for other tests.
            try { np.multithreading(true, 4); }
            finally { np.multithreading(false); }
        }

        [TestMethod]
        public unsafe void Unsafe_SpanAndPointer()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            a.Unsafe.Span<double>().Length.Should().Be(4);
            double* p = a.Unsafe.Pointer<double>();
            p[0].Should().Be(1.0);
        }

#if NET10_0_OR_GREATER
        [TestMethod]
        public void CSharp14_ExtensionMembers_ExtendNpAndNDArray()
        {
            var x = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            ((double)np.sumsq(x)).Should().Be(30.0, "static extension member on the np type");
            x.doubled().ToArray<double>().Should().Equal(new[] { 2.0, 4.0, 6.0, 8.0 }, "instance extension method");
            x.total.Should().Be(10.0, "instance extension property");
        }
#endif
    }

#if NET10_0_OR_GREATER
    /// <summary>
    ///     The doc's C# 14 extension-members example: static members on the <c>np</c> type and instance
    ///     members on <see cref="NumSharp.NDArray"/>. The doc declares these in the <c>NumSharp</c>
    ///     namespace so a consumer's <c>using NumSharp;</c> resolves them as if built in; here they sit
    ///     in the test's own (enclosing) namespace, which resolves them the same way. <c>file</c>-scoped
    ///     so the members do not leak across the rest of the test assembly.
    /// </summary>
    file static class DocExtensionMembersForBeginners
    {
        extension(np)
        {
            public static NDArray sumsq(NDArray a) => np.sum(a * a);
        }

        extension(NDArray a)
        {
            public NDArray doubled() => a * 2.0;
            public double total => (double)np.sum(a);
        }
    }
#endif
}
