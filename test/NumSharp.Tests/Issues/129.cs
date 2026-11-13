using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #129, "Doc : Better specification for all classes (What class has what task)"
    ///     (https://github.com/SciSharp/NumSharp/issues/129): ARCHITECTURE.md and the Under the hood page describe
    ///     <c>NDArray</c>, <c>UnmanagedStorage</c> (the data), <c>Shape</c> (dimensions, strides, offset, flags) and
    ///     <c>TensorEngine</c> (the computation), and the code has that split.
    /// </summary>
    [TestClass]
    public class Issue129
    {
        /// <summary>Both documents the answer links name all four classes.</summary>
        [TestMethod]
        public void ArchitectureDocuments_NameTheFourClasses()
        {
            foreach (var doc in new[] { "ARCHITECTURE.md", "docs/website-src/docs/advanced/under-the-hood.md" })
            {
                var text = IssueAnswerSupport.ReadRepoFile(doc);
                foreach (var cls in new[] { "NDArray", "UnmanagedStorage", "Shape", "TensorEngine" })
                    text.Should().Contain(cls, $"{doc} documents what {cls} owns");
            }
        }

        /// <summary>
        ///     The split the answer describes is the code's: an <c>NDArray</c> holds an <c>UnmanagedStorage</c> (the data and
        ///     its address), a <c>Shape</c> (dimensions, strides, offset and contiguity flags) and the <c>TensorEngine</c> that
        ///     computes on it.
        /// </summary>
        [TestMethod]
        public unsafe void Classes_OwnWhatTheAnswerSays()
        {
            var a = np.arange(6.0).reshape(2, 3);

            UnmanagedStorage storage = a.Storage;
            ((IntPtr)storage.Address).Should().NotBe(IntPtr.Zero);

            Shape shape = a.Shape;
            shape.dimensions.Should().Equal(2L, 3L);
            shape.strides.Should().Equal(3L, 1L);
            shape.offset.Should().Be(0);
            shape.IsContiguous.Should().BeTrue();

            a.TensorEngine.Should().BeSameAs(TensorEngine.Default);
        }
    }
}
