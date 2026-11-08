namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #433, "NDArray exists in both NumSharp.Core and NumSharp.Lite"
    ///     (https://github.com/SciSharp/NumSharp/issues/433) as not planned: the same TensorFlow.NET <c>NumSharp.Lite</c> clash
    ///     as #412, with the same answer.
    /// </summary>
    [TestClass]
    public class Issue433
    {
        /// <summary>
        ///     NumSharp's own assemblies define <c>NumSharp.NDArray</c> once (see <see cref="Issue412"/>).
        /// </summary>
        [TestMethod]
        public void NDArray_IsDefinedOnce_AcrossNumSharpsAssemblies()
        {
            Issue412.AssembliesDefiningNDArray().Should().Equal(typeof(NDArray).Assembly.GetName().Name);
        }
    }
}
