using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #412, "The type 'NDArray' exists in both 'NumSharp.Core' and 'NumSharp.Lite'"
    ///     (https://github.com/SciSharp/NumSharp/issues/412) as not planned: the clash came from TensorFlow.NET's
    ///     <c>NumSharp.Lite</c>, not from NumSharp, whose own assemblies define <c>NumSharp.NDArray</c> exactly once.
    /// </summary>
    [TestClass]
    public class Issue412
    {
        /// <summary>
        ///     Loads every <c>NumSharp*.dll</c> next to the tests and returns the names of those that define
        ///     <c>NumSharp.NDArray</c>.
        /// </summary>
        /// <returns>The defining assemblies' simple names, distinct.</returns>
        internal static string[] AssembliesDefiningNDArray() =>
            Directory.GetFiles(AppContext.BaseDirectory, "NumSharp*.dll")
                .Select(path =>
                {
                    // A test-output folder can hold native-dependency or reference-only files that fail to load as
                    // assemblies; they cannot define NDArray, so they are skipped.
                    try { return System.Reflection.Assembly.LoadFrom(path); }
                    catch (BadImageFormatException) { return null; }
                    catch (FileLoadException) { return null; }
                })
                .Where(a => a != null && a.GetType("NumSharp.NDArray", throwOnError: false) != null)
                .Select(a => a.GetName().Name)
                .Distinct()
                .ToArray();

        /// <summary>
        ///     Of all the NumSharp assemblies the test project deploys (Core, Bitmap, the interop bridges), only
        ///     <c>NumSharp</c> defines <c>NumSharp.NDArray</c>, so NumSharp's own packages cannot raise CS0433.
        /// </summary>
        [TestMethod]
        public void NDArray_IsDefinedOnce_AcrossNumSharpsAssemblies()
        {
            AssembliesDefiningNDArray().Should().Equal(typeof(NDArray).Assembly.GetName().Name);
        }
    }
}
