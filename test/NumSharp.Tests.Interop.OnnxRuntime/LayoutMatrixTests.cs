using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Exhaustive combinatorial coverage: every NumSharp dtype (15) × every catalogued layout (<see cref="Layouts.All{T}"/>)
    ///     × every export verb and round trip (<see cref="ExportContract"/>). The hand-written tests pin each behaviour on one or
    ///     two dtypes and a few layouts; the bugs this bridge had lived in the CROSS product (a Decimal array in a transposed
    ///     layout, an empty array through one verb), so here every cell is checked against NumSharp's own element reads and
    ///     ONNX Runtime's own tensor reads, and one assertion lists every failing cell.
    /// </summary>
    [TestClass]
    public class LayoutMatrixTests : OnnxTestBase
    {
        /// <summary>Every dtype × layout × verb cell of the export contract holds.</summary>
        [TestMethod]
        public void EveryDtype_EveryLayout_EveryVerb()
        {
            var failures = new Failures();
            Dtypes.ForEach(new Sweep(failures));
            failures.AssertNone("dtype x layout x verb");
        }

        /// <summary>The generic body of the sweep, run once per dtype.</summary>
        private sealed class Sweep : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the sweep.</summary>
            /// <param name="failures">Where failing cells go.</param>
            internal Sweep(Failures failures) => _failures = failures;

            /// <summary>Checks every catalogued layout of one dtype.</summary>
            /// <typeparam name="T">The element type.</typeparam>
            /// <param name="code">The dtype.</param>
            /// <param name="value">Its value generator.</param>
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                foreach (var (name, build) in Layouts.All(value))
                {
                    NDArray view;
                    try
                    {
                        view = build();
                    }
                    catch (Exception e)
                    {
                        _failures.Check($"{code} / {name} / build", () => $"building the layout threw {e.GetType().Name}: {e.Message}");
                        continue;
                    }
                    ExportContract.CheckAll(_failures, $"{code} / {name}", view, value);
                }
            }
        }
    }
}
