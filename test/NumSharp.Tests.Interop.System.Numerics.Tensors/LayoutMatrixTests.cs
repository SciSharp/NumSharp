using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Exhaustive combinatorial coverage of the export side: every NumSharp dtype (15) × every catalogued layout
    ///     (<see cref="Layouts.All{T}"/>, ~45) × every export verb, each cell checked against NumSharp's own element
    ///     reads (<see cref="ExportContract"/>). The point is the CROSS product — the hand-written tests pin each
    ///     behaviour on one or two dtypes, while a defect that only bites one element width on one layout (a 16-byte
    ///     element on a broadcast, a bool on a unit axis) would only show up here.
    /// </summary>
    [TestClass]
    public class LayoutMatrixTests : TensorsTestBase
    {
        /// <summary>
        ///     Every dtype × layout through AsTensorSpan (lengths, FlattenTo bytes, EVERY element through the BCL's own
        ///     strided indexer, writeability), write-through (each index written in turn must change exactly the
        ///     NumSharp element it names), ToTensor, and the three round trips back to NumSharp. A negative-stride
        ///     layout must be refused by AsTensorSpan and still copy through ToTensor.
        /// </summary>
        [TestMethod]
        public void EveryDtype_EveryLayout_EveryExportVerb()
        {
            var failures = new Failures();
            Dtypes.ForEach(new MatrixCheck(failures));
            failures.AssertNone("dtype x layout x verb");
        }

        /// <summary>Runs <see cref="ExportContract.CheckAll{T}"/> over every catalogued layout for one dtype.</summary>
        private sealed class MatrixCheck : IDtypeCheck
        {
            private readonly Failures _failures;

            /// <summary>Creates the check.</summary>
            /// <param name="failures">Where failing cells go.</param>
            public MatrixCheck(Failures failures) => _failures = failures;

            /// <inheritdoc />
            public void Run<T>(NPTypeCode code, Func<long, T> value) where T : unmanaged
            {
                foreach (var (name, build) in Layouts.All(value))
                {
                    string cell = $"{code} {name}";
                    NDArray view;
                    try
                    {
                        view = build();
                    }
                    catch (Exception e)
                    {
                        _failures.Add(cell, $"building the layout threw {e.GetType().Name}: {e.Message}");
                        continue;
                    }
                    ExportContract.CheckAll(_failures, cell, view, value);
                }
            }
        }
    }
}
