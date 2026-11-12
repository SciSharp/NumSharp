using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static NumSharp.Tests.Interop.TorchSharp.StateMatrixSupport;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     NumSharp → torch → NumSharp for every array state and shared dtype, in Auto both ways: the values come back, a
    ///     state the bridge shares comes back as a view of the SAME memory with the layout's flags, and one it copies comes
    ///     back independent.
    /// </summary>
    [TestClass]
    public class StateMatrixRoundTripTests : TorchTestBase
    {
        /// <summary>Every array state name of the fixture, one test case each.</summary>
        public static IEnumerable<object[]> States => StateMatrixExportTests.States;

        /// <summary>One array state, every dtype, a round trip through torch in Auto.</summary>
        /// <param name="state">The state's name.</param>
        [TestMethod]
        [DynamicData(nameof(States))]
        public void ArrayState_RoundTripsThroughTorch(string state)
        {
            var failures = new Failures();
            foreach (string dtype in StateOracle.ArrayStates[state].Keys)
            {
                string where = $"{state} [{dtype}]";
                try
                {
                    using var scope = Scope();
                    NDArray nd = StateCatalog.ArrayStates[state](dtype);
                    bool shared = StateCatalog.ExportBlocker(nd) is null && nd.size > 0;
                    using Tensor t = nd.AsTensor();
                    using NDArray back = t.AsNDArray();
                    failures.Check(back.shape.SequenceEqual(nd.shape), $"{where}: shape ({string.Join(",", back.shape)})");
                    failures.Check(back.typecode == nd.typecode, $"{where}: dtype {back.typecode}");
                    failures.Check(SameBits(back, t) && BytesOf(back).AsSpan().SequenceEqual(BytesOf(nd)), $"{where}: the values did not come back");
                    failures.Check(np.may_share_memory(back, nd) == shared, $"{where}: may_share_memory(back, source) {!shared}, expected {shared}");
                    if (shared)
                    {
                        // The same memory, described by torch's strides: contiguity follows them, the repeated axes of a
                        // broadcast stay read-only, and nothing is owned.
                        StateOracle.Flags flags = StateOracle.Flags.Of(back);
                        failures.Check(flags.C == t.is_contiguous() || nd.size <= 1, $"{where}: C {flags.C}, is_contiguous {t.is_contiguous()}");
                        failures.Check(flags.W == !HasRepeatedAxis(back.shape, back.Shape.strides), $"{where}: writeable {flags.W}");
                        failures.Check(!flags.O, $"{where}: a view of torch's storage claims to own its data");
                        failures.Check(SameStrides(nd.shape, back.Shape.strides, t.stride()), $"{where}: strides ({string.Join(",", back.Shape.strides)}) are not torch's");
                        if (flags.W)
                        {
                            NumSharpWriteFirst(back, Other(FirstValue(t)));
                            failures.Check(BytesOf(back).AsSpan().SequenceEqual(BytesOf(nd)), $"{where}: a write through the round trip did not reach the source");
                        }
                    }
                }
                catch (Exception ex)
                {
                    failures.Unexpected(where, ex);
                }
            }
            failures.None.Should().BeTrue(failures.ToString());
        }
    }
}
