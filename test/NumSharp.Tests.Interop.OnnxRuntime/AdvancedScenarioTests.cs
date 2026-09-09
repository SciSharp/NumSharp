using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Realistic "this happens in the wild" scenarios: concurrent inference on one session, a contiguous
    ///     slice fed zero-copy, a broadcast input (rejected low-level but copied by Tier-2 Run), a high-rank
    ///     tensor, and a reused session across many runs — each must be correct and leak-free.
    /// </summary>
    [TestClass]
    public class AdvancedScenarioTests : OnnxTestBase
    {
        [TestMethod]
        public void ParallelInferenceOnOneSession_IsThreadSafe_AndLeakFree()
        {
            using InferenceSession session = OpenSession("identity_float64");
            var errors = new ConcurrentBag<Exception>();

            Parallel.For(0, 200, i =>
            {
                try
                {
                    using NDArray x = (np.arange(6).astype(NPTypeCode.Double, copy: true) + i);
                    using NDArray y = session.Run(x);
                    if (y.GetDouble(5) != 5 + i)
                        throw new Exception($"iteration {i}: got {y.GetDouble(5)}");
                }
                catch (Exception e) { errors.Add(e); }
            });

            errors.Should().BeEmpty("ORT InferenceSession.Run is thread-safe and the interop uses per-call handles");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void ContiguousSliceInput_IsFedZeroCopy_InLogicalOrder()
        {
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray m = Arange(NPTypeCode.Double, 4, 3);
            using NDArray rows = m["1:3"];   // contiguous window (rows 1..2), non-zero offset
            rows.flags.c_contiguous.Should().BeTrue("a leading-axis slice stays C-contiguous, so Run shares it without a copy");

            using NDArray y = session.Run(rows);
            y.shape.Should().Equal(2, 3);
            using NDArray expected = rows.copy();
            BytesOf(y).Should().Equal(BytesOf(expected));
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void BroadcastInput_IsRefusedZeroCopy_ButTier2RunCopiesIt()
        {
            using NDArray row = np.array(new[] { 1.0, 2.0, 3.0 });
            using NDArray bcast = np.broadcast_to(row, new Shape(4, 3));
            bcast.flags.c_contiguous.Should().BeFalse();
            bcast.flags.writeable.Should().BeFalse("a broadcast view is read-only");

            // low-level zero-copy export refuses a broadcast (stride-0) view outright
            new Action(() => bcast.AsOrtValue()).Should().Throw<InvalidOperationException>().WithMessage("*not C-contiguous*");

            // but the ergonomic Tier-2 Run materializes it in logical order and runs
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray y = session.Run(bcast);
            y.shape.Should().Equal(4, 3);
            y.GetDouble(0, 0).Should().Be(1.0);
            y.GetDouble(3, 2).Should().Be(3.0, "each broadcast row is [1,2,3]");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the C-order copy was released with its handle");
        }

        [TestMethod]
        public void HighRankTensor_RoundTrips()
        {
            using InferenceSession session = OpenSession("identity_int32");
            using NDArray x = Arange(NPTypeCode.Int32, 2, 2, 2, 2, 2);   // 5-D, 32 elements
            using NDArray y = session.Run(x);
            y.shape.Should().Equal(2, 2, 2, 2, 2);
            y.ndim.Should().Be(5);
            BytesOf(y).Should().Equal(BytesOf(x));
        }

        [TestMethod]
        public void ReusedSession_ManySequentialRuns_DoNotLeak()
        {
            using InferenceSession session = OpenSession("identity_float32");
            for (int i = 0; i < 40; i++)
            {
                using NDArray x = Arange(NPTypeCode.Single, 3, 4);
                using NDArray y = session.Run(x);
                y.GetSingle(2, 3).Should().Be(11f);
            }
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "no handle leaked across 40 runs");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void MixedDtypeMultiInput_EachFedAndReadBackCorrectly()
        {
            // add_f32 broadcasts two float inputs; feed a contiguous (2,3) and a (3,) — a common shape combo
            using InferenceSession session = OpenSession("add_f32");
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArray b = np.array(new[] { 100f, 200f, 300f });
            IReadOnlyDictionary<string, NDArray> outputs =
                session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
            using NDArray c = outputs["C"];
            using NDArray expected = a + b;
            BytesOf(c).Should().Equal(BytesOf(expected));
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }
    }
}
