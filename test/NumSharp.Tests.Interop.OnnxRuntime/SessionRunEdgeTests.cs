using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Tier-2 <c>Run</c> corners <see cref="SessionRunTests"/> leaves: the null-argument guards on every
    ///     overload, a null array inside the input map, an explicitly supplied <see cref="RunOptions"/>, a
    ///     non-contiguous pre-allocated output, a shape ORT itself rejects, and a zero-copy empty output.
    /// </summary>
    [TestClass]
    public class SessionRunEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void NullSession_And_NullArgs_ThrowArgumentNull_OnEveryOverload()
        {
            InferenceSession none = null;
            using NDArray nd = Arange(NPTypeCode.Int32, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", nd } };

            new Action(() => none.Run(nd)).Should().Throw<ArgumentNullException>();
            new Action(() => none.Run("X", nd, "Y")).Should().Throw<ArgumentNullException>();
            new Action(() => none.Run(inputs)).Should().Throw<ArgumentNullException>();
            new Action(() => none.Run(inputs, inputs)).Should().Throw<ArgumentNullException>();

            using InferenceSession session = OpenSession("identity_int32");
            new Action(() => session.Run(null, nd, "Y")).Should().Throw<ArgumentNullException>("inputName is null");
            new Action(() => session.Run("X", nd, null)).Should().Throw<ArgumentNullException>("outputName is null");
            new Action(() => session.Run((IReadOnlyDictionary<string, NDArray>)null)).Should().Throw<ArgumentNullException>();
            new Action(() => session.Run(inputs, (IReadOnlyDictionary<string, NDArray>)null)).Should().Throw<ArgumentNullException>("outputs is null");
        }

        [TestMethod]
        public void NullArrayInsideTheInputMap_NamesTheInput()
        {
            using InferenceSession session = OpenSession("identity_int32");
            new Action(() => session.Run(new Dictionary<string, NDArray> { { "X", null } }))
                .Should().Throw<ArgumentNullException>().WithMessage("*input 'X' is null*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ExplicitRunOptions_ArePassedThrough()
        {
            using InferenceSession session = OpenSession("identity_int32");
            using NDArray nd = Arange(NPTypeCode.Int32, 5);
            using var ro = new RunOptions { LogId = "edge-run" };
            using NDArray y = session.Run(nd, ro);
            y.shape.Should().Equal(5);
            y.GetInt32(4).Should().Be(4);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void PreallocatedOutput_NonContiguous_IsRefused()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray big = np.zeros(new Shape(3, 2), typeof(float));
            using NDArray transposed = big.T;   // (2,3) but F-contiguous / non-C-contiguous
            transposed.Shape.IsContiguous.Should().BeFalse();

            new Action(() => session.Run(new Dictionary<string, NDArray> { { "X", x } },
                    new Dictionary<string, NDArray> { { "Y1", transposed } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y1'*not C-contiguous*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the refused output pinned nothing");
        }

        [TestMethod]
        public void PreallocatedOutput_WrongShape_IsRejectedByOrt_AndHandlesAreReleased()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray wrong = np.zeros(new Shape(2, 4), typeof(float));   // identity output is (2,3)

            new Action(() => session.Run(new Dictionary<string, NDArray> { { "X", x } },
                    new Dictionary<string, NDArray> { { "Y1", wrong } }))
                .Should().Throw<Exception>("ORT checks the pre-allocated output shape itself");
            // the finally in Run disposed both the input and output handles despite the throw
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ZeroCopyOutputs_OfAnEmptyOutput_AreEmptyViews_WithNoLease()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray empty = np.zeros(new Shape(0, 3), typeof(float));
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(
                new Dictionary<string, NDArray> { { "X", empty } }, zeroCopyOutputs: true);

            using NDArray y = outputs["Y"];
            y.shape.Should().Equal(0, 3);
            y.typecode.Should().Be(NPTypeCode.Single);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "an empty output owns its (already disposed) value but holds no lease");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
