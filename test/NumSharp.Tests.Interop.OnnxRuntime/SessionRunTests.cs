using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>Tier 2: <c>InferenceSession.Run(...)</c> straight on NDArrays — names, coercion, shape validation, output modes.</summary>
    [TestClass]
    public class SessionRunTests : OnnxTestBase
    {
        [TestMethod]
        public void Run_SingleInputSingleOutput_NamesComeFromMetadata_EveryDtype()
        {
            foreach (NPTypeCode code in ZeroCopyDtypes)
            {
                using InferenceSession session = OpenSession(IdentityModel(code));
                using NDArray nd = Arange(code, 2, 3);
                using NDArray y = session.Run(nd);
                y.typecode.Should().Be(code);
                y.shape.Should().Equal(2, 3);
                BytesOf(y).Should().Equal(BytesOf(nd), $"{code}");
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void Run_Named_PicksOneOutput()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 4);
            using NDArray y2 = session.Run("X", x, "Y2");
            y2.GetSingle(3).Should().Be(-3f);
            using NDArray y1 = session.Run("X", x, "Y1");
            y1.GetSingle(3).Should().Be(3f);
        }

        [TestMethod]
        public void Run_Dictionary_ReturnsEveryOutputByName()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 2);
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "X", x } });
            outputs.Keys.Should().BeEquivalentTo("Y1", "Y2");
            using NDArray y1 = outputs["Y1"];
            using NDArray y2 = outputs["Y2"];
            BytesOf(y1).Should().Equal(BytesOf(x));
            y2.GetSingle(1, 1).Should().Be(-3f);

            IReadOnlyDictionary<string, NDArray> onlyY2 = session.Run(new Dictionary<string, NDArray> { { "X", x } }, new[] { "Y2" });
            onlyY2.Keys.Should().Equal("Y2");
            onlyY2["Y2"].Dispose();
        }

        [TestMethod]
        public void Run_TwoInputs_Broadcasting_MatchesNumSharp()
        {
            using InferenceSession session = OpenSession("add_f32");
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArray b = np.array(new[] { 10f, 20f, 30f });
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
            using NDArray c = outputs["C"];
            using NDArray expected = a + b;
            c.shape.Should().Equal(2, 3);
            BytesOf(c).Should().Equal(BytesOf(expected), "a single float add is exact on both sides");
        }

        [TestMethod]
        public void Run_MatMul_IsCloseToNumSharp()
        {
            using InferenceSession session = OpenSession("matmul_f32");
            np.random.seed(7);
            using NDArray a = np.random.randn(4, 6).astype(NPTypeCode.Single);
            using NDArray b = np.random.randn(6, 3).astype(NPTypeCode.Single);
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
            using NDArray c = outputs["C"];
            using NDArray expected = np.matmul(a, b);
            c.shape.Should().Equal(4, 3);
            np.allclose(c, expected, 1e-5, 1e-6).Should().BeTrue("ORT's MLAS sgemm and NumSharp's GEMM agree to float tolerance");
        }

        [TestMethod]
        public void Run_CoercesAnInt32ArrayIntoAnInt64Input_TheBertCase()
        {
            using InferenceSession session = OpenSession("int64_input");
            using NDArray ids = np.array(new[] { 101, 2023, 2003, 102 }).reshape(1, 4);
            ids.typecode.Should().Be(NPTypeCode.Int32);
            using NDArray outp = session.Run(ids);
            outp.typecode.Should().Be(NPTypeCode.Int64);
            outp.shape.Should().Equal(1, 4);
            outp.GetInt64(0, 1).Should().Be(2023L);
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the int64 temporary was released with its handle");
        }

        [TestMethod]
        public void Run_RefusesALossyCastUnderSafe_AndAllowsItUnderSameKind()
        {
            using InferenceSession session = OpenSession("identity_float16");
            using NDArray x = np.array(new[] { 1.0, 2.5, 1e-9 });
            new Action(() => session.Run(x)).Should().Throw<InvalidCastException>()
                .WithMessage("*'X'*Double*Half*casting='safe'*same_kind*");

            using NDArray y = session.Run(x, casting: "same_kind");
            y.typecode.Should().Be(NPTypeCode.Half);
            ((float)y.GetHalf(1)).Should().Be(2.5f);

            using NDArray z = session.Run(x, casting: "unsafe");
            z.typecode.Should().Be(NPTypeCode.Half);
        }

        [TestMethod]
        public void Run_ValidatesDeclaredShape_BeforeOrtSeesIt()
        {
            using InferenceSession session = OpenSession("fixed_shape_f32");   // data: (1,3,4,4)
            using NDArray wrongExtent = np.zeros(new Shape(1, 3, 4, 5), typeof(float));
            new Action(() => session.Run(wrongExtent)).Should().Throw<ArgumentException>()
                .WithMessage("*'data'*(1, 3, 4, 4)*dimension 3 must be 4, not 5*");

            using NDArray wrongRank = np.zeros(new Shape(3, 4, 4), typeof(float));
            new Action(() => session.Run(wrongRank)).Should().Throw<ArgumentException>()
                .WithMessage("*'data'*rank 4*rank 3*");

            using NDArray right = np.full(new Shape(1, 3, 4, 4), -1f);
            using NDArray relu = session.Run(right);
            relu.shape.Should().Equal(1, 3, 4, 4);
            np.any(relu).Should().BeFalse("relu of negatives is all zero");

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void Run_SymbolicDims_AreFree()
        {
            using InferenceSession session = OpenSession("int64_input");   // ids: (1, 'seq')
            InferenceSessionExtensions.DescribeDims(session.InputMetadata["ids"]).Should().Be("(1, 'seq')");
            using NDArray longSeq = Arange(NPTypeCode.Int64, 1, 9);
            using NDArray outp = session.Run(longSeq);
            outp.shape.Should().Equal(1, 9);

            using NDArray twoRows = Arange(NPTypeCode.Int64, 2, 9);
            new Action(() => session.Run(twoRows)).Should().Throw<ArgumentException>().WithMessage("*dimension 0 must be 1, not 2*");
        }

        [TestMethod]
        public void Run_UnknownInput_ListsTheModelsInputs()
        {
            using InferenceSession session = OpenSession("add_f32");
            using NDArray a = Arange(NPTypeCode.Single, 3);
            new Action(() => session.Run("Q", a, "C")).Should().Throw<ArgumentException>().WithMessage("*'Q'*[A, B]*");
            new Action(() => session.Run(a)).Should().Throw<InvalidOperationException>().WithMessage("*2 inputs*name them explicitly*");
        }

        [TestMethod]
        public void Run_NonContiguousInput_IsCopiedInLogicalOrder()
        {
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray m = Arange(NPTypeCode.Double, 3, 4);
            using NDArray t = m.T;
            using NDArray y = session.Run(t);
            y.shape.Should().Equal(4, 3);
            using NDArray expected = t.copy();
            BytesOf(y).Should().Equal(BytesOf(expected));
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the C-order temporary was released");
        }

        [TestMethod]
        public void Run_ZeroCopyOutputs_AreViewsThatOwnTheirOrtValues()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "X", x } }, zeroCopyOutputs: true);
            NDArrayOnnxInterop.LiveImports.Should().Be(2, "one owning lease per output view");

            NDArray y1 = outputs["Y1"], y2 = outputs["Y2"];
            y1.Storage.IsView.Should().BeTrue();
            y2.GetSingle(2).Should().Be(-2f);
            using (NDArray sum = y1 + y2)
                np.any(sum).Should().BeFalse("x + (-x) == 0, computed on the views");

            y1.Dispose();
            NDArrayOnnxInterop.LiveImports.Should().Be(1);
            y2.Dispose();
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void Run_PreallocatedOutputs_WritesInPlace_ZeroCopyBothWays()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray y1 = np.zeros(new Shape(2, 3), typeof(float));
            using NDArray y2 = np.zeros(new Shape(2, 3), typeof(float));

            session.Run(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y1", y1 }, { "Y2", y2 } });

            BytesOf(y1).Should().Equal(BytesOf(x), "ORT wrote straight into y1");
            y2.GetSingle(1, 2).Should().Be(-5f);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);

            // the same buffers again: the hot-loop pattern
            x[0, 0] = 100f;
            session.Run(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y1", y1 } });
            y1.GetSingle(0, 0).Should().Be(100f);
        }

        [TestMethod]
        public void Run_PreallocatedOutputs_ValidateDtype_Writeability_AndName()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            using NDArray wrongDtype = np.zeros(new Shape(3), typeof(double));
            new Action(() => session.Run(inputs, new Dictionary<string, NDArray> { { "Y1", wrongDtype } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y1'*Float*Double*cannot be cast*");

            using NDArray readOnly = np.broadcast_to(NDArray.Scalar(0f), new Shape(3));
            new Action(() => session.Run(inputs, new Dictionary<string, NDArray> { { "Y1", readOnly } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y1'*read-only*");

            using NDArray fine = np.zeros(new Shape(3), typeof(float));
            new Action(() => session.Run(inputs, new Dictionary<string, NDArray> { { "Nope", fine } }))
                .Should().Throw<ArgumentException>().WithMessage("*'Nope'*[Y1, Y2]*");

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void Run_Decimal_And_Char_Inputs_TakeTheConversions()
        {
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray dec = np.array(new[] { 1.5m, 2.5m });
            using NDArray y = session.Run(dec);
            y.typecode.Should().Be(NPTypeCode.Double);
            y.GetDouble(1).Should().Be(2.5);

            using InferenceSession u16 = OpenSession("identity_uint16");
            using NDArray chars = np.array("ok".ToCharArray());
            using NDArray units = u16.Run(chars);
            units.typecode.Should().Be(NPTypeCode.UInt16);
            units.GetUInt16(1).Should().Be((ushort)'k');
        }

        [TestMethod]
        public void Run_Scalar_And_Empty_Inputs()
        {
            using InferenceSession session = OpenSession("identity_int32");
            using NDArray scalar = NDArray.Scalar(42);
            using NDArray y = session.Run(scalar);
            y.ndim.Should().Be(0);
            y.GetInt32().Should().Be(42);

            using NDArray empty = np.zeros(new Shape(0, 2), typeof(int));
            using NDArray e = session.Run(empty);
            e.shape.Should().Equal(0, 2);
        }
    }
}
