using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     <c>ndarray.flags</c> across every conversion path — the memory-layout contract each verb promises:
    ///     <c>ToNDArray</c> yields an OWNING C-contiguous array (<c>owndata=true</c>), <c>AsNDArray</c> a VIEW
    ///     over foreign memory (<c>owndata=false</c>, like <c>np.frombuffer</c>), a column-major DenseTensor
    ///     comes back an F-contiguous view, and the export/pre-allocated-output verbs enforce
    ///     C_CONTIGUOUS / WRITEABLE. Every <c>num</c> below is the exact NumPy flag integer (probed).
    /// </summary>
    [TestClass]
    public class FlagsTests : OnnxTestBase
    {
        // NumPy flag bits: C=1, F=2, OWNDATA=4, ALIGNED=256, WRITEABLE=1024.
        private const int C = 1, F = 2, OWN = 4, ALIGNED = 256, W = 1024;

        [TestMethod]
        public void ToNDArray_Copy_IsOwning_CContiguous_Writeable()
        {
            using NDArray src2d = Arange(NPTypeCode.Single, 2, 3);
            using OrtValue v = src2d.ToOrtValue();
            using NDArray a = v.ToNDArray();
            a.flags.c_contiguous.Should().BeTrue();
            a.flags.f_contiguous.Should().BeFalse("a 2-D C array is not also F-contiguous");
            a.flags.owndata.Should().BeTrue("ToNDArray allocates a fresh owning buffer");
            a.flags.writeable.Should().BeTrue();
            a.flags.aligned.Should().BeTrue();
            a.flags.num.Should().Be(C | OWN | ALIGNED | W).And.Be(1285);
        }

        [TestMethod]
        public void AsNDArray_View_IsNonOwning_CContiguous_Writeable()
        {
            using NDArray src2d = Arange(NPTypeCode.Single, 2, 3);
            using OrtValue v = src2d.ToOrtValue();
            using NDArray a = v.AsNDArray();
            a.flags.c_contiguous.Should().BeTrue();
            a.flags.owndata.Should().BeFalse("AsNDArray views ORT's own buffer (owndata==False, like np.frombuffer)");
            a.flags.writeable.Should().BeTrue("the view shares mutable ORT memory");
            a.flags.aligned.Should().BeTrue();
            a.flags.num.Should().Be(C | ALIGNED | W).And.Be(1281, "no OWNDATA bit vs the copy's 1285");
        }

        [TestMethod]
        public void OneDimensionalAndScalar_AreBothCAndFContiguous()
        {
            foreach (int[] shape in new[] { new[] { 4 }, new int[0] })
            {
                using NDArray src = shape.Length == 0 ? NDArray.Scalar(2.5f) : Arange(NPTypeCode.Single, shape);
                using OrtValue v = src.ToOrtValue();
                using NDArray copy = v.ToNDArray();
                copy.flags.c_contiguous.Should().BeTrue();
                copy.flags.f_contiguous.Should().BeTrue("1-D / 0-d is both C- and F-contiguous");
                copy.flags.num.Should().Be(C | F | OWN | ALIGNED | W).And.Be(1287);

                using NDArray view = v.AsNDArray();
                view.flags.owndata.Should().BeFalse();
                view.flags.num.Should().Be(C | F | ALIGNED | W).And.Be(1283);
            }
        }

        [TestMethod]
        public void AsNDArray_OfAnEmptyTensor_IsOwning_NotAView_NothingToShare()
        {
            using NDArray empty = np.zeros(new Shape(0, 3), typeof(float));
            using OrtValue v = empty.ToOrtValue();
            using NDArray view = v.AsNDArray();
            // an empty tensor has nothing to lease, so AsNDArray returns a fresh OWNING empty array,
            // not a foreign view — owndata is True here (unlike the non-empty view above).
            view.flags.owndata.Should().BeTrue("an empty AsNDArray owns its (zero-size) buffer");
            view.flags.c_contiguous.Should().BeTrue();
            view.flags.f_contiguous.Should().BeTrue("empty arrays are both C- and F-contiguous");
            view.flags.num.Should().Be(1287);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "no lease for an empty view");
        }

        [TestMethod]
        public void DenseTensor_ColumnMajor_AsNDArray_IsAnFContiguousView_ToNDArrayIsACContiguousCopy()
        {
            var cm = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 }, reverseStride: true);

            using NDArray view = cm.AsNDArray();
            view.flags.c_contiguous.Should().BeFalse();
            view.flags.f_contiguous.Should().BeTrue("column-major memory is an F-contiguous view — no transpose copy");
            view.flags.owndata.Should().BeFalse();
            view.flags.num.Should().Be(F | ALIGNED | W).And.Be(1282);

            using NDArray copy = cm.ToNDArray();
            copy.flags.c_contiguous.Should().BeTrue("ToNDArray always yields a C-contiguous owning copy");
            copy.flags.f_contiguous.Should().BeFalse();
            copy.flags.owndata.Should().BeTrue();
            copy.flags.num.Should().Be(1285);
        }

        [TestMethod]
        public void DenseTensor_RowMajor_AsNDArray_IsANonOwningCContiguousView()
        {
            var rm = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
            using NDArray view = rm.AsNDArray();
            view.flags.c_contiguous.Should().BeTrue();
            view.flags.owndata.Should().BeFalse("a view over the tensor's pinned managed buffer");
            view.flags.num.Should().Be(1281);
        }

        [TestMethod]
        public void ReadOnlyCContiguousArray_CanStillBeExportedAsAnInput()
        {
            using NDArray ro = Arange(NPTypeCode.Single, 2, 3);
            ro.setflags(write: false);
            ro.flags.writeable.Should().BeFalse();
            ro.flags.c_contiguous.Should().BeTrue();
            ro.flags.num.Should().Be(C | OWN | ALIGNED).And.Be(261, "read-only: no WRITEABLE bit");

            // AsOrtValue shares the buffer for ORT to READ — a read-only input is fine (ORT does not write inputs).
            using (OrtTensor t = ro.AsOrtValue())
                t.Value.GetTensorDataAsSpan<float>()[5].Should().Be(5f);

            ro.flags.writeable.Should().BeFalse("exporting does not change the source's flags");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ReadOnlyCContiguousArray_IsRefusedAsAPreAllocatedOutput_WriteableCheckIsolated()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray y = np.zeros(new Shape(2, 3), typeof(float));
            y.setflags(write: false);   // C-contiguous but read-only — isolates the WRITEABLE check from contiguity
            y.flags.c_contiguous.Should().BeTrue();
            y.flags.writeable.Should().BeFalse();

            new Action(() => session.Run(new Dictionary<string, NDArray> { { "X", x } },
                    new Dictionary<string, NDArray> { { "Y1", y } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y1'*read-only*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void FContiguousInput_IsRefusedByAsOrtValue()
        {
            using NDArray m = Arange(NPTypeCode.Single, 2, 3);
            using NDArray f = np.asfortranarray(m);
            f.flags.f_contiguous.Should().BeTrue();
            f.flags.c_contiguous.Should().BeFalse();
            new Action(() => f.AsOrtValue()).Should().Throw<InvalidOperationException>().WithMessage("*not C-contiguous*");
        }

        [TestMethod]
        public void RoundTripThroughASession_PreservesTheFlagContract()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray nd = Arange(NPTypeCode.Single, 3, 4);
            using OrtTensor input = nd.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs =
                session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "Y" });

            using (NDArray copy = outputs[0].ToNDArray())
            {
                copy.flags.owndata.Should().BeTrue("a real session output copied out owns its buffer");
                copy.flags.c_contiguous.Should().BeTrue();
                copy.flags.num.Should().Be(1285);
            }
            using (NDArray view = outputs[0].AsNDArray())
            {
                view.flags.owndata.Should().BeFalse("the zero-copy view of a real output does not own ORT's buffer");
                view.flags.num.Should().Be(1281);
            }
        }

        [TestMethod]
        public void SetflagsOnAView_MakesItReadOnly_WithoutClaimingOwnership()
        {
            using NDArray src = Arange(NPTypeCode.Double, 5);
            using OrtValue v = src.ToOrtValue();
            using NDArray view = v.AsNDArray();
            view.flags.writeable.Should().BeTrue();
            view.flags.owndata.Should().BeFalse();

            view.setflags(write: false);
            view.flags.writeable.Should().BeFalse("setflags(write:false) makes the view read-only in place");
            view.flags.owndata.Should().BeFalse("a read-only view still does not own ORT's buffer");
            view.GetDouble(4).Should().Be(4.0, "reads still work through the read-only view");
        }

        [TestMethod]
        public void FlagsToString_IsTheSixLineNumpyRepr()
        {
            using NDArray src = Arange(NPTypeCode.Single, 2, 3);
            using OrtValue v = src.ToOrtValue();
            using NDArray view = v.AsNDArray();
            view.flags.ToString().Should().Be(
                "  C_CONTIGUOUS : True\n" +
                "  F_CONTIGUOUS : False\n" +
                "  OWNDATA : False\n" +
                "  WRITEABLE : True\n" +
                "  ALIGNED : True\n" +
                "  WRITEBACKIFCOPY : False\n");
        }
    }
}
