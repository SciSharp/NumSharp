using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>NumSharp → ORT: <c>AsOrtValue</c> (zero-copy) and <c>ToOrtValue</c> (copy).</summary>
    [TestClass]
    public class ExportTests : OnnxTestBase
    {
        private static readonly int[][] Shapes = { new int[0], new[] { 0 }, new[] { 5 }, new[] { 2, 3 }, new[] { 2, 3, 4 }, new[] { 0, 4 }, new[] { 1 } };

        private static NDArray Make(NPTypeCode code, int[] shape)
        {
            if (shape.Length == 0)
                return NDArray.Scalar(3, code);
            return Arange(code, shape);
        }

        [TestMethod]
        public void AsOrtValue_RoundTrips_BitExact_EveryDtype_EveryShape()
        {
            foreach (NPTypeCode code in ZeroCopyDtypes)
            foreach (int[] shape in Shapes)
            {
                using NDArray nd = Make(code, shape);
                using OrtTensor t = nd.AsOrtValue();

                t.ElementType.Should().Be(NDArrayOnnxInterop.ToTensorElementType(code));
                OrtTensorTypeAndShapeInfo info = t.Value.GetTensorTypeAndShape();
                info.ElementDataType.Should().Be(t.ElementType, $"{code} {nd.Shape}");
                info.Shape.Should().Equal(Dims(shape), $"{code} {nd.Shape}");
                info.ElementCount.Should().Be(nd.size);

                using NDArray back = t.Value.ToNDArray();
                back.typecode.Should().Be(code);
                back.shape.Should().Equal(Dims(shape));
                BytesOf(back).Should().Equal(BytesOf(nd), $"{code} {nd.Shape} must round-trip byte-exact");
            }
        }

        [TestMethod]
        public unsafe void AsOrtValue_IsZeroCopy_TensorPointerIsTheNumSharpBuffer()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 4, 5);
            using OrtTensor t = nd.AsOrtValue();

            Span<byte> raw = t.Value.GetTensorMutableRawData();
            void* ortPtr = Unsafe.AsPointer(ref MemoryMarshal.GetReference(raw));
            void* ndPtr = nd.GetData().Address;
            ((IntPtr)ortPtr).Should().Be((IntPtr)ndPtr, "no copy: ORT's tensor points straight at NumSharp's memory");
            raw.Length.Should().Be(20 * sizeof(float));
        }

        [TestMethod]
        public void AsOrtValue_MutationsAreVisibleBothWays()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 6);
            using OrtTensor t = nd.AsOrtValue();

            t.Value.GetTensorMutableDataAsSpan<float>()[2] = 42f;
            nd.GetSingle(2).Should().Be(42f, "ORT wrote NumSharp's memory");

            nd[3] = 7f;
            t.Value.GetTensorDataAsSpan<float>()[3].Should().Be(7f, "NumSharp wrote the memory ORT reads");
        }

        [TestMethod]
        public void AsOrtValue_ContiguousViewAtOffset_SharesExactlyItsWindow()
        {
            using NDArray parent = Arange(NPTypeCode.Int32, 4, 3);
            NDArray[] halves = np.split(parent, 2);
            using NDArray second = halves[1];   // rows 2..3: contiguous, non-zero offset
            using (halves[0])
            using (OrtTensor t = second.AsOrtValue())
            {
                t.Value.GetTensorTypeAndShape().Shape.Should().Equal(2L, 3L);
                t.Value.GetTensorDataAsSpan<int>().ToArray().Should().Equal(6, 7, 8, 9, 10, 11);

                t.Value.GetTensorMutableDataAsSpan<int>()[0] = -1;
                parent.GetInt32(2, 0).Should().Be(-1, "the window is the parent's memory");
            }
        }

        [TestMethod]
        public void AsOrtValue_NonContiguousViews_AreRefused_WithTheMaterializeHint()
        {
            using NDArray m = Arange(NPTypeCode.Double, 4, 6);
            using NDArray transposed = m.T;
            using NDArray stepped = m["::2"];
            using NDArray reversed = m["::-1"];
            using NDArray broadcast = np.broadcast_to(m["0"], new Shape(3, 6));
            using NDArray fortran = np.asfortranarray(m);

            foreach ((NDArray view, string why) in new[] { (transposed, "transposed"), (stepped, "stepped"), (reversed, "reversed"), (broadcast, "broadcast"), (fortran, "fortran") })
            {
                view.Shape.IsContiguous.Should().BeFalse(why);
                new Action(() => view.AsOrtValue()).Should().Throw<InvalidOperationException>(why)
                    .WithMessage("*not C-contiguous*ToOrtValue*");
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a refused export pins nothing");
        }

        [TestMethod]
        public void AsOrtValue_Half_CrossesAsFloat16_BitIdentical_IncludingNaNPayloadsAndSubnormals()
        {
            ushort[] patterns = { 0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7D55, 0xFFFF, 0x03FF, 0x7BFF };
            var bytes = new byte[patterns.Length * 2];
            Buffer.BlockCopy(patterns, 0, bytes, 0, bytes.Length);
            using NDArray halves = np.frombuffer(bytes, NPTypeCode.Half);
            halves.typecode.Should().Be(NPTypeCode.Half);

            using OrtTensor t = halves.AsOrtValue();
            t.ElementType.Should().Be(TensorElementType.Float16);
            ReadOnlySpan<Float16> span = t.Value.GetTensorDataAsSpan<Float16>();
            span.Length.Should().Be(patterns.Length);
            for (int i = 0; i < patterns.Length; i++)
                span[i].value.Should().Be(patterns[i], $"pattern 0x{patterns[i]:X4} must cross untouched (no conversion)");

            using NDArray back = t.Value.ToNDArray();
            back.typecode.Should().Be(NPTypeCode.Half);
            BytesOf(back).Should().Equal(bytes);
        }

        [TestMethod]
        public void AsOrtValue_Char_ExportsUtf16CodeUnits_AsUInt16()
        {
            using NDArray chars = np.array("héllo".ToCharArray());
            chars.typecode.Should().Be(NPTypeCode.Char);

            using OrtTensor t = chars.AsOrtValue();
            t.ElementType.Should().Be(TensorElementType.UInt16);
            t.Value.GetTensorDataAsSpan<ushort>().ToArray().Should().Equal((ushort)'h', (ushort)'é', (ushort)'l', (ushort)'l', (ushort)'o');

            using NDArray back = t.Value.ToNDArray();
            back.typecode.Should().Be(NPTypeCode.UInt16, "directional: UInt16 comes back as UInt16, not Char");
        }

        [TestMethod]
        public void AsOrtValue_Decimal_ConvertsToAFloat64Temporary_TheHandleOwns()
        {
            using NDArray dec = np.array(new[] { 1.5m, -2.25m, 1e10m });
            dec.typecode.Should().Be(NPTypeCode.Decimal);

            using (OrtTensor t = dec.AsOrtValue())
            {
                t.ElementType.Should().Be(TensorElementType.Double);
                t.Source.Should().NotBeSameAs(dec, "the tensor reads the conversion temporary");
                t.Source.typecode.Should().Be(NPTypeCode.Double);
                t.Value.GetTensorDataAsSpan<double>().ToArray().Should().Equal(1.5, -2.25, 1e10);
                t.Value.GetTensorMutableDataAsSpan<double>()[0] = 9;
                dec.GetDecimal(0).Should().Be(1.5m, "mutations through ORT land in the temporary, not the Decimal source");
            }

            // the temporary is released with the handle: only the source's own buffer remains
            dec.IsDisposed.Should().BeFalse();
        }

        [TestMethod]
        public void ToOrtValue_CopiesAnyLayout_InLogicalOrder_NoLifetimeCoupling()
        {
            using NDArray m = Arange(NPTypeCode.Int32, 3, 4);
            using NDArray transposed = m.T;
            using NDArray reversed = m["::-1, ::-1"];
            using NDArray broadcast = np.broadcast_to(np.array(new[] { 1, 2, 3, 4 }), new Shape(2, 4));

            foreach (NDArray view in new[] { m, transposed, reversed, broadcast })
            {
                using OrtValue v = view.ToOrtValue();
                using NDArray expected = view.copy();
                using NDArray back = v.ToNDArray();
                back.shape.Should().Equal(view.shape);
                BytesOf(back).Should().Equal(BytesOf(expected), $"logical C-order contents of {view.Shape}");

                using OrtMemoryInfo memInfo = v.GetTensorMemoryInfo();
                memInfo.Name.Should().Be("Cpu", "ORT allocated the copy");
                NDArrayOnnxInterop.LiveExports.Should().Be(0, "a copy pins nothing");
            }
        }

        [TestMethod]
        public void ToOrtValue_IsIndependent_OfTheSource()
        {
            using NDArray nd = Arange(NPTypeCode.Double, 4);
            using OrtValue v = nd.ToOrtValue();
            nd[0] = 100.0;
            v.GetTensorDataAsSpan<double>()[0].Should().Be(0.0, "the copy does not see later writes");
            v.GetTensorMutableDataAsSpan<double>()[1] = -1;
            nd.GetDouble(1).Should().Be(1.0, "nor does the source see ORT's writes");
        }

        [TestMethod]
        public void ToOrtValue_Decimal_AndChar_ConvertLikeAsOrtValue()
        {
            using NDArray dec = np.array(new[] { 2.5m, 3m });
            using OrtValue dv = dec.ToOrtValue();
            dv.GetTensorTypeAndShape().ElementDataType.Should().Be(TensorElementType.Double);
            dv.GetTensorDataAsSpan<double>().ToArray().Should().Equal(2.5, 3.0);

            using NDArray chars = np.array("ab".ToCharArray());
            using OrtValue cv = chars.ToOrtValue();
            cv.GetTensorTypeAndShape().ElementDataType.Should().Be(TensorElementType.UInt16);
            cv.GetTensorDataAsSpan<ushort>().ToArray().Should().Equal((ushort)'a', (ushort)'b');
        }

        [TestMethod]
        public void Scalar_And_Empty_Cross_EitherWay()
        {
            using NDArray scalar = NDArray.Scalar(2.5);
            using OrtTensor ts = scalar.AsOrtValue();
            ts.Value.GetTensorTypeAndShape().Shape.Should().BeEmpty();
            ts.Value.GetTensorDataAsSpan<double>()[0].Should().Be(2.5);
            using OrtValue cs = scalar.ToOrtValue();
            cs.GetTensorTypeAndShape().Shape.Should().BeEmpty();

            using NDArray empty = np.zeros(new Shape(0, 3), typeof(int));
            using OrtTensor te = empty.AsOrtValue();
            te.Value.GetTensorTypeAndShape().Shape.Should().Equal(0L, 3L);
            te.Value.GetTensorTypeAndShape().ElementCount.Should().Be(0);
            using OrtValue ce = empty.ToOrtValue();
            ce.GetTensorTypeAndShape().Shape.Should().Equal(0L, 3L);
            using NDArray back = ce.ToNDArray();
            back.shape.Should().Equal(0, 3);
            back.typecode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void NullSource_Throws()
        {
            NDArray none = null;
            new Action(() => none.AsOrtValue()).Should().Throw<ArgumentNullException>();
            new Action(() => none.ToOrtValue()).Should().Throw<ArgumentNullException>();
        }
    }
}
