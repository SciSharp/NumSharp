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
    /// <summary>The legacy surface: <c>AsDenseTensor&lt;T&gt;</c> / <c>ToDenseTensor&lt;T&gt;</c> and the DenseTensor → NDArray twins.</summary>
    [TestClass]
    public class DenseTensorTests : OnnxTestBase
    {
        /// <summary>
        ///     The default mode shares a C-contiguous array through a row-major tensor: the handle reports a view, and a
        ///     write on either side is visible on the other.
        /// </summary>
        [TestMethod]
        public void AsDenseTensor_SharesTheBuffer_MutationsBothWays()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            using OrtTensor<float> h = nd.AsDenseTensor<float>();

            DenseTensor<float> t = h.Tensor;
            t.Dimensions.ToArray().Should().Equal(2, 3);
            t.IsReversedStride.Should().BeFalse();
            t[1, 2].Should().Be(5f);
            h.IsView.Should().BeTrue("a C-contiguous array of a bit-exact dtype is shared under the default mode");
            h.Source.Should().BeSameAs(nd);

            t[0, 1] = 42f;
            nd.GetSingle(0, 1).Should().Be(42f, "the tensor writes NumSharp's memory");
            nd[1, 0] = -3f;
            t[1, 0].Should().Be(-3f, "NumSharp writes the memory the tensor reads");
            h.Memory.Length.Should().Be(6);
        }

        /// <summary>
        ///     A Fortran-ordered array crosses as a ROW-major tensor over a C-order copy. The column-major share this verb
        ///     used to make (<c>reverseStride: true</c>) could not be fed to a session: a run that got it through
        ///     <c>NamedOnnxValue</c> threw, and an <c>OrtValue</c> built over its <c>Memory</c> read the elements in memory
        ///     order (0, 4, 8, 1, … for this array). Both ORT routes now read the array's logical elements, and the strict
        ///     mode refuses instead of copying.
        /// </summary>
        [TestMethod]
        public void AsDenseTensor_FortranOrder_IsARowMajorCopy_ThatBothOrtRoutesReadCorrectly()
        {
            using NDArray c = Arange(NPTypeCode.Single, 3, 4);
            using NDArray f = np.asfortranarray(c);
            f.Shape.IsFContiguous.Should().BeTrue();
            f.Shape.IsContiguous.Should().BeFalse();

            new Action(() => f.AsDenseTensor<float>(OnnxCodecMode.View)).Should().Throw<InvalidOperationException>()
                .WithMessage("*OnnxCodecMode.View*not C-contiguous*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a refused export pins nothing");

            using OrtTensor<float> h = f.AsDenseTensor<float>();
            h.IsView.Should().BeFalse("ORT cannot read column-major memory in place");
            h.Source.Should().NotBeSameAs(f);
            h.Tensor.IsReversedStride.Should().BeFalse("the tensor is always row-major");
            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                h.Tensor[i, j].Should().Be(c.GetSingle(i, j), $"[{i},{j}]");

            // Route 1: the legacy NamedOnnxValue run, which refused the column-major tensor outright.
            using InferenceSession session = OpenSession("identity_float32");
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("X", h.Tensor) }))
            using (NDArray back = results[0].ToNDArray())
            {
                back.shape.Should().Equal(3, 4);
                BytesOf(back).Should().Equal(BytesOf(c), "the model received the array's logical elements");
            }

            // Route 2: an OrtValue over the tensor's Memory, which read the column-major buffer as if it were row-major.
            using (OrtValue overMemory = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { 3, 4 }))
                overMemory.GetTensorDataAsSpan<float>().ToArray().Should().Equal(c.ToArray<float>());

            h.Tensor[2, 1] = 99;
            f.GetSingle(2, 1).Should().Be(c.GetSingle(2, 1), "a write through the tensor lands in the copy, not in the source");
            h.Source.GetSingle(2, 1).Should().Be(99f, "the write is readable through Source");
        }

        [TestMethod]
        public void AsDenseTensor_WrongElementType_IsRefused_NotReinterpreted()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 4);
            new Action(() => nd.AsDenseTensor<int>()).Should().Throw<ArgumentException>().WithMessage("*DenseTensor<Single>*");
            new Action(() => nd.ToDenseTensor<double>()).Should().Throw<ArgumentException>().WithMessage("*DenseTensor<Single>*");

            using NDArray half = Arange(NPTypeCode.Half, 4);
            new Action(() => half.AsDenseTensor<Half>()).Should().Throw<ArgumentException>()
                .WithMessage("*Float16*", "ORT's DenseTensor spells float16 as Microsoft.ML.OnnxRuntime.Float16");
            using OrtTensor<Float16> ok = half.AsDenseTensor<Float16>();
            ((float)ok.Tensor[3]).Should().Be(3f);

            using NDArray chars = np.array("xy".ToCharArray());
            using OrtTensor<ushort> asUnits = chars.AsDenseTensor<ushort>();
            asUnits.Tensor[1].Should().Be((ushort)'y');
        }

        /// <summary>
        ///     A stepped view: the strict mode refuses it; the default mode gives a row-major tensor over a C-order copy
        ///     with the view's elements; <see cref="OnnxCodecMode.Copy"/> copies even the contiguous base.
        /// </summary>
        [TestMethod]
        public void AsDenseTensor_StridedView_IsCopiedUnderAuto_AndRefusedUnderView()
        {
            using NDArray m = Arange(NPTypeCode.Double, 4, 4);
            using NDArray stepped = m["::2"];   // rows 0 and 2
            new Action(() => stepped.AsDenseTensor<double>(OnnxCodecMode.View)).Should().Throw<InvalidOperationException>()
                .WithMessage("*OnnxCodecMode.View*not C-contiguous*OnnxCodecMode.Auto*OnnxCodecMode.Copy*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            using (OrtTensor<double> h = stepped.AsDenseTensor<double>())
            {
                h.IsView.Should().BeFalse();
                h.Tensor.Dimensions.ToArray().Should().Equal(2, 4);
                h.Tensor.Buffer.ToArray().Should().Equal(0.0, 1, 2, 3, 8, 9, 10, 11);
                h.Tensor[1, 0] = -1;
                m.GetDouble(2, 0).Should().Be(8.0, "the tensor is a copy");
            }

            using (OrtTensor<double> h = m.AsDenseTensor<double>(OnnxCodecMode.Copy))
            {
                h.IsView.Should().BeFalse("Copy copies even an array ORT could share");
                h.Tensor[0, 0] = -1;
                m.GetDouble(0, 0).Should().Be(0.0);
            }

            // The reference state of the source is checked in ArcAccountingTests.CopyHandles_…: here the stepped view is
            // still alive and holds its own reference, so "uniquely referenced" cannot be asserted.
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ToDenseTensor_IsAnIndependentManagedCopy_InLogicalOrder()
        {
            using NDArray m = Arange(NPTypeCode.Int64, 2, 3);
            using NDArray t = m.T;   // (3,2), non-contiguous
            DenseTensor<long> copy = t.ToDenseTensor<long>();
            copy.Dimensions.ToArray().Should().Equal(3, 2);
            copy.Buffer.ToArray().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            copy[0, 0] = 77;
            m.GetInt64(0, 0).Should().Be(0L, "a copy shares nothing");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a copy pins nothing");
        }

        /// <summary>
        ///     Decimal crosses as float64 through both dense-tensor verbs; the handle verb's result is a copy, and its
        ///     strict mode refuses the conversion.
        /// </summary>
        [TestMethod]
        public void Decimal_ConvertsToDouble_ForBothDenseTensorVerbs()
        {
            using NDArray dec = np.array(new[] { 1.25m, 2m });
            using OrtTensor<double> h = dec.AsDenseTensor<double>();
            h.Tensor.Buffer.ToArray().Should().Equal(1.25, 2.0);
            h.Source.typecode.Should().Be(NPTypeCode.Double);
            h.IsView.Should().BeFalse("a conversion is a copy");
            dec.ToDenseTensor<double>().Buffer.ToArray().Should().Equal(1.25, 2.0);

            new Action(() => dec.AsDenseTensor<double>(OnnxCodecMode.View)).Should().Throw<InvalidOperationException>()
                .WithMessage("*OnnxCodecMode.View*Decimal*");
        }

        [TestMethod]
        public void DenseTensor_ToNDArray_And_AsNDArray_RowMajor()
        {
            var tensor = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

            using NDArray copy = tensor.ToNDArray();
            copy.typecode.Should().Be(NPTypeCode.Single);
            copy.shape.Should().Equal(2, 3);
            copy.GetSingle(1, 2).Should().Be(6f);
            copy[0, 0] = 100f;
            tensor[0, 0].Should().Be(1f, "ToNDArray copies");

            using (NDArray view = tensor.AsNDArray())
            {
                view.shape.Should().Equal(2, 3);
                view.GetSingle(1, 0).Should().Be(4f);
                view[1, 0] = -4f;
                tensor[1, 0].Should().Be(-4f, "AsNDArray shares the tensor's (pinned) buffer");
                NDArrayOnnxInterop.LiveImports.Should().Be(1);
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "disposing the view released the pin");
        }

        [TestMethod]
        public void DenseTensor_ColumnMajor_ToNDArray_TransposesBack_AsNDArray_IsAFortranView()
        {
            // memory 1..6 read column-major as (2,3): [[1,3,5],[2,4,6]]
            var tensor = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 }, reverseStride: true);
            tensor[0, 1].Should().Be(3);

            using NDArray copy = tensor.ToNDArray();
            copy.Shape.IsContiguous.Should().BeTrue("ToNDArray always yields a C-contiguous owning copy");
            copy.GetInt32(0, 1).Should().Be(3);
            copy.GetInt32(1, 2).Should().Be(6);

            using NDArray view = tensor.AsNDArray();
            view.Shape.IsFContiguous.Should().BeTrue("the view keeps the tensor's column-major layout — no copy");
            view.GetInt32(0, 1).Should().Be(3);
            view.GetInt32(1, 0).Should().Be(2);
            view[1, 2] = 60;
            tensor[1, 2].Should().Be(60);
        }

        [TestMethod]
        public void Float16Tensor_ReadsBackAsHalf()
        {
            var tensor = new DenseTensor<Float16>(new[] { (Float16)1.5f, (Float16)(-2f) }, new[] { 2 });
            using NDArray nd = tensor.ToNDArray();
            nd.typecode.Should().Be(NPTypeCode.Half);
            ((float)nd.GetHalf(0)).Should().Be(1.5f);
            ((float)nd.GetHalf(1)).Should().Be(-2f);
        }

        [TestMethod]
        public void AsDenseTensor_FeedsTheLegacyNamedOnnxValueRun_AndTheResultReadsBack()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            using OrtTensor<float> h = nd.AsDenseTensor<float>();

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("X", h.Tensor) };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
            results.Count.Should().Be(1);
            results[0].Name.Should().Be("Y");

            using NDArray back = results[0].ToNDArray();   // runtime dispatch on the tensor's element type
            back.typecode.Should().Be(NPTypeCode.Single);
            back.shape.Should().Equal(2, 3);
            BytesOf(back).Should().Equal(BytesOf(nd));
        }

        /// <summary>
        ///     The runtime dispatch of <c>NamedOnnxValue.ToNDArray</c> takes every element type the generic
        ///     <c>DenseTensor&lt;T&gt;.ToNDArray</c> takes. A <see cref="char"/> tensor and a <see cref="Half"/> tensor (a
        ///     caller's own; ORT's outputs use <see cref="Float16"/>) were refused here with "no NumSharp dtype".
        /// </summary>
        [TestMethod]
        public void NamedOnnxValue_WithACharOrHalfTensor_ReadsBack_LikeTheGenericVerb()
        {
            var chars = new DenseTensor<char>(new[] { 'h', 'é', 'y' }, new[] { 3 });
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("c", chars).ToNDArray())
            {
                nd.typecode.Should().Be(NPTypeCode.Char);
                nd.ToArray<char>().Should().Equal('h', 'é', 'y');
            }

            var halves = new DenseTensor<Half>(new[] { (Half)1.5f, (Half)(-2f), Half.NaN }, new[] { 3 });
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("h", halves).ToNDArray())
            {
                nd.typecode.Should().Be(NPTypeCode.Half);
                ((float)nd.GetHalf(0)).Should().Be(1.5f);
                ((float)nd.GetHalf(1)).Should().Be(-2f);
                Half.IsNaN(nd.GetHalf(2)).Should().BeTrue();
            }

            // column-major: the dispatch goes through the same transposing copy as the generic verb
            var columnMajor = new DenseTensor<char>(new[] { 'a', 'b', 'c', 'd', 'e', 'f' }, new[] { 2, 3 }, reverseStride: true);
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("f", columnMajor).ToNDArray())
                nd.ToArray<char>().Should().Equal('a', 'c', 'e', 'b', 'd', 'f');
        }

        /// <summary>
        ///     A tensor that is NOT a <see cref="DenseTensor{T}"/> (a caller's own <see cref="Tensor{T}"/> subclass) inside
        ///     a <see cref="NamedOnnxValue"/>: the dispatch densifies it through ORT's own <c>ToDenseTensor</c> and copies
        ///     that, for the element types an ORT output has and for the two only a caller's tensor has (char, Half).
        /// </summary>
        [TestMethod]
        public void NamedOnnxValue_WithANonDenseTensor_IsDensifiedAndRead()
        {
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("f", new ArrayTensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 })).ToNDArray())
            {
                nd.typecode.Should().Be(NPTypeCode.Single);
                nd.shape.Should().Equal(2, 3);
                nd.ToArray<float>().Should().Equal(1f, 2f, 3f, 4f, 5f, 6f);
            }
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("l", new ArrayTensor<long>(new[] { -1L, long.MaxValue }, new[] { 2 })).ToNDArray())
                nd.ToArray<long>().Should().Equal(-1L, long.MaxValue);
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("c", new ArrayTensor<char>(new[] { 'o', 'k' }, new[] { 2 })).ToNDArray())
            {
                nd.typecode.Should().Be(NPTypeCode.Char);
                nd.ToArray<char>().Should().Equal('o', 'k');
            }
            using (NDArray nd = NamedOnnxValue.CreateFromTensor("h", new ArrayTensor<Half>(new[] { (Half)0.5f, (Half)(-4f) }, new[] { 2 })).ToNDArray())
            {
                nd.typecode.Should().Be(NPTypeCode.Half);
                ((float)nd.GetHalf(1)).Should().Be(-4f);
            }
        }

        /// <summary>
        ///     A minimal tensor that is not dense as far as its type says: a row-major <see cref="Tensor{T}"/> subclass over
        ///     an array, the shape of "a caller's own tensor class". Only what ORT's <c>ToDenseTensor</c> and the test need
        ///     is given real behaviour.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        private sealed class ArrayTensor<T> : Tensor<T>
        {
            private readonly T[] _values;

            /// <summary>Wraps <paramref name="values"/> (row-major) as a tensor of the given dimensions.</summary>
            /// <param name="values">The elements; not copied.</param>
            /// <param name="dimensions">The dimensions, whose product is the element count.</param>
            internal ArrayTensor(T[] values, int[] dimensions) : base(dimensions, reverseStride: false) => _values = values;

            /// <summary>A tensor over a copy of the elements.</summary>
            /// <returns>The copy.</returns>
            public override Tensor<T> Clone() => new ArrayTensor<T>((T[])_values.Clone(), Dimensions.ToArray());

            /// <summary>A zero-filled tensor of another element type and the given dimensions.</summary>
            /// <typeparam name="TResult">The new element type.</typeparam>
            /// <param name="dimensions">The new dimensions.</param>
            /// <returns>The empty tensor.</returns>
            public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
            {
                int length = 1;
                foreach (int d in dimensions)
                    length *= d;
                return new ArrayTensor<TResult>(new TResult[length], dimensions.ToArray());
            }

            /// <summary>The element at a flat row-major index.</summary>
            /// <param name="index">The flat index.</param>
            /// <returns>The element.</returns>
            public override T GetValue(int index) => _values[index];

            /// <summary>Stores an element at a flat row-major index.</summary>
            /// <param name="index">The flat index.</param>
            /// <param name="value">The element.</param>
            public override void SetValue(int index, T value) => _values[index] = value;

            /// <summary>The same elements under other dimensions.</summary>
            /// <param name="dimensions">The new dimensions (same element count).</param>
            /// <returns>A tensor sharing this one's array.</returns>
            public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions) => new ArrayTensor<T>(_values, dimensions.ToArray());
        }

        [TestMethod]
        public void NamedOnnxValue_NonTensor_IsRefused()
        {
            var seq = NamedOnnxValue.CreateFromSequence("s", new List<float> { 1f });
            new Action(() => seq.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*not a tensor*");
        }

        /// <summary>
        ///     The runtime dispatch of <c>NamedOnnxValue.ToNDArray</c> has one branch per element type for a
        ///     <see cref="DenseTensor{T}"/> and one for any other <see cref="Tensor{T}"/>. Every one of the 14 element
        ///     types is read through both, and comes back with its dtype, its shape and its bytes. Only 4 of the 28
        ///     branches had a test.
        /// </summary>
        [TestMethod]
        public void NamedOnnxValue_EveryElementType_DenseAndNot_ReadsBackBitForBit()
        {
            NamedSweep(new float[] { 1.5f, -2f, float.NaN, 0f }, NPTypeCode.Single);
            NamedSweep(new double[] { 1.5, -2, double.NegativeInfinity, 0 }, NPTypeCode.Double);
            NamedSweep(new sbyte[] { -128, -1, 0, 127 }, NPTypeCode.SByte);
            NamedSweep(new byte[] { 0, 1, 254, 255 }, NPTypeCode.Byte);
            NamedSweep(new short[] { short.MinValue, -1, 0, short.MaxValue }, NPTypeCode.Int16);
            NamedSweep(new ushort[] { 0, 1, 65534, ushort.MaxValue }, NPTypeCode.UInt16);
            NamedSweep(new int[] { int.MinValue, -1, 0, int.MaxValue }, NPTypeCode.Int32);
            NamedSweep(new uint[] { 0, 1, uint.MaxValue - 1, uint.MaxValue }, NPTypeCode.UInt32);
            NamedSweep(new long[] { long.MinValue, -1, 0, long.MaxValue }, NPTypeCode.Int64);
            NamedSweep(new ulong[] { 0, 1, ulong.MaxValue - 1, ulong.MaxValue }, NPTypeCode.UInt64);
            NamedSweep(new bool[] { true, false, false, true }, NPTypeCode.Boolean);
            NamedSweep(new Float16[] { new Float16(0x3E00), new Float16(0xC000), new Float16(0x7E00), new Float16(0) }, NPTypeCode.Half);
            NamedSweep(new Half[] { (Half)1.5f, (Half)(-2f), Half.NaN, (Half)0f }, NPTypeCode.Half);
            NamedSweep(new char[] { 'a', 'é', '日', '\0' }, NPTypeCode.Char);
        }

        /// <summary>Reads one <c>[2, 2]</c> tensor of <typeparamref name="T"/> through the dispatch, as a dense tensor and as a tensor that is not dense.</summary>
        /// <typeparam name="T">The tensor's element type.</typeparam>
        /// <param name="values">Four elements, row-major.</param>
        /// <param name="expected">The NumSharp dtype the dispatch must return.</param>
        private static void NamedSweep<T>(T[] values, NPTypeCode expected) where T : unmanaged
        {
            byte[] want = Oracle.Bytes(values);
            foreach (bool dense in new[] { true, false })
            {
                Tensor<T> tensor = dense
                    ? new DenseTensor<T>((T[])values.Clone(), new[] { 2, 2 })
                    : new ArrayTensor<T>((T[])values.Clone(), new[] { 2, 2 });
                string cell = $"{typeof(T).Name}, {(dense ? "DenseTensor" : "another Tensor<T>")}";
                using NDArray nd = NamedOnnxValue.CreateFromTensor("t", tensor).ToNDArray();
                nd.typecode.Should().Be(expected, cell);
                nd.shape.Should().Equal(new long[] { 2, 2 }, cell);
                nd.flags.owndata.Should().BeTrue(cell);
                BytesOfArray(nd).Should().Equal(want, cell);
            }
        }

        /// <summary>The bytes of a C-contiguous owning array, read from its buffer.</summary>
        /// <param name="nd">The array.</param>
        /// <returns>A copy of its bytes.</returns>
        private static unsafe byte[] BytesOfArray(NDArray nd) =>
            new ReadOnlySpan<byte>((byte*)nd.Storage.InternalArray.Address, checked((int)(nd.size * nd.dtypesize))).ToArray();

        /// <summary>
        ///     What the dispatch refuses, each with a message that says why: a null named value, a named tensor that
        ///     holds no tensor, a BFloat16 tensor, and a tensor of an element type NumSharp has no dtype for.
        /// </summary>
        [TestMethod]
        public void NamedOnnxValue_NullEmptyBFloat16AndUnknownElementType_AreRefused()
        {
            NamedOnnxValue none = null;
            new Action(() => none.ToNDArray()).Should().Throw<ArgumentNullException>();

            NamedOnnxValue empty = NamedOnnxValue.CreateFromTensor<float>("empty", null);
            new Action(() => empty.ToNDArray()).Should().Throw<InvalidOperationException>().WithMessage("NamedOnnxValue 'empty' holds no value.");

            // The BFloat16 refusal is the one that says what to do, not the generic "holds a <type>" one (whose text
            // would name BFloat16 too, through the tensor's type name).
            NamedOnnxValue bfloat = NamedOnnxValue.CreateFromTensor("b", new DenseTensor<BFloat16>(new[] { new BFloat16(0x3FC0) }, new[] { 1 }));
            new Action(() => bfloat.ToNDArray()).Should().Throw<NotSupportedException>()
                .WithMessage("ORT BFloat16 has no NumSharp dtype*Cast the tensor to float32 or float16*");

            NamedOnnxValue guid = NamedOnnxValue.CreateFromTensor("g", new DenseTensor<Guid>(new[] { Guid.Empty }, new[] { 1 }));
            new Action(() => guid.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("NamedOnnxValue 'g' holds a *which has no NumSharp dtype.");
        }

        /// <summary>The four generic verbs of the legacy surface refuse a null argument by name.</summary>
        [TestMethod]
        public void DenseVerbs_NullArguments_Throw()
        {
            NDArray noArray = null;
            DenseTensor<float> noTensor = null;
            new Action(() => noArray.AsDenseTensor<float>()).Should().Throw<ArgumentNullException>().WithParameterName("source");
            new Action(() => noArray.ToDenseTensor<float>()).Should().Throw<ArgumentNullException>().WithParameterName("source");
            new Action(() => noTensor.ToNDArray()).Should().Throw<ArgumentNullException>().WithParameterName("tensor");
            new Action(() => noTensor.AsNDArray()).Should().Throw<ArgumentNullException>().WithParameterName("tensor");
        }
    }
}
