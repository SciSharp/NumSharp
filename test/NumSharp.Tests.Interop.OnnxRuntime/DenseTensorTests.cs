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
    }
}
