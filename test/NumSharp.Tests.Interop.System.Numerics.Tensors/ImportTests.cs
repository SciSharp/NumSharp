using System;
using System.Buffers;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>System.Numerics.Tensors → NumSharp: <c>ToNDArray</c> (copy) and <c>AsNDArray</c> (zero-copy view).</summary>
    [TestClass]
    public class ImportTests : TensorsTestBase
    {
        [TestMethod]
        public void ToNDArray_DenseTensor_CopiesValuesAndShape()
        {
            var t = Tensor.Create(new float[] { 1, 2, 3, 4, 5, 6 }, new nint[] { 2, 3 });
            using var nd = t.ToNDArray();

            nd.Shape.NDim.Should().Be(2);
            nd.shape[0].Should().Be(2);
            nd.shape[1].Should().Be(3);
            nd.typecode.Should().Be(NPTypeCode.Single);
            nd.GetSingle(0, 0).Should().Be(1f);
            nd.GetSingle(1, 2).Should().Be(6f);
        }

        [TestMethod]
        public void ToNDArray_StridedTensor_ReadInLogicalOrder()
        {
            // transposed 3x3 via strides [1,3]: t[i,j] = arr[i + 3j]
            var t = Tensor.Create(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, new nint[] { 3, 3 }, new nint[] { 1, 3 });
            t.IsDense.Should().BeFalse();
            using var nd = t.ToNDArray();

            nd.GetInt32(0, 1).Should().Be(3);
            nd.GetInt32(2, 0).Should().Be(2);
            nd.GetInt32(2, 2).Should().Be(8);
            nd.Shape.IsContiguous.Should().BeTrue("the copy is a fresh C-contiguous array");
        }

        [TestMethod]
        public void ToNDArray_Copy_IsIndependent()
        {
            var t = Tensor.Create(new double[] { 1, 2, 3 }, new nint[] { 3 });
            using var nd = t.ToNDArray();
            t[new nint[] { 0 }] = 999.0;                     // mutate the tensor after the copy
            nd.GetDouble(0).Should().Be(1.0, "ToNDArray is an owning copy");
        }

        [TestMethod]
        public void AsNDArray_DenseTensor_SharesMemory_WriteThrough()
        {
            var t = Tensor.Create(new double[] { 0, 0, 0, 0 }, new nint[] { 2, 2 });
            using (var nd = t.AsNDArray())
            {
                nd.GetDouble(0, 0).Should().Be(0.0);
                nd.SetDouble(42.0, 0, 1);                    // ndarray -> tensor backing
                ((double)t[new nint[] { 0, 1 }]).Should().Be(42.0);

                t[new nint[] { 1, 0 }] = -3.0;               // tensor -> ndarray
                nd.GetDouble(1, 0).Should().Be(-3.0);
            }
        }

        [TestMethod]
        public void AsNDArray_StridedTensor_SharesMemory_NoDensifyingCopy()
        {
            var t = Tensor.Create(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, new nint[] { 3, 3 }, new nint[] { 1, 3 });
            using var nd = t.AsNDArray();

            nd.shape[0].Should().Be(3);
            nd.shape[1].Should().Be(3);
            nd.GetInt32(0, 1).Should().Be(3);
            nd.GetInt32(2, 0).Should().Be(2);

            nd.SetInt32(555, 0, 1);                          // writes through the shared strided memory
            ((int)t[new nint[] { 0, 1 }]).Should().Be(555);
        }

        [TestMethod]
        public void AsNDArray_Empty_IsEmptyArray()
        {
            var t = Tensor.Create(Array.Empty<float>(), new nint[] { 0 });
            using var nd = t.AsNDArray();
            nd.size.Should().Be(0);
        }

        [TestMethod]
        public void ToNDArray_FromReadOnlyTensorSpan_Copies()
        {
            var t = Tensor.Create(new float[] { 10, 20, 30, 40 }, new nint[] { 2, 2 });
            ReadOnlyTensorSpan<float> ro = t.AsReadOnlyTensorSpan();
            using NDArray nd = ro.ToNDArray();

            nd.shape[0].Should().Be(2);
            nd.GetSingle(1, 1).Should().Be(40f);
        }

        [TestMethod]
        public void ToNDArray_FromTensorSpan_Copies()
        {
            var t = Tensor.Create(new int[] { 7, 8, 9 }, new nint[] { 3 });
            TensorSpan<int> s = t.AsTensorSpan();
            using NDArray nd = s.ToNDArray();
            nd.GetInt32(2).Should().Be(9);
        }

        [TestMethod]
        public void ToNDArray_And_AsNDArray_Null_Throw()
        {
            new Action(() => NDArrayTensorsInterop.ToNDArray<float>((Tensor<float>)null)).Should().Throw<ArgumentNullException>();
            new Action(() => NDArrayTensorsInterop.AsNDArray<float>((Tensor<float>)null)).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void ToNDArray_AllDtypes_RoundTripBytes()
        {
            AssertImportBytes<bool>(NPTypeCode.Boolean, new bool[] { true, false, true, true, false, true });
            AssertImportBytes<byte>(NPTypeCode.Byte, new byte[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<sbyte>(NPTypeCode.SByte, new sbyte[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<short>(NPTypeCode.Int16, new short[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<ushort>(NPTypeCode.UInt16, new ushort[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<int>(NPTypeCode.Int32, new int[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<uint>(NPTypeCode.UInt32, new uint[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<long>(NPTypeCode.Int64, new long[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<ulong>(NPTypeCode.UInt64, new ulong[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<char>(NPTypeCode.Char, new char[] { 'a', 'b', 'c', 'd', 'e', 'f' });
            AssertImportBytes<Half>(NPTypeCode.Half, new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5, (Half)6 });
            AssertImportBytes<float>(NPTypeCode.Single, new float[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<double>(NPTypeCode.Double, new double[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<decimal>(NPTypeCode.Decimal, new decimal[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<Complex>(NPTypeCode.Complex, new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8), new Complex(9, 10), new Complex(11, 12) });
        }

        private static void AssertImportBytes<T>(NPTypeCode code, T[] data) where T : unmanaged
        {
            var t = Tensor.Create(data, new nint[] { 2, 3 });
            using NDArray nd = t.ToNDArray();
            nd.typecode.Should().Be(code);
            nd.shape[0].Should().Be(2);
            nd.shape[1].Should().Be(3);
            BytesOf(nd).Should().Equal(MemoryMarshal.AsBytes<T>(data).ToArray(), $"{code} imports byte-exactly");
        }

        // ---- tensors that start past element 0 of their backing array ----------------------------------

        /// <summary>
        ///     A tensor that starts past element 0 of its backing array (a <c>Slice</c> / range indexer, or
        ///     <c>Tensor.Create(array, start, …)</c>) copies ITS OWN elements. Regression: the dense copy read from
        ///     <c>Tensor&lt;T&gt;.GetPinnedHandle()</c>'s pointer, which is the backing array's element 0, not the
        ///     tensor's start, so <c>t[1..3, ..]</c> came back as rows 0..1.
        /// </summary>
        [TestMethod]
        public void ToNDArray_OffsetTensors_CopyTheirOwnElements()
        {
            var t = Tensor.Create(Enumerable.Range(0, 12).ToArray(), new nint[] { 3, 4 });   // [[0..3],[4..7],[8..11]]

            // dense slice: starts at backing[4]
            using (NDArray rows = t.Slice(new NRange[] { new NRange(1, 3), NRange.All }).ToNDArray())
            {
                rows.shape.Should().Equal(2L, 4L);
                rows.ToArray<int>().Should().Equal(4, 5, 6, 7, 8, 9, 10, 11);
            }

            // strided slice: starts at backing[1]
            using (NDArray cols = t.Slice(new NRange[] { NRange.All, new NRange(1, 3) }).ToNDArray())
            {
                cols.shape.Should().Equal(3L, 2L);
                cols.ToArray<int>().Should().Equal(1, 2, 5, 6, 9, 10);
            }

            // explicit start offset: a dense (2,3) tensor at backing[2]
            var started = Tensor.Create(new[] { -1, -2, 0, 1, 2, 3, 4, 5 }, 2, new nint[] { 2, 3 }, new nint[] { 3, 1 });
            using (NDArray s = started.ToNDArray())
                s.ToArray<int>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        /// <summary>
        ///     The zero-copy view of an offset tensor aliases the tensor's OWN elements: it reads them, and a write
        ///     through the NumSharp view lands on the tensor element it names and nowhere else. Regression: the view
        ///     was built over <c>GetPinnedHandle()</c>'s pointer (the backing array's head), so it read the wrong
        ///     elements and <c>rows[0,0] = -1</c> overwrote <c>backing[0]</c>, outside the slice.
        /// </summary>
        [TestMethod]
        public void AsNDArray_OffsetTensors_AliasTheirOwnElements()
        {
            int[] backing = Enumerable.Range(0, 12).ToArray();
            var t = Tensor.Create(backing, new nint[] { 3, 4 });

            using (NDArray rows = t.Slice(new NRange[] { new NRange(1, 3), NRange.All }).AsNDArray())   // dense view at backing[4]
            {
                rows.ToArray<int>().Should().Equal(4, 5, 6, 7, 8, 9, 10, 11);
                rows.SetInt32(-1, 0, 0);
                backing[4].Should().Be(-1, "rows[0,0] is t[1,0], which is backing[4]");
                backing[0].Should().Be(0, "a write through the slice must not reach outside it");
            }

            using (NDArray cols = t.Slice(new NRange[] { NRange.All, new NRange(1, 3) }).AsNDArray())   // strided view at backing[1]
            {
                cols.ToArray<int>().Should().Equal(1, 2, 5, 6, 9, 10);
                cols.SetInt32(-2, 2, 1);
                backing[10].Should().Be(-2, "cols[2,1] is t[2,2], which is backing[10]");
                backing[9].Should().Be(9);
            }

            int[] raw = { -1, -2, 0, 1, 2, 3, 4, 5 };
            var started = Tensor.Create(raw, 2, new nint[] { 2, 3 }, new nint[] { 3, 1 });
            using (NDArray s = started.AsNDArray())
            {
                s.ToArray<int>().Should().Equal(0, 1, 2, 3, 4, 5);
                s.SetInt32(99, 1, 2);
                raw[7].Should().Be(99, "s[1,2] is raw[2 + 1*3 + 2]");
                raw[0].Should().Be(-1, "the elements before the start offset are not the tensor's");
            }
        }

        /// <summary>
        ///     Offset tensors import byte-exactly for all 15 element types, through the copy (<c>ToNDArray</c>) and the
        ///     view (<c>AsNDArray</c>) alike: a dense slice, a strided slice and an explicit start offset, each checked
        ///     against the tensor's own <c>FlattenTo</c> (which has always honoured the offset).
        /// </summary>
        [TestMethod]
        public void OffsetTensors_AllDtypes_ImportTheirOwnBytes()
        {
            AssertOffsetImport<bool>(i => i % 3 == 1);
            AssertOffsetImport<byte>(i => (byte)(i * 7 + 1));
            AssertOffsetImport<sbyte>(i => (sbyte)(i * 7 - 40));
            AssertOffsetImport<short>(i => (short)(i * 301 - 1000));
            AssertOffsetImport<ushort>(i => (ushort)(i * 301 + 5));
            AssertOffsetImport<int>(i => i * 100003 - 7);
            AssertOffsetImport<uint>(i => (uint)(i * 100003 + 7));
            AssertOffsetImport<long>(i => i * 10000000019L - 3);
            AssertOffsetImport<ulong>(i => (ulong)i * 10000000019UL + 3);
            AssertOffsetImport<char>(i => (char)('A' + i));
            AssertOffsetImport<Half>(i => (Half)(i * 0.5 - 3));
            AssertOffsetImport<float>(i => i * 1.25f - 7.5f);
            AssertOffsetImport<double>(i => i * 1.125 - 9.75);
            AssertOffsetImport<decimal>(i => i * 1.5m - 2.25m);
            AssertOffsetImport<Complex>(i => new Complex(i * 1.5 - 2, 1 - i * 0.25));
        }

        /// <summary>
        ///     Imports three offset tensors over one 12-element backing array (a dense row slice, a strided column
        ///     slice, and a start-offset tensor) and checks that both import verbs return exactly the tensor's own
        ///     logical bytes.
        /// </summary>
        /// <typeparam name="T">The element type under test (one of the 15 NumSharp dtypes).</typeparam>
        /// <param name="gen">Produces the backing array's value at an index; the values must differ at the offsets.</param>
        private static void AssertOffsetImport<T>(Func<int, T> gen) where T : unmanaged
        {
            var backing = new T[12];
            for (int i = 0; i < backing.Length; i++)
                backing[i] = gen(i);
            var t = Tensor.Create(backing, new nint[] { 3, 4 });

            Tensor<T>[] offsetTensors =
            {
                t.Slice(new NRange[] { new NRange(1, 3), NRange.All }),
                t.Slice(new NRange[] { NRange.All, new NRange(1, 3) }),
                Tensor.Create(backing, 2, new nint[] { 2, 3 }, new nint[] { 3, 1 }),
            };
            foreach (Tensor<T> view in offsetTensors)
            {
                var expected = new T[(int)view.FlattenedLength];
                view.FlattenTo(expected);
                byte[] want = MemoryMarshal.AsBytes<T>(expected).ToArray();

                using (NDArray copy = view.ToNDArray())
                    BytesOf(copy).Should().Equal(want, $"{typeof(T).Name}: ToNDArray copies the tensor's own elements");
                using (NDArray alias = view.AsNDArray())
                    BytesOf(alias).Should().Equal(want, $"{typeof(T).Name}: AsNDArray aliases the tensor's own elements");
            }
        }

        // ---- the BCL's rank-0 values --------------------------------------------------------------------

        /// <summary>
        ///     The BCL's rank-0 values (<c>Tensor&lt;T&gt;.Empty</c>, <c>TensorSpan&lt;T&gt;.Empty</c>,
        ///     <c>ReadOnlyTensorSpan&lt;T&gt;.Empty</c>, a <c>default</c> span) hold NO element, so they import as the
        ///     empty vector (0,). Regression: they were read as NumSharp's 0-d shape, which claims one element —
        ///     <c>ToNDArray</c> threw IndexOutOfRangeException, and <c>AsNDArray(Tensor&lt;T&gt;.Empty)</c> returned a
        ///     0-d array over one element past the end of an empty managed array.
        /// </summary>
        [TestMethod]
        public void RankZeroValues_ImportAsTheEmptyVector()
        {
            using (NDArray nd = Tensor<float>.Empty.ToNDArray())
                nd.shape.Should().Equal(0L);
            using (NDArray nd = Tensor<float>.Empty.AsNDArray())
                nd.shape.Should().Equal(0L);
            using (NDArray nd = ReadOnlyTensorSpan<float>.Empty.ToNDArray())
                nd.shape.Should().Equal(0L);
            using (NDArray nd = TensorSpan<float>.Empty.ToNDArray())
                nd.shape.Should().Equal(0L);
            using (NDArray nd = default(ReadOnlyTensorSpan<double>).ToNDArray())
            {
                nd.shape.Should().Equal(0L);
                nd.typecode.Should().Be(NPTypeCode.Double);
            }
        }

        // ---- more than int.MaxValue elements ------------------------------------------------------------

        /// <summary>
        ///     A tensor or span with more than <see cref="int.MaxValue"/> elements (a stride-0 tensor needs no memory
        ///     for that) cannot be COPIED — <c>FlattenTo</c> fills an int-indexed <c>Span&lt;T&gt;</c> — and both
        ///     copies refuse it with NotSupportedException BEFORE allocating the result. Regression: the size check ran
        ///     after the allocation, so 2^40 float32 elements threw OutOfMemoryException trying to allocate 4 TB.
        /// </summary>
        [TestMethod]
        public void Copies_OverIntMaxValue_AreRefusedBeforeAllocating()
        {
            var huge = Tensor.Create(new[] { 3f }, new nint[] { (nint)(1L << 40) }, new nint[] { 0 });

            new Action(() => huge.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*Span<T>*");
            new Action(() => huge.AsReadOnlyTensorSpan().ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*Span<T>*");
        }

        /// <summary>
        ///     The zero-copy view has no element-count limit: a stride-0 tensor of 2^40 elements comes back as a
        ///     read-only NumSharp broadcast view over the tensor's single backing element.
        /// </summary>
        [TestMethod]
        public void AsNDArray_StrideZeroTensor_OverIntMaxValue_SharesZeroCopy()
        {
            var huge = Tensor.Create(new[] { 3f }, new nint[] { (nint)(1L << 40) }, new nint[] { 0 });
            using NDArray view = huge.AsNDArray();

            view.size.Should().Be(1L << 40);
            view.Shape.IsWriteable.Should().BeFalse("a stride-0 view is a broadcast, which NumSharp keeps read-only");
            view.GetSingle((1L << 40) - 1).Should().Be(3f);
        }
    }
}
