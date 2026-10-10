using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Boundary values: the sizes where a byte count stops fitting in an <c>int</c> (ORT's managed API hands tensor
    ///     memory out as a <see cref="Span{T}"/>, a <see cref="DenseTensor{T}"/> is <c>int</c>-indexed), the exact
    ///     refusal boundary per element size, a zero-size array with an axis past <see cref="int.MaxValue"/>, and the pins
    ///     ORT's legacy <c>session.Run</c> takes on a NumSharp buffer. Tests past 2 GB use NumSharp memory that is never
    ///     touched where they can (virtual only); the one that must copy 2 GB runs only on machines with the memory and is
    ///     Inconclusive elsewhere.
    /// </summary>
    [TestClass]
    public class BoundaryTests : OnnxTestBase
    {
        /// <summary>The physical memory a test that really copies multi-gigabyte buffers needs before it runs.</summary>
        private const long BigTestMemory = 12L << 30;

        /// <summary>
        ///     The physical memory a test that only RESERVES multi-gigabyte buffers (never touched, so never backed) needs:
        ///     enough that the reservation cannot fail outright on a small machine.
        /// </summary>
        private const long VirtualTestMemory = 8L << 30;

        /// <summary>
        ///     A tensor of 2 GB or more that ORT owns cannot be read out of an <see cref="OrtValue"/>: ORT exposes the
        ///     memory only as a <see cref="Span{T}"/> of bytes whose length it casts to <c>int</c> unchecked. Both imports
        ///     refuse it up front with <see cref="NotSupportedException"/> naming the route that works (a pre-allocated
        ///     output), take nothing from the pool, and leave the value alive and the caller's (even when asked to own it,
        ///     in either mode that can own). Before: <c>ToNDArray</c> allocated the whole result first and abandoned it to
        ///     the finalizer when ORT's span failed (ArgumentOutOfRangeException at exactly 2 GB and at 2 GB + 16; an "ORT
        ///     reports 16 tensor bytes" InvalidOperationException at 4 GB + 16), and <c>AsNDArray</c> threw at 2 GB + 16
        ///     but returned a working view at 4 GB + 16, where the truncated span happened to start at the right address —
        ///     the outcome depended on bit 31 of the byte count.
        /// </summary>
        /// <remarks>
        ///     The values are made with ORT's own factory over reserved NumSharp memory, not with <c>AsOrtValue</c>: a value
        ///     the package made is NumSharp memory to the import verbs and is viewed at any size (the next test).
        /// </remarks>
        [TestMethod]
        public unsafe void OrtValueImports_OfTwoGigabytesOrMore_AreRefusedBeforeAnythingIsAllocated()
        {
            RequireMemory(VirtualTestMemory);
            var sizes = new (NPTypeCode Code, long Count, string What)[]
            {
                (NPTypeCode.Double, 1L << 28, "float64 x 2^28 = exactly 2^31 bytes"),
                (NPTypeCode.Byte, (1L << 31) + 16, "uint8, 2 GB + 16 bytes"),
                (NPTypeCode.Byte, (1L << 32) + 16, "uint8, 4 GB + 16 bytes"),
            };
            foreach (var (code, count, what) in sizes)
            {
                // Never written: the source and the tensor over it are address space only.
                using var source = new NDArray(code, new Shape(count), fillZeros: false);
                using OrtValue foreign = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, Oracle.ElementType(code),
                    new[] { count }, (IntPtr)source.Storage.InternalArray.Address, count * code.SizeOf());
                foreach (string verb in new[] { "ToNDArray", "AsNDArray", "AsNDArray(ownsValue)", "AsNDArray(ownsValue, Copy)" })
                {
                    long takes = Pool.Takes(() =>
                    {
                        var e = Assert.ThrowsException<NotSupportedException>(() =>
                        {
                            using NDArray made = verb switch
                            {
                                "ToNDArray" => foreign.ToNDArray(),
                                "AsNDArray" => foreign.AsNDArray(),
                                "AsNDArray(ownsValue)" => foreign.AsNDArray(ownsValue: true),
                                _ => foreign.AsNDArray(ownsValue: true, mode: OnnxCodecMode.Copy),
                            };
                        }, $"{what}: {verb}");
                        StringAssert.Contains(e.Message, "2 GB", $"{what}: {verb}");
                        StringAssert.Contains(e.Message, "pre-allocate", $"{what}: {verb} must name the route that works");
                    });
                    Assert.AreEqual(0L, takes, $"{what}: {verb} allocated before refusing");
                    // The refusal leaves the value alive and the caller's: it still answers.
                    Assert.AreEqual(count, foreign.GetTensorTypeAndShape().ElementCount, $"{what}: {verb} disposed the value");
                }
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     A value the package exported is NumSharp memory, so a view of it needs no span: at 4 GB + 16 bytes the view
        ///     is the buffer itself (same address, same element count), takes nothing from the pool, and outlives the
        ///     handle. Such a value was refused like an ORT-owned one, although nothing about it needs ORT's managed
        ///     data API.
        /// </summary>
        [TestMethod]
        public unsafe void ExportedValue_PastTwoGigabytes_IsViewedAsNumSharpMemory_WithoutAllocating()
        {
            RequireMemory(VirtualTestMemory);
            long count = (1L << 32) + 16;
            // Never written: address space only.
            using var source = new NDArray(NPTypeCode.Byte, new Shape(count), fillZeros: false);
            OrtTensor h = source.AsOrtValue();
            NDArray view = null;
            long takes = Pool.Takes(() => view = h.Value.AsNDArray());
            using (view)
            {
                Assert.AreEqual(0L, takes, "a view allocates no buffer");
                Assert.AreEqual(count, view.size);
                Assert.AreEqual((IntPtr)source.Storage.InternalArray.Address, (IntPtr)view.Storage.InternalArray.Address, "the view is the exported buffer");
                Assert.IsFalse(view.flags.owndata);
                h.Dispose();
                Assert.IsFalse(source.Storage.InternalArray.IsUniquelyReferenced, "the view holds its own reference past the handle");
            }
            Assert.IsTrue(source.Storage.InternalArray.IsUniquelyReferenced, "disposing the view released its reference");
        }

        /// <summary>
        ///     A <see cref="DenseTensor{T}"/> past 2 GB of bytes (2^30 + 1 ushorts) copies whole through
        ///     <c>ToNDArray</c> and through the legacy <see cref="NamedOnnxValue"/> route, byte for byte — and its view
        ///     still works. The copy used to count bytes in an <c>int</c> (<c>MemoryMarshal.AsBytes</c>) and threw
        ///     OverflowException for every such tensor, although NumSharp holds it and <c>AsDenseTensor</c> /
        ///     <c>ToDenseTensor</c> produce it.
        /// </summary>
        [TestMethod]
        public unsafe void DenseTensor_PastTwoGigabytes_CopiesWhole()
        {
            RequireMemory(BigTestMemory);
            int n = (1 << 30) + 1;
            ushort[] data = GC.AllocateUninitializedArray<ushort>(n);
            // Sentinels around both ends and the 2 GB byte boundary, where an int byte count first breaks.
            foreach (int i in new[] { 0, 1, (1 << 30) - 1, 1 << 30, n / 2, n - 1 })
                data[i] = (ushort)(i * 7 + 13);
            var tensor = new DenseTensor<ushort>(data, new[] { n });

            using (NDArray copy = tensor.ToNDArray())
                AssertSameElements(copy, data, "ToNDArray");
            using (NDArray copy = NamedOnnxValue.CreateFromTensor("X", tensor).ToNDArray())
                AssertSameElements(copy, data, "NamedOnnxValue.ToNDArray");
            using (NDArray view = tensor.AsNDArray())
                AssertSameElements(view, data, "AsNDArray");
            GC.KeepAlive(data);
        }

        /// <summary>
        ///     <c>ToOrtValue</c> refuses exactly from the first element count whose bytes pass <see cref="int.MaxValue"/>,
        ///     for every element size (1, 2, 4, 8, and Decimal fed as 8), and allocates nothing to refuse. The sources are
        ///     broadcast views of one element, so the sizes cost no memory.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_RefusesExactlyPastTheSpanLimit_ForEveryItemSize()
        {
            foreach (NPTypeCode code in new[] { NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Int64, NPTypeCode.Double, NPTypeCode.Decimal })
            {
                int fed = code == NPTypeCode.Decimal ? 8 : code.SizeOf();
                long first = (long)int.MaxValue / fed + 1;   // the first element count whose bytes pass int.MaxValue
                using NDArray one = np.zeros(new Shape(1), code);
                using NDArray past = np.broadcast_to(one, new Shape(first));
                long takes = Pool.Takes(() =>
                {
                    var e = Assert.ThrowsException<NotSupportedException>(() => past.ToOrtValue(), $"{code} x {first}");
                    StringAssert.Contains(e.Message, "2 GB", code.ToString());
                    StringAssert.Contains(e.Message, "AsOrtValue", $"{code}: the message must name the zero-copy export");
                    Assert.IsFalse(e.Message.Contains("AsNDArray"), $"{code}: the message must not send an export to an import verb");
                });
                Assert.AreEqual(0L, takes, $"{code}: ToOrtValue allocated before refusing");
            }
        }

        /// <summary>
        ///     A zero-size array with an axis longer than <see cref="int.MaxValue"/>: the two dense-tensor verbs refuse it
        ///     (<see cref="DenseTensor{T}"/> dimensions are <c>int</c>) naming the axis, while the OrtValue verbs carry the
        ///     <c>long</c> dimensions through ORT and back exactly — shape, dtype, and nothing to read.
        /// </summary>
        [TestMethod]
        public void ZeroSizeArray_WithAnAxisPastIntMaxValue_CrossesAsOrtValueOnly()
        {
            long huge = (long)int.MaxValue + 1;
            using NDArray empty = np.zeros(new Shape(0, huge), NPTypeCode.Single);
            Assert.AreEqual(0L, empty.size);

            var asDense = Assert.ThrowsException<NotSupportedException>(() => empty.AsDenseTensor<float>());
            StringAssert.Contains(asDense.Message, $"axis 1 has {huge}");
            var toDense = Assert.ThrowsException<NotSupportedException>(() => empty.ToDenseTensor<float>());
            StringAssert.Contains(toDense.Message, $"axis 1 has {huge}");

            using (OrtTensor h = empty.AsOrtValue())
            {
                CollectionAssert.AreEqual(new[] { 0L, huge }, h.Value.GetTensorTypeAndShape().Shape, "AsOrtValue dims");
                using NDArray back = h.Value.ToNDArray();
                CollectionAssert.AreEqual(new[] { 0L, huge }, back.shape, "AsOrtValue -> ToNDArray shape");
                using NDArray view = h.Value.AsNDArray();
                CollectionAssert.AreEqual(new[] { 0L, huge }, view.shape, "AsOrtValue -> AsNDArray shape");
            }
            using (OrtValue copy = empty.ToOrtValue())
            {
                CollectionAssert.AreEqual(new[] { 0L, huge }, copy.GetTensorTypeAndShape().Shape, "ToOrtValue dims");
                using NDArray back = copy.ToNDArray();
                CollectionAssert.AreEqual(new[] { 0L, huge }, back.shape, "ToOrtValue -> ToNDArray shape");
                Assert.AreEqual(NPTypeCode.Single, back.typecode);
            }
        }

        /// <summary>
        ///     ORT's legacy <c>session.Run(NamedOnnxValue…)</c> pins the input tensor's <see cref="Memory{T}"/> for the run.
        ///     Over an <c>AsDenseTensor</c> handle every such pin takes a reference on the NumSharp buffer; once the run
        ///     returns and its results are disposed, every one of them must have been released — the source ends uniquely
        ///     referenced as soon as the handle is disposed, for every dtype ORT's identity model accepts, run after run.
        /// </summary>
        [TestMethod]
        public void LegacyRun_ReleasesEveryPinItTakesOnANumSharpBuffer()
        {
            foreach (NPTypeCode code in new[] { NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Int64, NPTypeCode.Byte, NPTypeCode.Boolean, NPTypeCode.Half })
            {
                using InferenceSession session = OpenSession(IdentityModel(code));
                using NDArray source = Arange(code, 3, 4);
                IArraySlice slice = source.Storage.InternalArray;
                for (int run = 0; run < 3; run++)
                {
                    switch (code)
                    {
                        case NPTypeCode.Single: LegacyRun<float>(session, source); break;
                        case NPTypeCode.Double: LegacyRun<double>(session, source); break;
                        case NPTypeCode.Int32: LegacyRun<int>(session, source); break;
                        case NPTypeCode.Int64: LegacyRun<long>(session, source); break;
                        case NPTypeCode.Byte: LegacyRun<byte>(session, source); break;
                        case NPTypeCode.Boolean: LegacyRun<bool>(session, source); break;
                        case NPTypeCode.Half: LegacyRun<Float16>(session, source); break;
                    }
                    Assert.IsTrue(slice.IsUniquelyReferenced, $"{code} run {run}: a pin ORT took on the NumSharp buffer was never released");
                }
            }
        }

        /// <summary>One legacy run over an <c>AsDenseTensor</c> handle, its single output checked against the source.</summary>
        /// <typeparam name="T">ORT's element type for the source's dtype.</typeparam>
        /// <param name="session">An identity session for the dtype.</param>
        /// <param name="source">The array fed (and expected back).</param>
        private static void LegacyRun<T>(InferenceSession session, NDArray source) where T : unmanaged
        {
            using OrtTensor<T> h = source.AsDenseTensor<T>();
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("X", h.Tensor) }))
            using (NDArray back = results.First().ToNDArray())
                Assert.IsTrue(Oracle.SameBytes(Oracle.LogicalBytes(back), Oracle.LogicalBytes(source)), $"{source.typecode}: the run's output differs");
        }

        /// <summary>Asserts a 1-D array holds exactly <paramref name="want"/>'s elements, compared as raw memory.</summary>
        /// <param name="actual">The array (C-contiguous).</param>
        /// <param name="want">The expected elements.</param>
        /// <param name="what">The route, for the message.</param>
        private static unsafe void AssertSameElements(NDArray actual, ushort[] want, string what)
        {
            Assert.AreEqual((long)want.Length, actual.size, $"{what}: size");
            Assert.AreEqual(NPTypeCode.UInt16, actual.typecode, $"{what}: dtype");
            byte* a = (byte*)actual.Storage.Address + actual.Shape.Offset * sizeof(ushort);
            fixed (ushort* w = want)
            {
                long bytes = (long)want.Length * sizeof(ushort);
                // Compared in 1 GB chunks: a span cannot cover the whole buffer, which is the point of the test.
                for (long off = 0; off < bytes; off += 1 << 30)
                {
                    int chunk = (int)Math.Min(1 << 30, bytes - off);
                    Assert.IsTrue(new ReadOnlySpan<byte>(a + off, chunk).SequenceEqual(new ReadOnlySpan<byte>((byte*)w + off, chunk)),
                        $"{what}: bytes differ in the chunk at {off}");
                }
            }
        }

        /// <summary>Skips the calling test (Inconclusive) unless the machine has at least <paramref name="bytes"/> of physical memory.</summary>
        /// <param name="bytes">The memory the test needs.</param>
        /// <exception cref="AssertInconclusiveException">The machine has less.</exception>
        private static void RequireMemory(long bytes)
        {
            long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (total < bytes)
                Assert.Inconclusive($"needs {bytes >> 30} GB of physical memory, the machine has {total >> 30} GB");
        }
    }
}
