using System;
using System.Buffers;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Reference accounting and fault injection: every path through the bridge — success, refusal, failure half-way —
    ///     must leave NumSharp's buffer references and its buffer pool exactly balanced. A reference taken and never
    ///     released pins a buffer forever (and makes <c>resize(refcheck)</c> refuse); a reference released too early, or a
    ///     pin that takes none, lets the pool hand live memory to another array; a temporary dropped on an error path is a
    ///     pool leak. Each check reads the counted state directly (<see cref="IArraySlice.IsUniquelyReferenced"/>,
    ///     <see cref="IArraySlice.IsReleased"/>) or the pool's take/return balance (<see cref="Pool.Balance"/>).
    /// </summary>
    [TestClass]
    public class ArcAccountingTests : OnnxTestBase
    {
        /// <summary>
        ///     Every refusal leaves the source exactly as referenced as before: a check that pins the buffer, or converts
        ///     it, and then throws must release what it took. Each case refuses a view of a base array; once the view is
        ///     disposed the base must be uniquely referenced again, and the whole case must give back every buffer it took
        ///     (a Decimal conversion made before a refusal is a temporary nobody else can free).
        /// </summary>
        [TestMethod]
        public void Refusals_LeaveTheSourceUniquelyReferenced_AndThePoolBalanced()
        {
            using InferenceSession session = OpenSession("identity_int64");
            var cases = new (string Name, NPTypeCode Code, Func<NDArray, NDArray> MakeView, Action<NDArray> Refuse)[]
            {
                ("AsOrtValue(View) of a transposed view", NPTypeCode.Single, b => b.T, v => v.AsOrtValue(OnnxCodecMode.View)),
                ("AsOrtValue of Complex", NPTypeCode.Complex, b => b["::1"], v => v.AsOrtValue()),
                ("AsOrtValue(Copy) of Complex (a copy cannot give it an ONNX type)", NPTypeCode.Complex, b => b.T, v => v.AsOrtValue(OnnxCodecMode.Copy)),
                ("AsOrtValue(View) of a transposed Decimal", NPTypeCode.Decimal, b => b.T, v => v.AsOrtValue(OnnxCodecMode.View)),
                ("AsOrtValue(View) of a contiguous Decimal", NPTypeCode.Decimal, b => b["::1"], v => v.AsOrtValue(OnnxCodecMode.View)),
                ("AsDenseTensor(View) of a stepped view", NPTypeCode.Single, b => b["::2"], v => v.AsDenseTensor<float>(OnnxCodecMode.View)),
                ("AsDenseTensor<float> of Decimal (fed as float64)", NPTypeCode.Decimal, b => b["::2"], v => v.AsDenseTensor<float>()),
                ("AsDenseTensor<double>(View) of Decimal", NPTypeCode.Decimal, b => b["::2"], v => v.AsDenseTensor<double>(OnnxCodecMode.View)),
                ("AsDenseTensor with the wrong T", NPTypeCode.Single, b => b["::1"], v => v.AsDenseTensor<double>()),
                ("ToDenseTensor with the wrong T", NPTypeCode.Single, b => b.T, v => v.ToDenseTensor<int>()),
                ("ToOrtValue of Complex", NPTypeCode.Complex, b => b.T, v => v.ToOrtValue()),
                ("Run refuses an int32 -> int64 cast under 'no'", NPTypeCode.Int32, b => b.T, v => session.Run(v, casting: "no")),
                ("Run refuses a float64 -> int64 cast under 'safe'", NPTypeCode.Double, b => b.T, v => session.Run(v)),
            };

            foreach (var (name, code, makeView, refuse) in cases)
            {
                long balance = Pool.Balance(() =>
                {
                    using NDArray b = Arange(code, 6, 4);
                    NDArray view = makeView(b);
                    bool threw = false;
                    try
                    {
                        refuse(view);
                    }
                    catch (Exception)
                    {
                        threw = true;
                    }
                    Assert.IsTrue(threw, $"{name}: expected a refusal");
                    view.Dispose();
                    Assert.IsTrue(b.Storage.InternalArray.IsUniquelyReferenced, $"{name}: the refusal kept a reference on the source");
                });
                Assert.AreEqual(0L, balance, $"{name}: {balance} buffer(s) taken and never returned");
            }
        }

        /// <summary>
        ///     Each export handle holds exactly ONE reference on the source's buffer, and releases it exactly once: after the
        ///     first <c>Dispose</c> the source is uniquely referenced again, and a second <c>Dispose</c> neither releases it a
        ///     second time (which would free the buffer under its owner) nor throws.
        /// </summary>
        [TestMethod]
        public void Handles_HoldOneReference_ReleasedExactlyOnce()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            IArraySlice slice = nd.Storage.InternalArray;
            var exports = new (string Name, Func<IDisposable> Export)[]
            {
                ("AsOrtValue", () => nd.AsOrtValue()),
                ("AsDenseTensor", () => nd.AsDenseTensor<float>()),
            };
            foreach (var (name, export) in exports)
            {
                IDisposable h = export();
                Assert.IsFalse(slice.IsUniquelyReferenced, $"{name}: the handle holds no reference");
                h.Dispose();
                Assert.IsTrue(slice.IsUniquelyReferenced, $"{name}: Dispose did not release the handle's reference");
                h.Dispose();
                Assert.IsTrue(slice.IsUniquelyReferenced, $"{name}: a second Dispose changed the count");
                Assert.IsFalse(slice.IsReleased, $"{name}: a second Dispose released the owner's own reference");
            }
        }

        /// <summary>
        ///     A handle over a COPY (<see cref="OnnxCodecMode.Auto"/> for an array ORT cannot read in place,
        ///     <see cref="OnnxCodecMode.Copy"/> for any array) holds no reference on the caller's buffer and owns its
        ///     temporary: while the handle lives the source is uniquely referenced once its view is gone, and after the
        ///     handle is disposed the whole export has given back every buffer it took (for a non-contiguous Decimal view
        ///     that is two temporaries: the conversion and its C-order copy).
        /// </summary>
        [TestMethod]
        public void CopyHandles_TakeNoReferenceOnTheSource_AndReleaseTheirTemporaries()
        {
            var exports = new (string Name, NPTypeCode Code, Func<NDArray, NDArray> MakeView, Func<NDArray, IDisposable> Export)[]
            {
                ("AsOrtValue (Auto) of a transposed view", NPTypeCode.Single, b => b.T, v => v.AsOrtValue()),
                ("AsOrtValue (Auto) of a transposed Decimal", NPTypeCode.Decimal, b => b.T, v => v.AsOrtValue()),
                ("AsOrtValue (Copy) of a contiguous array", NPTypeCode.Single, b => b["::1"], v => v.AsOrtValue(OnnxCodecMode.Copy)),
                ("AsOrtValue (Copy) of a contiguous Decimal", NPTypeCode.Decimal, b => b["::1"], v => v.AsOrtValue(OnnxCodecMode.Copy)),
                ("AsDenseTensor (Auto) of a stepped view", NPTypeCode.Single, b => b["::2"], v => v.AsDenseTensor<float>()),
                ("AsDenseTensor (Auto) of a transposed Decimal", NPTypeCode.Decimal, b => b.T, v => v.AsDenseTensor<double>()),
                ("AsDenseTensor (Copy) of a contiguous array", NPTypeCode.Single, b => b["::1"], v => v.AsDenseTensor<float>(OnnxCodecMode.Copy)),
            };
            foreach (var (name, code, makeView, export) in exports)
            {
                long balance = Pool.Balance(() =>
                {
                    using NDArray b = Arange(code, 6, 4);
                    NDArray view = makeView(b);
                    IDisposable h = export(view);
                    view.Dispose();
                    Assert.IsTrue(b.Storage.InternalArray.IsUniquelyReferenced, $"{name}: a handle over a copy holds a reference on the source");
                    h.Dispose();
                });
                Assert.AreEqual(0L, balance, $"{name}: {balance} buffer(s) taken and never returned");
            }
        }

        /// <summary>
        ///     The Decimal → float64 conversion a handle makes is a temporary only the handle can free: disposing the handle
        ///     must give its buffer back (the whole export balances to zero once the source is disposed too), for both
        ///     zero-copy verbs.
        /// </summary>
        [TestMethod]
        public void DecimalConversion_IsReleasedWithTheHandle()
        {
            Assert.AreEqual(0L, Pool.Balance(() =>
            {
                using NDArray nd = Arange(NPTypeCode.Decimal, 3, 4);
                using (OrtTensor h = nd.AsOrtValue())
                    Assert.AreEqual(TensorElementType.Double, h.ElementType);
            }), "AsOrtValue's conversion was not released");
            Assert.AreEqual(0L, Pool.Balance(() =>
            {
                using NDArray nd = Arange(NPTypeCode.Decimal, 3, 4);
                using (OrtTensor<double> h = nd.AsDenseTensor<double>())
                    Assert.AreEqual(4, h.Tensor.Dimensions[1]);
            }), "AsDenseTensor's conversion was not released");
        }

        /// <summary>
        ///     <c>session.Run</c> casts and densifies inputs into temporaries the input handles own; once the outputs are
        ///     disposed, a run gives back every buffer it took — for a cast input, a strided input and a Decimal input.
        /// </summary>
        [TestMethod]
        public void SessionRun_ReleasesEveryTemporary()
        {
            using InferenceSession f64 = OpenSession("identity_float64");
            using InferenceSession i64 = OpenSession("identity_int64");
            var runs = new (string Name, InferenceSession Session, NPTypeCode Code, Func<NDArray, NDArray> MakeView)[]
            {
                ("cast int32 -> int64", i64, NPTypeCode.Int32, b => b["::1"]),
                ("strided float64", f64, NPTypeCode.Double, b => b.T),
                ("cast and strided int32 -> int64", i64, NPTypeCode.Int32, b => b["::2"]),
                ("Decimal -> float64, strided", f64, NPTypeCode.Decimal, b => b.T),
            };
            foreach (var (name, session, code, makeView) in runs)
            {
                long balance = Pool.Balance(() =>
                {
                    using NDArray b = Arange(code, 4, 3);
                    using NDArray view = makeView(b);
                    using NDArray output = session.Run(view);
                    Assert.AreEqual(view.size, output.size, name);
                });
                Assert.AreEqual(0L, balance, $"{name}: {balance} buffer(s) taken and never returned");
            }
        }

        /// <summary>
        ///     A pin taken on <see cref="OrtTensor{T}.Memory"/> keeps the NumSharp buffer alive for as long as the pin lives —
        ///     past the handle's <c>Dispose</c> and past the source's. The memory manager behind it used to take no reference
        ///     at all, so every consumer of the documented <c>Memory</c> route read freed memory once the handle and source
        ///     were gone (a probe read another array's 777s): a NumSharp view over the shared <see cref="DenseTensor{T}"/>
        ///     (<c>tensor.AsNDArray()</c>), an <see cref="OrtValue"/> built by <c>CreateTensorValueFromMemory</c>, and a raw
        ///     <see cref="Memory{T}.Pin"/>. Once every pin is released the buffer is released exactly once.
        /// </summary>
        [TestMethod]
        public void MemoryPins_KeepTheBufferAlive_PastTheHandleAndTheSource()
        {
            var routes = new (string Name, Func<OrtTensor<double>, IDisposable> Pin, Func<IDisposable, double[]> Read)[]
            {
                ("a NumSharp view over the DenseTensor", h => h.Tensor.AsNDArray(), p => ((NDArray)p).ToArray<double>()),
                ("an OrtValue from CreateTensorValueFromMemory", h => OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { 2, 3 }),
                    p =>
                    {
                        using NDArray copy = ((OrtValue)p).ToNDArray();
                        return copy.ToArray<double>();
                    }),
                ("a raw Memory<T>.Pin()", h => new PinBox(h.Memory.Pin()), p => ((PinBox)p).Read(6)),
            };

            foreach (var (name, pin, read) in routes)
            {
                NDArray nd = Arange(NPTypeCode.Double, 2, 3);
                IArraySlice slice = nd.Storage.InternalArray;
                OrtTensor<double> h = nd.AsDenseTensor<double>();
                IDisposable pinned = pin(h);
                h.Dispose();
                nd.Dispose();

                Assert.IsFalse(slice.IsReleased, $"{name}: the buffer was released while the pin still reads it");
                using (NDArray other = np.full(new Shape(6), 777.0))   // would take the buffer if it had gone back to the pool
                    CollectionAssert.AreEqual(new[] { 0.0, 1, 2, 3, 4, 5 }, read(pinned), $"{name}: the pin read another array's data");

                pinned.Dispose();
                Assert.IsTrue(slice.IsReleased, $"{name}: releasing the last pin did not release the buffer");
            }
        }

        /// <summary>
        ///     Once the handle is disposed, its <see cref="Memory{T}"/> can no longer be pinned (or read): a pin taken after the
        ///     release would address memory the handle no longer holds.
        /// </summary>
        [TestMethod]
        public void Memory_AfterTheHandleIsDisposed_CannotBePinned()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 4);
            OrtTensor<float> h = nd.AsDenseTensor<float>();
            Memory<float> memory = h.Memory;
            h.Dispose();
            Assert.ThrowsException<ObjectDisposedException>(() => memory.Pin());
            Assert.ThrowsException<ObjectDisposedException>(() => memory.Span[0]);
            Assert.IsTrue(nd.Storage.InternalArray.IsUniquelyReferenced, "the refused pin took a reference");
        }

        /// <summary>
        ///     While a pin on the exported memory lives, the buffer is referenced, so <c>resize(refcheck: true)</c> refuses
        ///     to reallocate it (NumPy's rule for a referenced buffer); once the pin is released it works.
        /// </summary>
        [TestMethod]
        public void Resize_RefusesWhileAMemoryPinLives()
        {
            using NDArray nd = Arange(NPTypeCode.Double, 4);
            OrtTensor<double> h = nd.AsDenseTensor<double>();
            MemoryHandle pin = h.Memory.Pin();
            h.Dispose();
            Assert.ThrowsException<IncorrectShapeException>(() => nd.resize(new Shape(8)));
            pin.Dispose();
            nd.resize(new Shape(8));
            Assert.AreEqual(8L, nd.size);
        }

        /// <summary>
        ///     Fault injection on the map reader: a map whose VALUES are strings converts its keys first and then fails on the
        ///     values. The keys array it had already allocated used to be dropped (a pool leak on every such call); it is now
        ///     released, and the refusal says how to read string values.
        /// </summary>
        [TestMethod]
        public void ToMap_WithStringValues_ReleasesWhatItAllocated_AndSaysHowToReadThem()
        {
            using OrtValue map = OrtValue.CreateMapWithStringValues(new long[] { 1, 2, 3 }, new[] { "a", "b", "c" });
            NotSupportedException refusal = null;
            long balance = Pool.Balance(() => refusal = Assert.ThrowsException<NotSupportedException>(() => map.ToMap()));
            Assert.AreEqual(0L, balance, "the keys array converted before the failure was not released");
            StringAssert.Contains(refusal.Message, "STRING values");
            StringAssert.Contains(refusal.Message, "ReadStringTensor");
        }

        /// <summary>
        ///     A NumSharp view over a LEGACY run's output tensor (<c>DisposableNamedOnnxValue.AsTensor&lt;T&gt;()</c> →
        ///     <c>AsNDArray</c>) keeps reading the output after the results collection is dropped (not disposed), a full
        ///     collection runs, and more runs reuse ORT's memory. ORT's memory manager for such a tensor pins nothing, so the
        ///     view used to keep nothing alive: the output's <see cref="OrtValue"/> was finalized under it and the next runs'
        ///     outputs showed through (20 of 20 rounds in a probe). The view now roots the tensor, the rule
        ///     <c>AsNDArray(OrtValue)</c> already follows; once the view is disposed the output is collectable again.
        /// </summary>
        [TestMethod]
        public void ViewOverALegacyOutput_SurvivesTheResultsBeingDropped()
        {
            using InferenceSession session = OpenSession("identity_float32");
            for (int round = 0; round < 8; round++)
            {
                float fill = 1000f + round;
                NDArray view = ViewOverDroppedLegacyOutput(session, fill);
                GcQuiescence.CollectFull();
                for (int k = 0; k < 8; k++)
                    OverwriteWithAnotherRun(session);
                using (view)
                {
                    Assert.AreEqual(fill, view.GetSingle(0), $"round {round}: the view read freed memory");
                    Assert.AreEqual(fill, view.GetSingle(4095), $"round {round}: the view read freed memory");
                }
            }
        }

        /// <summary>
        ///     Runs the identity model through the legacy API and returns a view over its output tensor, DROPPING the
        ///     results collection without disposing it. A separate, non-inlined frame, so nothing in the caller keeps the
        ///     collection reachable.
        /// </summary>
        /// <param name="session">An identity session for float32.</param>
        /// <param name="fill">The value every element is set to.</param>
        /// <returns>The view; the caller disposes it.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static NDArray ViewOverDroppedLegacyOutput(InferenceSession session, float fill)
        {
            var x = new DenseTensor<float>(new[] { 4096 });
            x.Buffer.Span.Fill(fill);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("X", x) });
            var tensor = (DenseTensor<float>)results[0].AsTensor<float>();
            return tensor.AsNDArray();
        }

        /// <summary>One more legacy run of the same shape, disposed at once: ORT hands its output the memory the last one freed.</summary>
        /// <param name="session">An identity session for float32.</param>
        private static void OverwriteWithAnotherRun(InferenceSession session)
        {
            var x = new DenseTensor<float>(new[] { 4096 });
            x.Buffer.Span.Fill(-7f);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("X", x) });
        }

        /// <summary>
        ///     A refusal decided by the dtype and the shape alone takes nothing from the pool, even for a Decimal source the
        ///     verb would otherwise convert: <c>AsDenseTensor</c> converted a Decimal array to float64 BEFORE checking the
        ///     requested element type, the element count and the <c>int</c> dimensions, so a refused call still paid a
        ///     full-size conversion (and, for a broadcast view, converted every element it addresses).
        /// </summary>
        [TestMethod]
        public void ShapeDecidedRefusals_TakeNothing_EvenForADecimalSource()
        {
            // Every array the cases refuse is made OUTSIDE the measured calls: only the bridge's own takes may count.
            using NDArray dec = np.zeros(new Shape(3, 4), NPTypeCode.Decimal);
            using NDArray floats = Arange(NPTypeCode.Single, 3, 4);
            using NDArray complex = np.zeros(new Shape(2), NPTypeCode.Complex);
            using (NDArray decT = dec.T)
            {
                var cases = new (string Name, Action Refuse)[]
                {
                    ("AsDenseTensor<float> of Decimal (fed as float64)", () => Assert.ThrowsException<ArgumentException>(() => dec.AsDenseTensor<float>())),
                    ("AsDenseTensor<float> of a transposed Decimal", () => Assert.ThrowsException<ArgumentException>(() => decT.AsDenseTensor<float>())),
                    ("ToDenseTensor<float> of Decimal", () => Assert.ThrowsException<ArgumentException>(() => dec.ToDenseTensor<float>())),
                    ("AsDenseTensor<double> of float32", () => Assert.ThrowsException<ArgumentException>(() => floats.AsDenseTensor<double>())),
                    ("AsDenseTensor of Complex", () => Assert.ThrowsException<NotSupportedException>(() => complex.AsDenseTensor<double>())),
                    // The mode's own refusals are decided from the dtype and the layout too: the strict mode must not convert
                    // or copy an array before it refuses it, and Copy must not copy an array that has no ONNX type.
                    ("AsOrtValue(View) of a transposed Decimal", () => Assert.ThrowsException<InvalidOperationException>(() => decT.AsOrtValue(OnnxCodecMode.View))),
                    ("AsOrtValue(View) of Decimal", () => Assert.ThrowsException<InvalidOperationException>(() => dec.AsOrtValue(OnnxCodecMode.View))),
                    ("AsDenseTensor<double>(View) of a transposed Decimal", () => Assert.ThrowsException<InvalidOperationException>(() => decT.AsDenseTensor<double>(OnnxCodecMode.View))),
                    ("AsOrtValue(Copy) of Complex", () => Assert.ThrowsException<NotSupportedException>(() => complex.AsOrtValue(OnnxCodecMode.Copy))),
                    ("AsDenseTensor(Copy) of Complex", () => Assert.ThrowsException<NotSupportedException>(() => complex.AsDenseTensor<double>(OnnxCodecMode.Copy))),
                };
                foreach (var (name, refuse) in cases)
                {
                    refuse();   // first call unmeasured: static caches a refusal path may fill are not this test's subject
                    Assert.AreEqual(0L, Pool.Takes(refuse), $"{name}: allocated before refusing");
                }
            }
            Assert.IsTrue(dec.Storage.InternalArray.IsUniquelyReferenced, "a refusal left a reference on the Decimal source");
        }

        /// <summary>
        ///     A <see cref="DenseTensor{T}"/> kept past its <c>AsDenseTensor</c> handle's <c>Dispose</c> is refused by both
        ///     imports with <see cref="ObjectDisposedException"/> — empty or not — and BEFORE anything is allocated: the
        ///     copy pins the buffer first, so a stale tensor never costs a result array that then has to be thrown away.
        /// </summary>
        [TestMethod]
        public void TensorOfADisposedHandle_IsRefusedByBothImports_BeforeAllocating()
        {
            foreach (int n in new[] { 0, 6 })
            {
                using NDArray source = Arange(NPTypeCode.Single, n);
                OrtTensor<float> h = source.AsDenseTensor<float>();
                DenseTensor<float> stale = h.Tensor;
                h.Dispose();
                long takes = Pool.Takes(() =>
                {
                    Assert.ThrowsException<ObjectDisposedException>(() => stale.ToNDArray(), $"{n} elements: ToNDArray");
                    Assert.ThrowsException<ObjectDisposedException>(() => stale.AsNDArray(), $"{n} elements: AsNDArray");
                });
                Assert.AreEqual(0L, takes, $"{n} elements: an import allocated before refusing the stale tensor");
                Assert.IsTrue(source.Storage.InternalArray.IsUniquelyReferenced, $"{n} elements: a refusal left a reference behind");
            }
        }

        /// <summary>A <see cref="MemoryHandle"/> as an <see cref="IDisposable"/> the routes above can treat uniformly.</summary>
        private sealed unsafe class PinBox : IDisposable
        {
            private MemoryHandle _handle;

            /// <summary>Takes ownership of a pin.</summary>
            /// <param name="handle">The pin.</param>
            internal PinBox(MemoryHandle handle) => _handle = handle;

            /// <summary>Reads the pinned doubles.</summary>
            /// <param name="count">How many to read.</param>
            /// <returns>A copy of them.</returns>
            internal double[] Read(int count) => new ReadOnlySpan<double>(_handle.Pointer, count).ToArray();

            /// <summary>Releases the pin.</summary>
            public void Dispose() => _handle.Dispose();
        }
    }
}
