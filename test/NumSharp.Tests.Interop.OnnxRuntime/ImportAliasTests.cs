using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The import side of a round trip: a NumSharp view of a value THIS PACKAGE exported (<c>nd.AsOrtValue().Value</c>,
    ///     <c>nd.AsDenseTensor&lt;T&gt;().Tensor</c>) is NumSharp memory, not foreign memory. It must be a counted alias of
    ///     the buffer (valid after the handle and the source are gone), inherit the source's read-only flag, and follow the
    ///     ownership and mode rules of <c>AsNDArray</c>. Also here: what an import does to the caller's value when it
    ///     fails, and when a view reports memory pressure.
    /// </summary>
    [TestClass]
    public class ImportAliasTests : OnnxTestBase
    {
        /// <summary>
        ///     The defect this pins: <c>nd.AsOrtValue().Value.AsNDArray()</c> was leased like an ORT buffer, so the view
        ///     kept only the OrtValue alive, and that value owns nothing. With the handle and the source disposed the
        ///     buffer went back to the pool, and the view read the next array allocated there (777 instead of 0..5). The
        ///     view is now an alias that holds its own counted reference: the buffer is released when the view is.
        /// </summary>
        [TestMethod]
        public void ViewOfAnExportedValue_IsACountedAlias_ValidAfterTheHandleAndTheSourceAreDisposed()
        {
            NDArray nd = Arange(NPTypeCode.Double, 6);
            IArraySlice buffer = nd.Storage.InternalArray;
            OrtTensor h = nd.AsOrtValue();
            NDArray view = h.Value.AsNDArray();

            view.flags.owndata.Should().BeFalse("a view never owns its buffer");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "an alias of NumSharp memory is not a lease on foreign memory");

            h.Dispose();
            nd.Dispose();
            buffer.IsReleased.Should().BeFalse("the view holds its own reference on the buffer");
            using (NDArray other = np.full(new Shape(6), 777.0))   // would take the buffer if it had gone back to the pool
                view.ToArray<double>().Should().Equal(new[] { 0.0, 1, 2, 3, 4, 5 }, "the view must not read another array's data");

            view.Dispose();
            buffer.IsReleased.Should().BeTrue("the view's reference was the last one");
        }

        /// <summary>
        ///     <c>ownsValue: true</c> on an exported value: the handle (and with it the OrtValue) is disposed as soon as
        ///     the view exists, and the view goes on sharing the source's memory in both directions.
        /// </summary>
        [TestMethod]
        public void ViewOfAnExportedValue_OwningTheValue_DisposesTheHandle_AndStillSharesTheSource()
        {
            using NDArray nd = Arange(NPTypeCode.Int32, 4);
            OrtTensor h = nd.AsOrtValue();
            using NDArray view = h.Value.AsNDArray(ownsValue: true);

            h.IsDisposed.Should().BeTrue("ownership of the value went to the view, which needs neither the value nor the handle");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            view.ToArray<int>().Should().Equal(0, 1, 2, 3);
            view[1] = 42;
            nd.GetInt32(1).Should().Be(42, "the view is the source's memory");
            nd[2] = -5;
            view.GetInt32(2).Should().Be(-5);
        }

        /// <summary>
        ///     A value whose handle holds a COPY (a transposed array under the default mode): the owning view keeps the
        ///     temporary's buffer alive past the handle, reads the logical elements, and gives the buffer back when it is
        ///     disposed (the whole round trip balances to zero).
        /// </summary>
        [TestMethod]
        public void ViewOfACopyHandlesValue_KeepsTheTemporaryAlive_PastTheHandle()
        {
            long balance = Pool.Balance(() =>
            {
                using NDArray m = Arange(NPTypeCode.Single, 2, 3);
                using NDArray transposed = m.T;                       // (3,2): [[0,3],[1,4],[2,5]]
                OrtTensor h = transposed.AsOrtValue();
                h.IsView.Should().BeFalse();
                NDArray view = h.Value.AsNDArray(ownsValue: true);
                h.IsDisposed.Should().BeTrue();
                using (NDArray other = np.full(new Shape(6), 777f))   // would reuse the temporary if it had been freed
                    view.ToArray<float>().Should().Equal(0f, 3f, 1f, 4f, 2f, 5f);
                view.shape.Should().Equal(3, 2);
                view.flags.writeable.Should().BeTrue("the temporary is the handle's own writeable copy");
                view.Dispose();
            });
            balance.Should().Be(0, "the view released the temporary it kept alive");
        }

        /// <summary>
        ///     A read-only array stays read-only through a round trip. Both handle kinds used to hand back a WRITEABLE view
        ///     of it, so <c>ro.AsOrtValue().Value.AsNDArray()[0] = x</c> wrote memory NumSharp had marked read-only. A
        ///     writeable source still gives a writeable view.
        /// </summary>
        [TestMethod]
        public void ViewsOfAnExport_InheritTheSourcesReadOnlyFlag_ThroughBothHandleKinds()
        {
            using NDArray ro = Arange(NPTypeCode.Single, 2, 3);
            ro.setflags(write: false);
            using (OrtTensor h = ro.AsOrtValue())
            using (NDArray view = h.Value.AsNDArray())
            {
                h.IsView.Should().BeTrue("a read-only array is still shared: ORT reads inputs");
                view.flags.writeable.Should().BeFalse("OrtValue route");
                new Action(() => view[0, 0] = 9f).Should().Throw<Exception>().WithMessage("*read-only*");
            }
            using (OrtTensor<float> h = ro.AsDenseTensor<float>())
            using (NDArray view = h.Tensor.AsNDArray())
            {
                view.flags.writeable.Should().BeFalse("DenseTensor route");
                new Action(() => view[0, 0] = 9f).Should().Throw<Exception>().WithMessage("*read-only*");
            }
            ro.GetSingle(0, 0).Should().Be(0f, "nothing was written");

            using NDArray rw = Arange(NPTypeCode.Single, 2, 3);
            using (OrtTensor h = rw.AsOrtValue())
            using (NDArray view = h.Value.AsNDArray())
                view.flags.writeable.Should().BeTrue("a writeable source gives a writeable view");
            using (OrtTensor<float> h = rw.AsDenseTensor<float>())
            using (NDArray view = h.Tensor.AsNDArray())
                view.flags.writeable.Should().BeTrue();
        }

        /// <summary>
        ///     A Char array exports as UInt16, so the view of its value is a UInt16 view of the SAME buffer (a
        ///     reinterpreting alias), not a copy: a write through it changes the characters.
        /// </summary>
        [TestMethod]
        public void ViewOfAnExportedCharValue_IsAUInt16AliasOfTheSameBuffer()
        {
            using NDArray chars = np.array("abc".ToCharArray());
            using OrtTensor h = chars.AsOrtValue();
            using NDArray view = h.Value.AsNDArray();
            view.typecode.Should().Be(NPTypeCode.UInt16, "an ORT tensor carries no 'these are characters' bit");
            view.ToArray<ushort>().Should().Equal((ushort)'a', (ushort)'b', (ushort)'c');
            view[1] = (ushort)'Z';
            chars.GetChar(1).Should().Be('Z', "the view shares the Char buffer");
        }

        /// <summary>
        ///     <see cref="OnnxCodecMode.Copy"/> on the import verbs returns an independent array; with
        ///     <c>ownsValue: true</c> the value is disposed once the copy exists (through its handle when the package made
        ///     it). <see cref="OnnxCodecMode.View"/> is the default's behaviour, and an undefined mode is refused.
        /// </summary>
        [TestMethod]
        public void ImportModes_CopyIsIndependent_AndOwningACopyDisposesTheValue()
        {
            using NDArray src = Arange(NPTypeCode.Double, 4);

            // a value ORT owns
            using (OrtValue v = src.ToOrtValue())
            using (NDArray copy = v.AsNDArray(mode: OnnxCodecMode.Copy))
            {
                copy.flags.owndata.Should().BeTrue("a copy owns its buffer");
                NDArrayOnnxInterop.LiveImports.Should().Be(0, "a copy is not a lease");
                copy[0] = 99.0;
                v.GetTensorDataAsSpan<double>()[0].Should().Be(0.0, "the copy shares nothing with ORT's buffer");
                using NDArray view = v.AsNDArray(mode: OnnxCodecMode.View);
                view.flags.owndata.Should().BeFalse();
                view[1] = -1.0;
                v.GetTensorDataAsSpan<double>()[1].Should().Be(-1.0, "View shares ORT's buffer, like the default");
            }

            // owning + Copy: the value is disposed, the copy lives on
            long pressure = NDArrayOnnxInterop.ImportPressureBytes;
            OrtValue owned = src.ToOrtValue();
            using (NDArray copy = owned.AsNDArray(ownsValue: true, mode: OnnxCodecMode.Copy))
            {
                copy.ToArray<double>().Should().Equal(0.0, 1, 2, 3);
                NDArrayOnnxInterop.LiveImports.Should().Be(0);
                NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure, "a copy reports no pressure: NumSharp's allocator accounts for it");
            }

            // owning + Copy on a value the package made: its handle is disposed, not left to the finalizer
            OrtTensor h = src.AsOrtValue();
            using (NDArray copy = h.Value.AsNDArray(ownsValue: true, mode: OnnxCodecMode.Copy))
            {
                h.IsDisposed.Should().BeTrue();
                copy[0] = 7.0;
                src.GetDouble(0).Should().Be(0.0, "a copy of an exported value does not share the source");
            }
            src.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();

            // the DenseTensor twin
            var tensor = new DenseTensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
            using (NDArray copy = tensor.AsNDArray(OnnxCodecMode.Copy))
            {
                copy[0] = 50;
                tensor[0].Should().Be(1, "Copy does not pin or share the tensor's buffer");
                NDArrayOnnxInterop.LiveImports.Should().Be(0);
            }

            using OrtValue any = src.ToOrtValue();
            new Action(() => any.AsNDArray(mode: (OnnxCodecMode)9)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => tensor.AsNDArray((OnnxCodecMode)9)).Should().Throw<ArgumentOutOfRangeException>();
        }

        /// <summary>
        ///     Only a view that OWNS its ORT buffer reports the buffer to the GC, once, and takes the report back when the
        ///     last view over the memory is gone. A non-owning view reports nothing (the caller's OrtValue owns the bytes),
        ///     and neither does a view of a tensor's pinned managed array. Before, every view reported its full size, so
        ///     several views of one output multiplied the pressure.
        /// </summary>
        [TestMethod]
        public void MemoryPressure_IsReportedOnlyByAnOwningView_AndTakenBack()
        {
            using NDArray src = Arange(NPTypeCode.Double, 1000);
            long baseline = NDArrayOnnxInterop.ImportPressureBytes;

            using (OrtValue v = src.ToOrtValue())
            using (NDArray a = v.AsNDArray())
            using (NDArray b = v.AsNDArray())
                NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline, "non-owning views report nothing, however many there are");

            var tensor = new DenseTensor<double>(new double[1000], new[] { 1000 });
            using (NDArray view = tensor.AsNDArray())
                NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline, "a pinned managed array is already counted by the GC");

            OrtValue owned = src.ToOrtValue();
            NDArray owning = owned.AsNDArray(ownsValue: true);
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline + 8000, "the owning view reports its 8,000 bytes");
            NDArray slice = owning["100:200"];
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline + 8000, "a derived slice shares the one lease");
            owning.Dispose();
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline + 8000, "the slice still holds the memory");
            slice.Dispose();
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(baseline, "the last view took the report back");
        }

        /// <summary>
        ///     A refused <c>AsNDArray(ownsValue: true)</c> leaves the value alive and the caller's: ownership transfers only
        ///     with a returned array. Checked for the refusals a caller can reach (a string tensor, a sequence), for both
        ///     modes that can own.
        /// </summary>
        [TestMethod]
        public void RefusedOwningImport_LeavesTheValueAliveAndTheCallers()
        {
            using OrtValue strings = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 2 });
            strings.StringTensorSetElementAt("a".AsSpan(), 0);
            foreach (OnnxCodecMode mode in new[] { OnnxCodecMode.Auto, OnnxCodecMode.Copy })
            {
                new Action(() => strings.AsNDArray(ownsValue: true, mode: mode)).Should().Throw<NotSupportedException>($"{mode}");
                strings.GetStringElement(0).Should().Be("a", $"{mode}: the refused import must not dispose the value");
            }

            using NDArray element = Arange(NPTypeCode.Single, 3);
            using OrtValue tensor = element.ToOrtValue();
            using OrtValue sequence = OrtValue.CreateSequence(new List<OrtValue> { tensor });
            new Action(() => sequence.AsNDArray(ownsValue: true)).Should().Throw<NotSupportedException>();
            sequence.GetValueCount().Should().Be(1, "the refused import must not dispose the sequence");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     The mechanism behind "a failure never disposes the caller's value": a lease that is DISARMED ends without
        ///     running its release action, takes back the pressure it reported and its place in the import count, and
        ///     turns the later <c>Release</c> (the memory block's dispose hook, which already carries the lease) into a
        ///     no-op. A released lease runs its action exactly once.
        /// </summary>
        [TestMethod]
        public void ImportLease_Disarm_EndsTheLeaseWithoutRunningItsAction()
        {
            int imports = NDArrayOnnxInterop.LiveImports;
            long pressure = NDArrayOnnxInterop.ImportPressureBytes;

            int ran = 0;
            var disarmed = new ImportLease(() => ran++, ownedBytes: 4096);
            NDArrayOnnxInterop.LiveImports.Should().Be(imports + 1);
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure + 4096);
            disarmed.Disarm();
            disarmed.IsReleased.Should().BeTrue();
            disarmed.Release();   // what the memory block's hook does later
            disarmed.Disarm();
            ran.Should().Be(0, "a disarmed lease never runs its action");
            NDArrayOnnxInterop.LiveImports.Should().Be(imports);
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure);

            var released = new ImportLease(() => ran++, ownedBytes: 0);
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure, "a lease that owns no bytes reports none");
            released.Release();
            released.Release();
            released.Disarm();
            ran.Should().Be(1, "a released lease runs its action once");
            NDArrayOnnxInterop.LiveImports.Should().Be(imports);
        }

        /// <summary>
        ///     <c>ToNDArray</c> of an exported value copies from the NumSharp buffer itself: the result owns its data, has
        ///     the crossed dtype and the tensor's shape, and shares nothing. Checked for a view handle, for a copy handle
        ///     (whose buffer is the temporary) and for a 0-d value.
        /// </summary>
        [TestMethod]
        public void ToNDArray_OfAnExportedValue_CopiesFromTheBuffer()
        {
            using NDArray m = Arange(NPTypeCode.Int16, 2, 3);
            using (OrtTensor h = m.AsOrtValue())
            using (NDArray copy = h.Value.ToNDArray())
            {
                copy.flags.owndata.Should().BeTrue();
                copy.shape.Should().Equal(2, 3);
                copy.ToArray<short>().Should().Equal((short)0, (short)1, (short)2, (short)3, (short)4, (short)5);
                copy[0, 0] = 9;
                m.GetInt16(0, 0).Should().Be((short)0);
            }

            using NDArray reversed = m["::-1"];
            using (OrtTensor h = reversed.AsOrtValue())   // a copy handle
            using (NDArray copy = h.Value.ToNDArray())
                copy.ToArray<short>().Should().Equal((short)3, (short)4, (short)5, (short)0, (short)1, (short)2);

            using NDArray scalar = NDArray.Scalar(2.5f);
            using (OrtTensor h = scalar.AsOrtValue())
            using (NDArray copy = h.Value.ToNDArray())
            {
                copy.ndim.Should().Be(0);
                copy.GetSingle().Should().Be(2.5f);
            }
        }
    }
}
