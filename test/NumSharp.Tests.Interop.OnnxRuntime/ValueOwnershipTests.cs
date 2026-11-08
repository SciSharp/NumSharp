using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Who keeps an <see cref="OrtValue"/> alive, and who disposes it. The bridge promises three things that no
    ///     counter of the suite observes, because a value that is only dropped is freed later by ONNX Runtime's own
    ///     finalizer and every balance still closes:
    ///     <para>- a NON-owning view roots the value it reads, so a collection cannot free the buffer under the view;</para>
    ///     <para>- an OWNING view disposes its value when the last NumSharp view over the memory is gone;</para>
    ///     <para>- the copy form of <c>Run</c> disposes the values ONNX Runtime returned, the zero-copy form hands each
    ///     value to its view.</para>
    ///     <para>The review's mutation run planted a bug for each (a lease that roots nothing, an owning lease that
    ///     disposes nothing, a copy path that disposes nothing) and all three passed the suite. These tests read ORT's
    ///     own record of a value's disposal (<see cref="OrtProbe"/>) and the value's reachability.</para>
    /// </summary>
    [TestClass]
    public class ValueOwnershipTests : OnnxTestBase
    {
        /// <summary>A float64 tensor of <paramref name="elements"/> elements in ONNX Runtime's own memory, holding <c>start, start + 1, …</c>.</summary>
        /// <param name="elements">The element count.</param>
        /// <param name="start">The first value.</param>
        /// <returns>A value the caller owns.</returns>
        private static OrtValue OrtOwnedRamp(int elements, double start)
        {
            OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Double, new long[] { elements });
            Span<double> span = value.GetTensorMutableDataAsSpan<double>();
            for (int i = 0; i < elements; i++)
                span[i] = start + i;
            return value;
        }

        /// <summary>
        ///     A non-owning view is the only thing that references its value here: the value survives full collections
        ///     for as long as the view lives, the view reads the right elements afterwards, and once the view is
        ///     disposed the value can be collected.
        /// </summary>
        [TestMethod]
        public void NonOwningView_RootsItsValue_AndLetsItGoWithTheView()
        {
            WeakReference value = ReadThroughAViewOfAValueNobodyElseHolds();
            GcQuiescence.WaitCollected(value).Should().BeTrue("once the view is disposed, nothing roots the value");
        }

        /// <summary>
        ///     The body of <see cref="NonOwningView_RootsItsValue_AndLetsItGoWithTheView"/> up to the view's disposal, in a
        ///     frame of its own: a debug-build JIT keeps a frame's references alive until the method returns, and the
        ///     test must not hold the view when it waits for the value's collection.
        /// </summary>
        /// <returns>A weak reference to the value the view read.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static WeakReference ReadThroughAViewOfAValueNobodyElseHolds()
        {
            (NDArray view, WeakReference value) = ViewOfAValueNobodyElseHolds();
            try
            {
                GcQuiescence.CollectFull();
                GcQuiescence.CollectFull();
                // Checked BEFORE the view is read: with a lease that roots nothing, the value is finalized here and
                // the read below would be of memory ONNX Runtime has freed.
                value.IsAlive.Should().BeTrue("a view must root the OrtValue whose buffer it reads");
                view.ToArray<double>().Should().Equal(10, 11, 12, 13);
                NDArrayOnnxInterop.LiveImports.Should().Be(1);
            }
            finally
            {
                view.Dispose();
            }

            return value;
        }

        /// <summary>Makes a value, views it without owning it, and returns the view with a weak reference to the value: this frame is the last that holds the value itself.</summary>
        /// <returns>The view and the weak reference.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static (NDArray, WeakReference) ViewOfAValueNobodyElseHolds()
        {
            OrtValue value = OrtOwnedRamp(4, 10);
            return (value.AsNDArray(), new WeakReference(value));
        }

        /// <summary>
        ///     An owning view disposes its value when the LAST NumSharp view over the memory is disposed, not before:
        ///     a slice of the view keeps the buffer after the view itself is gone. An empty tensor has nothing to share,
        ///     so its owning import disposes the value at once.
        /// </summary>
        [TestMethod]
        public void OwningView_DisposesItsValue_WithTheLastViewOverTheMemory()
        {
            OrtValue value = OrtOwnedRamp(4, 10);
            NDArray view = value.AsNDArray(ownsValue: true);
            NDArray tail = view["1:"];
            OrtProbe.IsDisposed(value).Should().BeFalse();

            view.Dispose();
            OrtProbe.IsDisposed(value).Should().BeFalse("a slice of the view still reads the buffer");
            tail.ToArray<double>().Should().Equal(11, 12, 13);

            tail.Dispose();
            OrtProbe.IsDisposed(value).Should().BeTrue("the last view over the memory disposes the value it owns");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);

            OrtValue empty = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Double, new long[] { 0 });
            using (NDArray none = empty.AsNDArray(ownsValue: true))
            {
                none.size.Should().Be(0);
                OrtProbe.IsDisposed(empty).Should().BeTrue("nothing is shared with an empty tensor, so owning it means disposing it");
            }
        }

        /// <summary>
        ///     What the two output collectors of <c>Run</c> do with the values ONNX Runtime returned. The copy form
        ///     disposes every one (the copies need nothing from them). The zero-copy form disposes none: each view
        ///     owns its value and disposes it when the view is disposed.
        /// </summary>
        [TestMethod]
        public void RunOutputs_CopiedOnesAreDisposed_ViewedOnesAreOwnedByTheirViews()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 4);
            using OrtTensor handle = x.AsOrtValue();
            using var options = new RunOptions();
            string[] inNames = { "X" };
            string[] outNames = { "Y1", "Y2" };
            OrtTensor[] inHandles = { handle };
            OrtValue[] inValues = { handle.Value };

            // ---- the copy form
            IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, inNames, inValues, outNames);
            OrtValue[] held = outputs.ToArray();   // the collection empties itself when it is disposed
            held.Should().OnlyContain(v => !OrtProbe.IsDisposed(v));
            IReadOnlyDictionary<string, NDArray> copies = InferenceSessionExtensions.CopyOutputs(outputs, outNames, inNames, inHandles);
            using (NDArray y1 = copies["Y1"])
            using (NDArray y2 = copies["Y2"])
            {
                held.Should().OnlyContain(v => OrtProbe.IsDisposed(v), "a copy needs nothing from the value ONNX Runtime returned");
                y1.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f);
                y2.ToArray<float>().Should().Equal(-0f, -1f, -2f, -3f);
                y1.flags.owndata.Should().BeTrue();
            }

            // ---- the zero-copy form
            outputs = session.Run(options, inNames, inValues, outNames);
            held = outputs.ToArray();
            IReadOnlyDictionary<string, NDArray> views = InferenceSessionExtensions.ViewOutputs(outputs, outNames, inNames, inHandles);
            held.Should().OnlyContain(v => !OrtProbe.IsDisposed(v), "each view owns its value: nothing is disposed while the views live");
            NDArrayOnnxInterop.LiveImports.Should().Be(2);
            views["Y2"].ToArray<float>().Should().Equal(-0f, -1f, -2f, -3f);

            views["Y1"].Dispose();
            OrtProbe.IsDisposed(held[0]).Should().BeTrue("the view of Y1 disposed the value it owned");
            OrtProbe.IsDisposed(held[1]).Should().BeFalse("the value of Y2 belongs to another view");
            views["Y2"].Dispose();
            OrtProbe.IsDisposed(held[1]).Should().BeTrue();
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     A caller can keep the value of an export handle past the handle (<c>v = h.Value; h.Dispose();</c>). ONNX
        ///     Runtime has released that value and does not guard a released value: the first native call on it is an
        ///     access violation that ends the process, which is what every import verb did with it. The bridge now
        ///     remembers the values of its disposed handles and refuses them with
        ///     <see cref="ObjectDisposedException"/>, in every verb and mode, for a numeric handle and for a text one.
        /// </summary>
        [TestMethod]
        public void ValueOfADisposedHandle_IsRefused_ByEveryImportVerb_InsteadOfCrashingInOnnxRuntime()
        {
            using NDArray a = Arange(NPTypeCode.Double, 4);
            OrtTensor handle = a.AsOrtValue();
            OrtValue value = handle.Value;
            handle.Dispose();

            new Action(() => value.AsNDArray()).Should().Throw<ObjectDisposedException>().WithMessage("AsNDArray: this OrtValue belongs to an OrtTensor handle that has been disposed*");
            new Action(() => value.AsNDArray(ownsValue: true)).Should().Throw<ObjectDisposedException>();
            new Action(() => value.AsNDArray(mode: OnnxCodecMode.View)).Should().Throw<ObjectDisposedException>();
            new Action(() => value.AsNDArray(mode: OnnxCodecMode.Copy)).Should().Throw<ObjectDisposedException>();
            new Action(() => value.ToNDArray()).Should().Throw<ObjectDisposedException>().WithMessage("ToNDArray:*disposed*");
            new Action(() => value.ToNDArrays()).Should().Throw<ObjectDisposedException>().WithMessage("ToNDArrays:*");
            new Action(() => value.ToMap()).Should().Throw<ObjectDisposedException>().WithMessage("ToMap:*");
            new Action(() => value.ToMaps()).Should().Throw<ObjectDisposedException>().WithMessage("ToMaps:*");
            new Action(() => value.GetMapKeys()).Should().Throw<ObjectDisposedException>().WithMessage("GetMapKeys:*");
            new Action(() => value.ReadStringTensor()).Should().Throw<ObjectDisposedException>().WithMessage("ReadStringTensor:*");
            a.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("a refused import takes no reference");

            // a text handle: its value is a string tensor ONNX Runtime owns, released with the handle all the same
            using NDArray text = np.array(new[] { "a", "bb" });
            OrtTensor textHandle = text.AsOrtValue();
            OrtValue textValue = textHandle.Value;
            textValue.ReadStringTensor().Should().Equal("a", "bb");
            textHandle.Dispose();
            new Action(() => textValue.ReadStringTensor()).Should().Throw<ObjectDisposedException>();
            new Action(() => textValue.ToNDArray()).Should().Throw<ObjectDisposedException>();
            new Action(() => textValue.AsNDArray()).Should().Throw<ObjectDisposedException>();

            // a handle that is disposed two times, and a value of ONNX Runtime's own, are not affected
            handle.Dispose();
            using (OrtValue own = OrtOwnedRamp(2, 5))
            using (NDArray copy = own.ToNDArray())
                copy.ToArray<double>().Should().Equal(5, 6);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The internal contract the <c>Run</c> tier relies on when it hands a temporary to the export core: with
        ///     <c>ownsSource</c> the handle owns what it is given. When the core then copies again (the copy mode), the
        ///     temporary it was given is superseded and disposed at once, and the handle owns the copy.
        /// </summary>
        [TestMethod]
        public void ExportCore_OwningItsSource_DisposesATemporaryItSupersedes()
        {
            NDArray temporary = Arange(NPTypeCode.Single, 3);
            using (OrtTensor handle = NDArrayOnnxInterop.AsOrtValueCore(temporary, ownsSource: true, OnnxCodecMode.Copy))
            {
                temporary.IsDisposed.Should().BeTrue("the copy the handle made replaces the temporary it was given");
                handle.IsView.Should().BeFalse();
                handle.Source.Should().NotBeSameAs(temporary);
                handle.Value.GetTensorDataAsSpan<float>().ToArray().Should().Equal(0f, 1f, 2f);
            }

            NDArray shared = Arange(NPTypeCode.Single, 3);
            using (OrtTensor handle = NDArrayOnnxInterop.AsOrtValueCore(shared, ownsSource: true, OnnxCodecMode.View))
            {
                shared.IsDisposed.Should().BeFalse("a view reads the array itself, so the handle keeps it");
                handle.Source.Should().BeSameAs(shared);
            }

            shared.IsDisposed.Should().BeTrue("the handle owned its source and disposed it with itself");

            // the same contract for the legacy tensor handle
            NDArray denseTemporary = Arange(NPTypeCode.Single, 3);
            OrtTensor<float> dense = NDArrayOnnxInterop.AsDenseTensorCore<float>(denseTemporary, ownsSource: true, OnnxCodecMode.Copy);
            denseTemporary.IsDisposed.Should().BeTrue();
            dense.IsView.Should().BeFalse();
            dense.IsDisposed.Should().BeFalse();
            dense.Tensor.ToArray().Should().Equal(0f, 1f, 2f);
            dense.Dispose();
            dense.IsDisposed.Should().BeTrue();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
