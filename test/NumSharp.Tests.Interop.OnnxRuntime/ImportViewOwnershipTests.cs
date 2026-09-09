using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     NumPy-parity OWNERSHIP semantics of a zero-copy import view (<c>AsNDArray</c>) — the residual
    ///     the pythonnet bridge's <c>OwnershipAndWriteabilityTests</c> covers that the ORT suite's
    ///     <see cref="ImportTests"/> does not: a same-size <c>resize</c> is a reshape-in-place that stays
    ///     on ORT's memory (only a size-CHANGING resize refuses, which <see cref="ImportTests"/> already
    ///     pins), a derived slice inherits the non-owning semantics, and a <b>0-d</b> value goes through
    ///     the real lease path (not the empty short-circuit).
    ///
    ///     <para>The WRITEABILITY half of the sibling test has no ORT analog: an ORT output tensor is
    ///     always a writeable native buffer, and there is no read-only ORT-tensor source (the read-only
    ///     case ORT does model — a broadcast/read-only NumSharp array as a pre-allocated OUTPUT — is pinned
    ///     in <see cref="SessionRunTests"/>).</para>
    /// </summary>
    [TestClass]
    public class ImportViewOwnershipTests : OnnxTestBase
    {
        [TestMethod]
        public void ImportedView_SameSizeResize_IsReshapeInPlace_StaysOnOrtsMemory()
        {
            using NDArray src = Arange(NPTypeCode.Double, 6);
            using OrtValue v = src.ToOrtValue();           // an independent ORT-allocated buffer
            using NDArray view = v.AsNDArray();            // non-owning lease over ORT's memory
            NDArrayOnnxInterop.LiveImports.Should().Be(1);

            // numpy parity: a same-total-size resize on a non-owner is a pure reshape — it never
            // reallocates, so the view must stay on ORT's buffer (a size-CHANGING resize refuses,
            // pinned by ImportTests.AsNDArray_DoesNotOwnItsData_ResizeRefuses).
            view.resize(new Shape(2, 3));
            view.ndim.Should().Be(2);
            view.Storage.IsView.Should().BeTrue("the reshape did not detach from ORT's memory");

            view[1, 2] = -7.0;                              // flat index 5
            v.GetTensorDataAsSpan<double>()[5].Should().Be(-7.0, "the reshaped view still writes ORT's buffer");
        }

        [TestMethod]
        public void ImportedView_DerivedSlice_InheritsNonOwningSemantics()
        {
            using NDArray src = Arange(NPTypeCode.Int32, 8);
            using OrtValue v = src.ToOrtValue();
            using NDArray view = v.AsNDArray();
            using NDArray tail = view["4:"];                // a derived slice extends the same lease

            tail.Storage.IsView.Should().BeTrue("a slice of a non-owning view is itself non-owning");
            tail.GetInt32(0).Should().Be(4);
            new Action(() => tail.resize(16)).Should().Throw<Exception>().WithMessage("*does not own its data*",
                "a grow-resize of a derived import slice refuses, exactly like the parent view");
        }

        [TestMethod]
        public void ZeroDimValue_GoesThroughTheRealLeasePath_NotTheEmptyShortCircuit()
        {
            using NDArray scalar = NDArray.Scalar(-7.5);
            OrtValue v = scalar.ToOrtValue();               // a 0-d ORT value: ElementCount == 1, NOT empty
            NDArray view = v.AsNDArray(ownsValue: true);    // count==1 => a genuine lease, not the empty branch
            NDArrayOnnxInterop.LiveImports.Should().Be(1, "a 0-d value is one element to share, so it leases");

            view.ndim.Should().Be(0);
            view.GetDouble().Should().Be(-7.5, "the 0-d view reads ORT's scalar");

            view.Dispose();                                 // owns the value => disposes it and releases the lease
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }
    }
}
