using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     NumPy-parity semantics of zero-copy import views around OWNERSHIP and WRITEABILITY —
    ///     the two places where a NumSharp view over foreign (Python-owned) memory must NOT behave
    ///     like an array that owns its buffer:
    ///
    ///     <para><b>Ownership</b> — numpy arrays built over foreign buffers report
    ///     <c>flags.owndata == False</c> and <c>ndarray.resize</c> refuses to reallocate them
    ///     ("cannot resize this array: it does not own its data"). A NumSharp import view that
    ///     instead claimed ownership would let <c>resize</c> silently REALLOCATE into NumSharp-owned
    ///     memory and detach from Python — subsequent mutations would stop flowing to the exporter
    ///     with no error anywhere.</para>
    ///
    ///     <para><b>Writeability</b> — numpy marks arrays over read-only buffers
    ///     <c>writeable=False</c> and raises "assignment destination is read-only" on writes.
    ///     A NumSharp view over a read-only source (<c>bytes</c>, <c>setflags(write=False)</c>)
    ///     that stayed writeable would corrupt immutable Python objects on the first write.</para>
    /// </summary>
    [TestClass]
    public class OwnershipAndWriteabilityTests : InteropTestBase
    {
        // ------------------------------------------------------------------------------------------
        // ownership: ndarray.resize over Python-owned memory (finding: silent detach)
        // ------------------------------------------------------------------------------------------

        [TestMethod]
        public void ImportedView_GrowResize_RefusesLikeNumPy_AndStaysAttached()
        {
            int i0 = PythonConvert.LiveImports;
            PyExec("rz_src = np.arange(8, dtype='f8')");
            var view = ViewOf("rz_src");

            // numpy: b.resize(12) on a frombuffer view -> ValueError: cannot resize this array:
            // it does not own its data. The NumSharp view over Python memory must refuse the same
            // way — succeeding would REALLOCATE into NumSharp-owned memory and silently detach.
            ((Action)(() => view.resize(new Shape(16))))
                .Should().Throw<IncorrectShapeException>().WithMessage("*does not own its data*");

            // The refusal must leave the view fully attached: same lease, same shared memory.
            PythonConvert.LiveImports.Should().Be(i0 + 1, "the refused resize must not have touched the lease");
            view.size.Should().Be(8, "the refused resize must not have changed the view");

            WriteAt(view, -3.5, 2);
            PyFloat("float(rz_src[2])").Should().BeApproximately(-3.5, 1e-12,
                "writes must still flow to Python — a silently detached view is the failure this test hunts");

            PyExec("rz_src[5] = 55.5");
            ReadAt<double>(view, 5).Should().BeApproximately(55.5, 1e-12, "and Python writes must still be visible");

            // refcheck:false bypasses the SHARING check, not the ownership check (numpy behaves the same).
            ((Action)(() => view.resize(new Shape(16), refcheck: false)))
                .Should().Throw<IncorrectShapeException>().WithMessage("*does not own its data*");

            view.Dispose();
            WaitFor(() => PythonConvert.LiveImports == i0).Should().BeTrue();
        }

        [TestMethod]
        public void ImportedStridedView_GrowResize_AlsoRefuses()
        {
            PyExec("rzs_src = np.arange(20, dtype='i8')");
            var strided = ViewOf("rzs_src[::2]");   // __array_interface__ route

            // numpy's own error for a STRIDED view (probed): the single-segment gate fires before
            // the ownership gate — `base[::2].resize(32)` raises exactly this message.
            ((Action)(() => strided.resize(new Shape(32))))
                .Should().Throw<IncorrectShapeException>().WithMessage("*single-segment*");

            WriteAt(strided, -9L, 1);
            PyLong("int(rzs_src[2])").Should().Be(-9, "the strided view must stay attached after the refusal");
        }

        [TestMethod]
        public void ImportedView_SameSizeResize_ReshapesInPlace_StillShared()
        {
            // numpy parity (probed): same-total-size resize is legal on a non-owner — it is a pure
            // reshape and never reallocates, so the view must stay on Python's memory.
            PyExec("rq_src = np.arange(6, dtype='f8')");
            var view = ViewOf("rq_src");

            view.resize(new Shape(2, 3));
            view.ndim.Should().Be(2);

            WriteAt(view, -7.0, 1, 2);   // flat index 5
            PyFloat("float(rq_src[5])").Should().BeApproximately(-7.0, 1e-12,
                "a same-size resize is a reshape in place — the memory must still be Python's");
        }

        [TestMethod]
        public void ImportedView_ReportsViewSemantics_NotOwnership()
        {
            PyExec("own_src = np.arange(6, dtype='f8')");
            var view = ViewOf("own_src");

            // numpy: np.frombuffer(...).flags.owndata is False and .base is the exporter.
            // The NumSharp equivalent of that contract:
            view.Storage.IsView.Should().BeTrue("a view over Python-owned memory must not claim to own its data");

            // Public observable of the same fact: np.require(..., "O") (OWNDATA) must produce a COPY
            // for a non-owning array, exactly as numpy does for frombuffer views.
            var owned = np.require(view, (Type)null, "O");
            unsafe
            {
                ((IntPtr)owned.Storage.Address).Should().NotBe((IntPtr)view.Storage.Address,
                    "require('O') on a non-owning view must copy into NumSharp-owned memory");
            }

            WriteAt(owned, -1.0, 0);
            PyFloat("float(own_src[0])").Should().BeApproximately(0.0, 1e-12, "the required copy must be detached");
            ReadAt<double>(view, 0).Should().BeApproximately(0.0, 1e-12, "and the original view untouched");
        }

        // ------------------------------------------------------------------------------------------
        // writeability: read-only sources with allowReadonly:true (finding: writable view over
        // immutable Python memory)
        // ------------------------------------------------------------------------------------------

        [TestMethod]
        public void ReadonlyNumpy_OptIn_ViewIsNonWriteable_AndWritesThrow()
        {
            PyExec("ro_np = np.arange(5, dtype='f8'); ro_np.setflags(write=False)");
            var view = ViewOf("ro_np", allowReadonly: true);

            view.Shape.IsWriteable.Should().BeFalse(
                "a view over a read-only exporter must carry numpy's writeable=False, not a caller promise");

            // Guarded write paths must raise numpy's exact refusal instead of corrupting the source.
            ((Action)(() => np.copyto(view, np.zeros(new Shape(5)))))
                .Should().Throw<NumSharpException>().WithMessage("*read-only*");
            ((Action)(() => view["1:3"] = np.zeros(new Shape(2))))
                .Should().Throw<NumSharpException>().WithMessage("*read-only*");

            PyStr("ro_np.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0, 4.0]",
                "the refused writes must have left the read-only Python data untouched");

            // Derived views inherit the protection, exactly like slicing a read-only numpy array.
            var derived = view["1:"];
            derived.Shape.IsWriteable.Should().BeFalse("derived views of a read-only import stay read-only");

            // And the flag round-trips through a re-export.
            ExportTo("ro_back", view);
            PyBool("ro_back.flags.writeable").Should().BeFalse("read-only must survive the round trip to numpy");
            PyBool("np.shares_memory(ro_back, ro_np)").Should().BeTrue("still the same single buffer");
        }

        [TestMethod]
        public void ReadonlyBytes_OptIn_ViewIsNonWriteable()
        {
            // bytes are the canonical immutable exporter: CPython interns/caches them, so a write
            // through a shared view could poison every other user of the same object.
            var view = ViewOf("b'abcd'", allowReadonly: true);

            view.Shape.IsWriteable.Should().BeFalse("bytes are immutable; the view must be read-only");
            ((Action)(() => np.copyto(view, np.zeros(new Shape(4), typeof(byte)))))
                .Should().Throw<NumSharpException>().WithMessage("*read-only*");

            ReadAt<byte>(view, 0).Should().Be(97, "reading stays fully functional");
        }

        [TestMethod]
        public void ReadonlyStridedNumpy_OptIn_ArrayInterfaceRoute_IsNonWriteable()
        {
            PyExec("ro_base = np.arange(10, dtype='i8')\nro_view = ro_base[::2]\nro_view.setflags(write=False)");
            var view = ViewOf("ro_view", allowReadonly: true);   // non-contiguous -> __array_interface__ route

            view.shape[0].Should().Be(5);
            ReadAt<long>(view, 2).Should().Be(4, "strided layout must be intact");
            view.Shape.IsWriteable.Should().BeFalse("the interface reports data=(ptr, readonly=True)");

            ((Action)(() => np.copyto(view, np.zeros(new Shape(5), typeof(long)))))
                .Should().Throw<NumSharpException>().WithMessage("*read-only*");
            PyStr("ro_base.tolist()").Should().Be("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "base data untouched");
        }

        [TestMethod]
        public void WritableSource_WithOptIn_StaysFullyWriteable()
        {
            // allowReadonly is an OPT-IN for read-only sources, not a downgrade for writable ones.
            PyExec("rw_src = np.arange(4, dtype='f8')");
            var view = ViewOf("rw_src", allowReadonly: true);

            view.Shape.IsWriteable.Should().BeTrue("a writable exporter keeps a writable lease");
            np.copyto(view, np.arange(4).astype(NPTypeCode.Double) * 2);
            PyStr("rw_src.tolist()").Should().Be("[0.0, 2.0, 4.0, 6.0]", "writes flow to Python as before");
        }

        // ------------------------------------------------------------------------------------------
        // coverage gaps found by the audit: 0-d zero-copy views, the AsNDArray extension
        // ------------------------------------------------------------------------------------------

        [TestMethod]
        public void View_ZeroDim_IsLeased_AndAliasesTheScalar()
        {
            int i0 = PythonConvert.LiveImports;
            PyExec("zd = np.array(3.25)");   // 0-d, writable ndarray

            var view = ViewOf("zd");
            view.ndim.Should().Be(0);
            view.size.Should().Be(1);
            PythonConvert.LiveImports.Should().Be(i0 + 1, "a 0-d view is a real lease, not a copy");

            ReadAt<double>(view).Should().BeApproximately(3.25, 1e-12);
            WriteAt(view, 9.5);
            PyFloat("float(zd)").Should().BeApproximately(9.5, 1e-12, "0-d writes must land in Python's scalar");

            view.Dispose();
            WaitFor(() => PythonConvert.LiveImports == i0).Should().BeTrue("disposing the 0-d view releases its lease");
        }

        [TestMethod]
        public void AsNDArray_Extension_IsTheZeroCopyVerb()
        {
            int i0 = PythonConvert.LiveImports;
            PyExec("ext_src = np.arange(6, dtype='f8')");

            NDArray view;
            using (Gil())
            {
                using PyObject src = Scope.Get("ext_src");
                view = src.AsNDArray();   // the extension: As... = share, like numpy asarray
            }

            PythonConvert.LiveImports.Should().Be(i0 + 1, "AsNDArray must lease, not copy");
            WriteAt(view, -2.5, 4);
            PyFloat("float(ext_src[4])").Should().BeApproximately(-2.5, 1e-12, "the extension view aliases Python memory");

            view.Dispose();
            WaitFor(() => PythonConvert.LiveImports == i0).Should().BeTrue();
        }
    }
}
