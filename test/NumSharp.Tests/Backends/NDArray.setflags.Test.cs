using System;
using System.IO;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     <see cref="NDArray.setflags"/> — the port of NumPy's <c>ndarray.setflags(write, align, uic)</c>
    ///     (<c>array_setflags</c>, <c>methods.c</c>) — and the <see cref="NDArrayFlags"/> setters that
    ///     route through it (<c>flags.writeable=</c> / <c>flags["W"/"A"/"X"]=</c>, exactly as NumPy's
    ///     <c>arrayflags</c> setters call <c>arr.setflags(…)</c>). Every behavior probed against
    ///     NumPy 2.4.2, error messages verbatim.
    /// </summary>
    [TestClass]
    public class NDArray_setflags_Test
    {
        // ---- basic / owned ---------------------------------------------------------------

        [TestMethod]
        public void NoArgs_IsNoOp()
        {
            var a = np.arange(6);
            a.setflags();
            a.flags.writeable.Should().BeTrue();
            a.flags.aligned.Should().BeTrue();
        }

        [TestMethod]
        public void Write_RoundTrip_OnOwnedArray()
        {
            var a = np.arange(6);
            a.setflags(write: false);
            a.flags.writeable.Should().BeFalse();
            ((Action)(() => a[0] = (NDArray)9L)).Should().Throw<NumSharpException>().WithMessage("*read-only*");
            a.setflags(write: true);
            a.flags.writeable.Should().BeTrue();
            a[0] = (NDArray)9L;
            a.GetInt64(0).Should().Be(9);
        }

        [TestMethod]
        public void ZeroD_View_RoundTrips()
        {
            // NumPy: a 0-d view of a writeable base clears and re-enables fine (probed).
            var m = np.arange(6);
            var zv = m[":1"].reshape(new Shape());
            zv.setflags(write: false);
            zv.flags.writeable.Should().BeFalse();
            zv.setflags(write: true);
            zv.flags.writeable.Should().BeTrue();
        }

        // ---- write=True on views: NumPy's _IsWriteable rule --------------------------------

        [TestMethod]
        public void Write_True_OnViewOfWriteableBase_Allowed()
        {
            // NumPy permits re-enabling a view whose base is writeable (probed: reshape view).
            var b = np.arange(6);
            var v = b.reshape(2, 3);
            v.flags.writeable = false;
            v.setflags(write: true);
            v.flags.writeable.Should().BeTrue();
        }

        [TestMethod]
        public void Write_True_OnBroadcastToView_Allowed_AndWritesAlias()
        {
            // Probed on NumPy 2.4.2: broadcast_to views CAN be made writeable (base is writeable);
            // a write then aliases across the stride-0 axis AND reaches the source array.
            var src = np.arange(3);
            var bc = np.broadcast_to(src, new Shape(4, 3));
            bc.Shape.IsWriteable.Should().BeFalse("broadcast_to is born read-only");

            bc.setflags(write: true);
            bc.flags.writeable.Should().BeTrue();
            bc.Shape.IsBroadcasted.Should().BeTrue("still a stride-0 broadcast view");

            bc[0, 0] = (NDArray)99L;
            src.GetInt64(0).Should().Be(99, "the write reaches the source");
            bc.GetInt64(3, 0).Should().Be(99, "the write aliases across the broadcast axis");
        }

        [TestMethod]
        public void Write_True_OnBroadcast_Toggle_Repeats()
        {
            var bt = np.broadcast_to(np.arange(3), new Shape(2, 3));
            bt.setflags(write: true);
            bt.setflags(write: false);
            bt.setflags(write: true);
            bt.flags.writeable.Should().BeTrue();
        }

        [TestMethod]
        public void Write_True_OnViewOfReadOnlyOwner_Refused()
        {
            var c = np.arange(6);
            c.setflags(write: false);
            var v = c["1:"];
            v.flags.writeable.Should().BeFalse("a view taken after the owner went read-only inherits it");
            ((Action)(() => v.setflags(write: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
            v.flags.writeable.Should().BeFalse();
        }

        [TestMethod]
        public void Write_True_ChecksUnconditionally_EvenWhenViewStillWriteable()
        {
            // Probed: a view created BEFORE the base went read-only is still writeable, yet
            // setflags(write=True) on it STILL refuses (the _IsWriteable check always runs) —
            // and the view stays writeable afterwards (entry-state rollback).
            var k = np.arange(6);
            var kv = k["1:"];
            k.setflags(write: false);
            kv.flags.writeable.Should().BeTrue("pre-existing views are unaffected by the owner's clear");

            ((Action)(() => kv.setflags(write: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
            kv.flags.writeable.Should().BeTrue("the refusal restores the entry state");
        }

        // ---- write=True on foreign read-only memory ---------------------------------------

        [TestMethod]
        public void Write_True_OnFrombufferReadonly_Refused()
        {
            // NumPy: np.frombuffer(b"...").setflags(write=True) → ValueError (read-only exporter).
            var src = np.arange(4).astype(np.uint8);
            src.flags.writeable = false;
            var fb = np.frombuffer(src.data, typeof(byte));
            fb.flags.writeable.Should().BeFalse();
            ((Action)(() => fb.setflags(write: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
        }

        [TestMethod]
        public void Write_True_OnFrombufferWritable_RoundTrips()
        {
            // NumPy: a writable buffer (bytearray) → frombuffer result round-trips write=False/True.
            var fb = np.frombuffer(np.arange(4).astype(np.uint8).data, typeof(byte));
            fb.flags.writeable.Should().BeTrue();
            fb.setflags(write: false);
            fb.setflags(write: true);
            fb.flags.writeable.Should().BeTrue();
        }

        [TestMethod]
        public void Write_True_OnMemmapR_Refused_OwnerAndView()
        {
            string dir = Path.Combine(Path.GetTempPath(), "ns_setflags_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string p = Path.Combine(dir, "a.npy");
                np.save(p, np.arange(5.0));

                var m = (NDArray)np.load(p, mmap_mode: "r");
                ((Action)(() => m.setflags(write: true)))
                    .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
                m.flags.writeable.Should().BeFalse();

                var v = m["1:"];
                ((Action)(() => v.setflags(write: true)))
                    .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");

                v.Dispose();
                m.Dispose();
            }
            finally
            {
                FullGc();
                try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
            }
        }

        [TestMethod]
        public void Write_True_OnMemmapR_FortranOrder_Refused()
        {
            // The F-order memmap is a transpose VIEW over the wrap storage — the read-only protection
            // must sit on the wrap storage itself, or the base-chain rule would re-enable a write into
            // PROT_READ pages.
            string dir = Path.Combine(Path.GetTempPath(), "ns_setflags_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string p = Path.Combine(dir, "f.npy");
                np.save(p, np.asfortranarray(np.arange(6.0).reshape(2, 3)));

                var m = (NDArray)np.load(p, mmap_mode: "r");
                m.flags.writeable.Should().BeFalse();
                ((Action)(() => m.setflags(write: true)))
                    .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
                m.Dispose();
            }
            finally
            {
                FullGc();
                try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
            }
        }

        [TestMethod]
        public void Write_True_OnMemmapRPlus_RoundTrips()
        {
            // NumPy: an 'r+' memmap's buffer is writable, so write=False → write=True round-trips.
            string dir = Path.Combine(Path.GetTempPath(), "ns_setflags_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string p = Path.Combine(dir, "b.npy");
                np.save(p, np.arange(5.0));

                var m = (NDArray)np.load(p, mmap_mode: "r+");
                m.flags.writeable = false;
                m.setflags(write: true);
                m.flags.writeable.Should().BeTrue();
                m.Dispose();
            }
            finally
            {
                FullGc();
                try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
            }
        }

        private static void FullGc()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        // ---- align ------------------------------------------------------------------------

        [TestMethod]
        public void Align_False_IsObservable_AndTrueRestores()
        {
            // Probed: aligned=False, num drops the 0x100 bit (1287→1031 on an owned 1-D),
            // behaved/carray follow; align=True restores.
            var d = np.arange(6);
            d.setflags(align: false);
            d.flags.aligned.Should().BeFalse();
            d.flags.num.Should().Be(1031);
            d.flags.behaved.Should().BeFalse();
            d.flags.carray.Should().BeFalse();
            d.flags.farray.Should().BeFalse("1-D is C-contiguous, and farray requires !C");

            d.setflags(align: true);
            d.flags.aligned.Should().BeTrue();
            d.flags.num.Should().Be(1287);
        }

        [TestMethod]
        public void Align_False_FContiguous_MatchesNumpyNumAndFarray()
        {
            // Probed: F-order 2-D with align cleared → num=1030 (F|O|W), farray stays True
            // (NumPy's any-bit farray getter: WRITEABLE alone satisfies the left side).
            var j = np.asfortranarray(np.arange(12).reshape(3, 4));
            j.setflags(align: false);
            j.flags.num.Should().Be(1030);
            j.flags.farray.Should().BeTrue();
            j.flags.fnc.Should().BeTrue();
        }

        [TestMethod]
        public void Align_False_ToString_ShowsFalse()
        {
            var d = np.arange(6);
            d.setflags(align: false);
            d.flags.ToString().Should().Contain("  ALIGNED : False\n");
        }

        [TestMethod]
        public void Align_False_FreshViewsAndCopies_ReportAligned()
        {
            // Probed: NumPy recomputes ALIGNED per new array object — views and copies of an
            // align-cleared array come back aligned; only the array it was cleared on reports False.
            var d = np.arange(6);
            d.setflags(align: false);

            d[":"].flags.aligned.Should().BeTrue("full-slice view");
            d["1:"].flags.aligned.Should().BeTrue("sliced view");
            d.reshape(2, 3).flags.aligned.Should().BeTrue("reshape view");
            d.reshape(2, 3).T.flags.aligned.Should().BeTrue("transpose view");
            d.view(np.int64).flags.aligned.Should().BeTrue("dtype view");
            d.copy().flags.aligned.Should().BeTrue("copy");
            d.flags.aligned.Should().BeFalse("the source keeps its cleared flag");
        }

        [TestMethod]
        public void OneCall_WriteFalse_AlignFalse_BothApply()
        {
            // Probed: setflags(write=False, align=False) → num == C|F|O == 7 on an owned 1-D.
            var p = np.arange(6);
            p.setflags(write: false, align: false);
            p.flags.num.Should().Be(7);
        }

        // ---- uic --------------------------------------------------------------------------

        [TestMethod]
        public void Uic_True_Raises_Verbatim()
        {
            var e = np.arange(6);
            ((Action)(() => e.setflags(uic: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEBACKIFCOPY flag to True");
        }

        [TestMethod]
        [Misaligned]
        public void Uic_False_IsNoOp_BaseKeptAlive()
        {
            // DIVERGENCE (memory safety): NumPy's uic=False clears WRITEBACKIFCOPY and SEVERS the
            // view's base reference (v.base becomes None; the data can dangle once the owner dies).
            // NumSharp has no writeback state and its base chain roots the owner for the view's
            // lifetime, so uic=False is a pure no-op — the view keeps its base and stays valid.
            var e = np.arange(6);
            var v = e["1:"];
            v.setflags(uic: false);
            v.Storage.IsView.Should().BeTrue("the base reference is kept, unlike NumPy's severing");
            v.flags.owndata.Should().BeFalse();
            v.flags.writeable.Should().BeTrue();
            v[0] = (NDArray)42L;
            e.GetInt64(1).Should().Be(42, "the view still writes through its (kept) base");
        }

        // ---- rollback on error (NumPy's flagback restore) ----------------------------------

        [TestMethod]
        public void UicError_RollsBack_SameCallAlignChange()
        {
            // Probed: setflags(align=False, uic=True) raises AND aligned stays True.
            var f = np.arange(6);
            ((Action)(() => f.setflags(align: false, uic: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEBACKIFCOPY flag to True");
            f.flags.aligned.Should().BeTrue("the failed call rolled back its own align change");
        }

        [TestMethod]
        public void WriteRefusal_RollsBack_SameCallAlignChange()
        {
            // Probed: setflags(align=False, write=True) on a refused view raises AND aligned stays True.
            var g = np.arange(6);
            g.setflags(write: false);
            var gv = g["1:"];
            ((Action)(() => gv.setflags(align: false, write: true)))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
            gv.flags.aligned.Should().BeTrue("the failed call rolled back its own align change");
        }

        // ---- the flags-object setters route through setflags -------------------------------

        [TestMethod]
        public void FlagsWriteableSetter_RoutesThroughSetflags_SameRefusal()
        {
            var a = np.arange(6);
            a.setflags(write: false);
            var v = a["1:"];
            ((Action)(() => v.flags.writeable = true))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
            ((Action)(() => v.flags["W"] = true))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEABLE flag to True of this array");
        }

        [TestMethod]
        public void FlagsAlignedSetter_IsLive()
        {
            var a = np.arange(6);
            a.flags["A"] = false;
            a.flags.aligned.Should().BeFalse();
            a.flags["ALIGNED"] = true;
            a.flags.aligned.Should().BeTrue();
        }

        [TestMethod]
        public void FlagsWritebackifcopySetter_TrueRaises()
        {
            var a = np.arange(6);
            ((Action)(() => a.flags["X"] = true))
                .Should().Throw<ValueError>().WithMessage("cannot set WRITEBACKIFCOPY flag to True");
            a.flags["X"] = false; // no-op
        }

        [TestMethod]
        [Misaligned]
        public void FlagsObject_ReadsLive_NumpySnapshotIsStale()
        {
            // DIVERGENCE (deliberate, strictly more useful): NumPy's a.flags object snapshots the
            // flags int at creation, so a held flags object goes STALE after setflags; NumSharp's
            // NDArrayFlags reads live from the array.
            var h = np.arange(6);
            var fobj = h.flags;
            h.setflags(write: false);
            fobj.writeable.Should().BeFalse("NumSharp reads live (NumPy's held object would still say True)");
        }
    }
}
