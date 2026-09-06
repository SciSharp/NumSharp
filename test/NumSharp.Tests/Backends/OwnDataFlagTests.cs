using System;
using System.IO;
using System.Numerics;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The OWNDATA flag — <c>Shape.OwnsData</c> / <c>ArrayFlags.OWNDATA</c> / <c>nd.flags.owndata</c> —
    ///     across the creation / view / copy / in-place surface. Every expectation below is the LIVE-probed
    ///     NumPy 2.4.2 output (<c>a.flags.owndata</c> + <c>a.flags.num</c>); NumPy's rule is
    ///     <c>ctors.c</c>: an allocating constructor raises <c>NPY_ARRAY_OWNDATA</c>, a data-passed-in
    ///     constructor clears it, and <c>base != NULL ⟹ !OWNDATA</c> (asserted in <c>common.c</c>'s
    ///     <c>_IsWriteable</c>). In NumSharp the bit is maintained by
    ///     <c>UnmanagedStorage.OnReshaped()</c> at every storage funnel, mirroring
    ///     <c>UnmanagedStorage.OwnsData</c> (<c>_baseStorage is null &amp;&amp; !ExternalBase</c>), and
    ///     <c>nd.flags.owndata</c> reads the Shape bit — so the flags oracle gates the mirror.
    ///
    ///     <para>
    ///     Deliberately NOT asserted here — the five KNOWN open op-level ownership divergences vs NumPy
    ///     (each returns an owned copy where NumPy returns a view of an internal temp, or vice versa;
    ///     values are identical, only ownership/layout observables differ): <c>np.squeeze</c> no-op on an
    ///     owner (NumPy returns SELF), the <c>np.resize</c> FUNCTION (NumPy: reshape-of-concatenate view),
    ///     <c>np.flatnonzero</c> / <c>np.argwhere</c> (NumPy: views into nonzero's shared buffer), and
    ///     trailing-advanced-axis fancy indexing <c>X[:, [i,j]]</c> (NumPy: F-contiguous transposed-back
    ///     view of a temp). Do not add green assertions for them without fixing the ops first.
    ///     </para>
    /// </summary>
    [TestClass]
    public class OwnDataFlagTests
    {
        // NumPy 2.4.2 flags.num composites used below:
        //   1287 = C|F|OWNDATA|ALIGNED|WRITEABLE   (owning 1-D / 0-d)
        //   1285 = C|OWNDATA|ALIGNED|WRITEABLE     (owning 2-D C-order)
        //   1283 = C|F|ALIGNED|WRITEABLE           (contiguous 1-D view)
        //   1282 = F|ALIGNED|WRITEABLE             (F-contiguous 2-D view, e.g. .T)
        //   1281 = C|ALIGNED|WRITEABLE             (C-contiguous 2-D view)
        //   1280 = ALIGNED|WRITEABLE               (non-contiguous view)
        //    263 = C|F|OWNDATA|ALIGNED             (owner made read-only)
        //    259 = C|F|ALIGNED                     (view made read-only)
        //    256 = ALIGNED                         (read-only non-contiguous view)
        private const int OWNDATA = (int)ArrayFlags.OWNDATA;

        private static void AssertOwns(NDArray nd, string because)
        {
            nd.Shape.OwnsData.Should().BeTrue(because);
            ((int)nd.Shape.Flags & OWNDATA).Should().Be(OWNDATA, because);
            nd.flags.owndata.Should().BeTrue(because);
        }

        private static void AssertView(NDArray nd, string because)
        {
            nd.Shape.OwnsData.Should().BeFalse(because);
            ((int)nd.Shape.Flags & OWNDATA).Should().Be(0, because);
            nd.flags.owndata.Should().BeFalse(because);
        }

        // ---- owners: allocating creation --------------------------------------------------------

        [TestMethod]
        public void Creation_AllocatingConstructors_Own()
        {
            AssertOwns(np.arange(10), "np.arange allocates (NumPy owndata=True, num=1287)");
            np.arange(10).flags.num.Should().Be(1287);

            AssertOwns(np.zeros(new Shape(5)), "np.zeros allocates (NumPy num=1287)");
            AssertOwns(np.ones(new Shape(2, 3)), "np.ones allocates (NumPy num=1285)");
            np.ones(new Shape(2, 3)).flags.num.Should().Be(1285);
            AssertOwns(np.full(new Shape(2, 2), 7), "np.full allocates (NumPy num=1285)");
            AssertOwns(np.array(new[] { 1, 2, 3 }), "np.array copies a managed array (NumPy num=1287)");
            AssertOwns(np.eye(3), "np.eye allocates (NumPy num=1285)");
            AssertOwns(np.linspace(0.0, 1.0, 5).copy(), "a copy always owns");
        }

        [TestMethod]
        public void Creation_ZeroDim_And_Empty_Own()
        {
            AssertOwns(NDArray.Scalar(5), "a 0-d scalar array owns its buffer (NumPy np.array(5): num=1287)");
            NDArray.Scalar(5).flags.num.Should().Be(1287);

            AssertOwns(np.zeros(new Shape(0, 3)), "an empty array still owns its (zero-length) buffer");
        }

        [TestMethod]
        public void Creation_ScalarAndVectorCtors_AllDtypes_Own()
        {
            // The ~30 codegen'd UnmanagedStorage scalar/vector ctors assign the pre-owned shape statics;
            // spot the families across dtypes (flags are dtype-independent, but the ctors are per-dtype).
            AssertOwns(NDArray.Scalar(true), "bool scalar ctor");
            AssertOwns(NDArray.Scalar((byte)1), "byte scalar ctor");
            AssertOwns(NDArray.Scalar(1.5), "double scalar ctor");
            AssertOwns(NDArray.Scalar(new Complex(1, 2)), "complex scalar ctor");
            AssertOwns(NDArray.Scalar((decimal)1.5), "decimal scalar ctor");

            AssertOwns(np.array(new bool[] { true, false }), "bool vector ctor");
            AssertOwns(np.array(new byte[] { 1, 2 }), "byte vector ctor");
            AssertOwns(np.array(new float[] { 1, 2 }), "float vector ctor");
            AssertOwns(np.array(new Complex[] { new(1, 2) }), "complex vector ctor");
        }

        [TestMethod]
        public void CopyAndCast_Own()
        {
            var a = np.arange(10);
            AssertOwns(a.copy(), ".copy() allocates (NumPy num=1287)");
            AssertOwns(a.astype(np.float64), "astype allocates (NumPy num=1287)");
            AssertOwns(a.astype(a.dtype), "astype same-dtype still copies by default (NumPy copy=True)");
            AssertOwns(np.ascontiguousarray(a["::2"]), "materializing a strided view allocates (NumPy num=1287)");
        }

        // ---- owners: copy-returning ops ---------------------------------------------------------

        [TestMethod]
        public void CopyReturningOps_Own()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);

            AssertOwns(b2.flatten(), "flatten always copies (NumPy num=1287)");
            AssertOwns(b2.T.ravel(), "ravel of a NON-contiguous array copies via PyArray_Flatten (NumPy num=1287)");
            AssertOwns(a[np.array(new[] { 0, 2 })], "leading-axis fancy indexing materializes (NumPy num=1287)");
            AssertOwns(a[a > 3], "boolean-mask selection materializes (NumPy num=1287)");
            AssertOwns(np.sort(a), "np.sort returns a fresh copy (NumPy num=1287)");
            NDArray uniq = np.unique(a);
            AssertOwns(uniq, "np.unique returns a fresh array (NumPy num=1287)");
            AssertOwns(np.concatenate((a, a)), "concatenate allocates (NumPy num=1287)");
            AssertOwns(np.take(a, np.array(new[] { 0, 1 })), "np.take gathers into a fresh array (NumPy num=1287)");
            AssertOwns(np.repeat(a, 2), "np.repeat allocates (NumPy num=1287)");
            AssertOwns(np.diag(a), "np.diag 1-D→2-D CONSTRUCTS a fresh matrix (NumPy num=1285)");
        }

        [TestMethod]
        public void ArithmeticAndReductions_Own()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);

            AssertOwns(a + a, "an elementwise result allocates (NumPy num=1287)");
            AssertOwns(b2.sum(0), "an axis reduction allocates (NumPy num=1287)");
            AssertOwns(np.sum(b2, 0, keepdims: true), "keepdims reductions allocate too (NumPy num=1287)");
            AssertOwns(a.cumsum(), "cumsum allocates (NumPy num=1287)");
        }

        // ---- views: slicing ---------------------------------------------------------------------

        [TestMethod]
        public void Slicing_Views_DoNotOwn()
        {
            var a = np.arange(10);

            AssertView(a["1:5"], "a basic slice is a view (NumPy num=1283)");
            a["1:5"].flags.num.Should().Be(1283);
            AssertView(a["::2"], "a stepped slice is a view (NumPy num=1280)");
            a["::2"].flags.num.Should().Be(1280);
            AssertView(a["::-1"], "a negative-stride slice is a view (NumPy num=1280)");
            AssertView(a["8:100"], "an out-of-range (empty) slice is still a view (NumPy num=1283)");
        }

        [TestMethod]
        public void FullSliceAndView_BorrowTheParentShape_ButDoNotOwn()
        {
            // The sharpest funnel: a[":"] routes through Alias(), which copies the PARENT's shape
            // verbatim — the parent's OWNDATA bit rides the copied struct and must be cleared.
            var a = np.arange(10);
            AssertView(a[":"], "a[:] is a full view of an owner (NumPy a[:]: owndata=False, num=1283)");
            AssertView(a.view(), "ndarray.view() is a same-dtype view (NumPy num=1283)");
        }

        [TestMethod]
        public void ViewOfView_ChainsToOwner_StillDoesNotOwn()
        {
            var a = np.arange(10);
            var v1 = a["2:8"];
            var v2 = v1["1:3"];
            AssertView(v2, "a view of a view stays a view (base chains to the ultimate owner)");
            v2.Storage.BaseStorage.Should().BeSameAs(a.Storage, "base chains collapse to the owner (NumPy semantics)");
        }

        [TestMethod]
        public void RowAndElementViews_DoNotOwn()
        {
            var b2 = np.arange(12).reshape(3, 4);
            AssertView(b2[0], "a row sub-array is a view (NumPy num=1283)");
            AssertView(b2[0][0], "a 0-d element view shares storage (NumSharp GetData view; NumPy a[0,0] is a scalar)");
        }

        // ---- views: axis movers -----------------------------------------------------------------

        [TestMethod]
        public void AxisMovers_AreViews()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);

            AssertView(b2.T, ".T is a view (NumPy num=1282, F-contiguous)");
            b2.T.flags.num.Should().Be(1282);
            AssertView(a.T, "even the identity 1-D transpose is a fresh view in NumPy (num=1283)");
            AssertView(np.transpose(a), "np.transpose of 1-D is a view (NumPy num=1283)");
            AssertView(np.swapaxes(b2, 0, 1), "swapaxes is a view (NumPy num=1282)");
            AssertView(np.moveaxis(b2, 0, 1), "moveaxis is a view (NumPy num=1282)");
            AssertView(np.rollaxis(b2, 0, 0), "the identity rollaxis is STILL a transpose view (NumPy num=1281)");
            np.rollaxis(b2, 0, 0).flags.num.Should().Be(1281);
        }

        // ---- views: reshape family --------------------------------------------------------------

        [TestMethod]
        public void ReshapeFamily_ViewsAndViewOfCopy_DoNotOwn()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);

            AssertView(a.reshape(2, 5), "reshape of a contiguous array is a view (NumPy num=1281)");
            AssertView(b2.reshape(-1), "reshape(-1) of a contiguous array is a view (NumPy num=1283)");
            AssertView(b2.ravel(), "ravel of a contiguous array is a reshape view (NumPy num=1283)");

            // NumPy 2.x _reshape_with_copy_arg: when reshape MUST copy, the returned array is a
            // view OF THE INTERNAL COPY — owndata=False with a base, NOT an owner (probed num=1283).
            AssertView(b2.T.reshape(-1), "the reshape COPY path returns a view-of-internal-copy (NumPy owndata=False)");
            b2.T.reshape(-1).flags.num.Should().Be(1283);
        }

        [TestMethod]
        public void NewAxisAndSqueeze_Views()
        {
            var a = np.arange(10);
            AssertView(np.expand_dims(a, 0), "expand_dims is a view (NumPy num=1283)");
            AssertView(np.squeeze(np.expand_dims(a, 0)), "squeeze that removes axes is a view (NumPy num=1283)");
        }

        // ---- views: broadcast / diagonal / composition ops --------------------------------------

        [TestMethod]
        public void BroadcastViews_DoNotOwn()
        {
            var a = np.arange(10);
            var bc = np.broadcast_to(a, new Shape(3, 10));
            AssertView(bc, "broadcast_to is a read-only view (NumPy num=256)");
            bc.flags.num.Should().Be(256);

            var (lhs, rhs) = np.broadcast_arrays(a, np.arange(30).reshape(3, 10));
            AssertView(lhs, "broadcast_arrays results are views (NumPy owndata=False)");
            AssertView(rhs, "both broadcast_arrays results are views (NumPy owndata=False)");
        }

        [TestMethod]
        public void DiagonalAndLaneViews_DoNotOwn()
        {
            var b2 = np.arange(12).reshape(3, 4);
            AssertView(b2.diagonal(), "diagonal is a read-only view (NumPy num=256)");
            AssertView(np.diag(b2), "np.diag 2-D→1-D EXTRACTS a view (NumPy num=256)");

            var c = np.array(new Complex[] { new(1, 2), new(3, 4) });
            AssertView(c.real, ".real of complex is a strided lane view (NumPy num=1280)");
            c.real.flags.num.Should().Be(1280);
            AssertView(c.imag, ".imag of complex is a strided lane view (NumPy num=1280)");
        }

        [TestMethod]
        public void ViewReturningOps_DoNotOwn()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);

            AssertView(np.flip(a), "np.flip is a stride-negated view (NumPy num=1280)");
            AssertView(np.rot90(b2), "np.rot90 is a flip+transpose view (NumPy num=1280)");
            AssertView(np.split(a, 2)[0], "np.split children are views (NumPy num=1283)");
            AssertView(np.tile(a, 2), "np.tile ends in a reshape → view of its internal buffer (NumPy num=1283)");
            AssertView(np.roll(a, 1), "np.roll returns a view of its internal result (NumPy num=1283)");
            AssertView(np.nonzero(a)[0], "nonzero columns are views into one shared index matrix (NumPy num=1283)");
            AssertView(np.where(a > 3)[0], "where(cond) inherits nonzero's shared-buffer views (NumPy num=1283)");
        }

        // ---- views: dtype reinterpretation ------------------------------------------------------

        [TestMethod]
        public void DtypeReinterpretingViews_DoNotOwn()
        {
            var i32 = np.arange(4).astype(NPTypeCode.Int32);
            AssertView(i32.view(typeof(float)), "view(dtype) same itemsize is a reinterpreting view (NumPy num=1283)");

            var i64 = np.arange(4);
            AssertView(i64.view(typeof(int)), "view(dtype) different itemsize is still a view (NumPy num=1283)");
            AssertView(i64.getfield(typeof(int), 0), "getfield is a byte-field view (NumPy owndata=False)");
        }

        // ---- in-place mutations preserve ownership ----------------------------------------------

        [TestMethod]
        public void InPlaceMutations_PreserveOwnership()
        {
            // NumPy methods.c: ndarray.resize requires OWNDATA and re-ENABLES it after realloc.
            var r = np.arange(10);
            r.resize(new Shape(20));
            AssertOwns(r, "in-place resize keeps ownership (NumPy num=1287)");

            var q = np.arange(10);
            q.Storage.Reshape(2, 5);
            AssertOwns(q, "an in-place storage reshape must not strip the owner's bit");

            var e = np.arange(10);
            e.Storage.ExpandDimension(0);
            AssertOwns(e, "Storage.ExpandDimension rebuilds the shape — the bit must survive");

            var p = np.arange(10);
            p.Storage.ReplaceData(np.arange(3));
            AssertOwns(p, "ReplaceData swaps in a fresh shape — the bit must survive");
        }

        [TestMethod]
        public void Setflags_PreservesOwnershipOnBothSides()
        {
            // Owner made read-only: OWNDATA survives (NumPy num=263 = C|F|OWNDATA|ALIGNED).
            var s = np.arange(10);
            s.setflags(write: false);
            AssertOwns(s, "setflags(write=False) never changes OWNDATA (NumPy num=263)");
            s.flags.num.Should().Be(263);
            s.setflags(write: true);
            AssertOwns(s, "re-enabling writeable never changes OWNDATA (NumPy num=1287)");

            // View made read-only: stays a view (NumPy num=259 = C|F|ALIGNED).
            var sv = np.arange(10)["1:5"];
            sv.setflags(write: false);
            AssertView(sv, "setflags on a view never grants ownership (NumPy num=259)");
            sv.flags.num.Should().Be(259);

            // Align-cleared owner keeps the bit too (setflags routes through SetShapeUnsafe).
            var al = np.arange(10);
            al.setflags(align: false);
            AssertOwns(al, "setflags(align=False) never changes OWNDATA");
        }

        [TestMethod]
        public void ReadOnlyOwner_ImagOfReal_OwnsButNotWriteable()
        {
            // np.imag of a REAL array is NumPy's fresh zeros with writeable=False — an owner that is
            // read-only (probed num=263): OWNDATA and WRITEABLE are independent bits.
            var im = np.imag(np.arange(3).astype(NPTypeCode.Double));
            AssertOwns(im, "np.imag(real) returns a fresh read-only ZEROS array (NumPy owndata=True)");
            im.flags.num.Should().Be(263);
        }

        // ---- memmap / foreign memory ------------------------------------------------------------

        [TestMethod]
        public void Memmap_EveryMode_NeverOwns()
        {
            // NumPy's memmap has the mmap object as its non-array base: owndata=False in EVERY mode.
            // 'r+' is the sharpest regression: it takes NO later shape swap, so only the explicit
            // OnReshaped() after ExternalBase=true clears the wrap ctor's bit.
            string dir = Path.Combine(Path.GetTempPath(), "ns_owndata_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                foreach (string mode in new[] { "r", "r+", "c" })
                {
                    // A live 'r' mapping LOCKS its file until the array is collected — a fresh file
                    // per mode (the FlagsOracleRecipes rule) instead of re-opening one.
                    string p = Path.Combine(dir, $"m_{mode.Replace('+', 'w')}.npy");
                    np.save(p, np.arange(6).astype(NPTypeCode.Int32));
                    var m = (NDArray)np.load(p, mmap_mode: mode);
                    AssertView(m, $"a memmap ('{mode}') never owns — its memory belongs to the file mapping");
                    m.Storage.OwnsData.Should().BeFalse("ExternalBase marks the foreign owner");
                }
            }
            finally
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
            }
        }

        // ---- the mirror invariant ---------------------------------------------------------------

        [TestMethod]
        public void Mirror_ShapeBit_StorageTruth_FlagsSurface_AlwaysAgree()
        {
            var a = np.arange(10);
            var b2 = np.arange(12).reshape(3, 4);
            var c = np.array(new Complex[] { new(1, 2), new(3, 4), new(5, 6) });

            var cases = new (string name, NDArray nd)[]
            {
                ("arange", a), ("zeros", np.zeros(new Shape(5))), ("scalar", NDArray.Scalar(1.5)),
                ("copy", a.copy()), ("astype", a.astype(np.float32)),
                ("slice", a["1:5"]), ("step", a["::2"]), ("reversed", a["::-1"]), ("full-slice", a[":"]),
                ("T", b2.T), ("swapaxes", np.swapaxes(b2, 0, 1)), ("reshape-view", a.reshape(2, 5)),
                ("reshape-copy-path", b2.T.reshape(-1)), ("ravel-noncontig", b2.T.ravel()),
                ("expand_dims", np.expand_dims(a, 0)), ("squeeze", np.squeeze(np.expand_dims(a, 0))),
                ("broadcast_to", np.broadcast_to(a, new Shape(3, 10))),
                ("diagonal", b2.diagonal()), ("row", b2[0]), ("element-0d", b2[0][0]),
                ("fancy", a[np.array(new[] { 0, 2 })]), ("mask", a[a > 3]),
                ("real-lane", c.real), ("imag-lane", c.imag),
                ("view()", a.view()), ("view(f4)", np.arange(4).astype(NPTypeCode.Int32).view(typeof(float))),
                ("getfield", a.getfield(typeof(int), 0)),
                ("flip", np.flip(a)), ("split-child", np.split(a, 2)[0]),
                ("nonzero", np.nonzero(a)[0]), ("sum-axis", b2.sum(0)), ("a+a", a + a),
                ("empty", np.zeros(new Shape(0, 3))), ("empty-slice", a["8:100"]),
            };

            foreach (var (name, nd) in cases)
            {
                bool shapeBit = nd.Shape.OwnsData;
                bool enumBit = ((int)nd.Shape.Flags & OWNDATA) != 0;
                bool storageTruth = nd.Storage.OwnsData;
                bool flagsSurface = nd.flags.owndata;
                bool flagsKey = nd.flags["O"];
                bool numBit = (nd.flags.num & OWNDATA) != 0;

                shapeBit.Should().Be(storageTruth, $"[{name}] Shape.OwnsData must mirror Storage.OwnsData");
                enumBit.Should().Be(storageTruth, $"[{name}] Shape.Flags OWNDATA must mirror Storage.OwnsData");
                flagsSurface.Should().Be(storageTruth, $"[{name}] flags.owndata must mirror Storage.OwnsData");
                flagsKey.Should().Be(storageTruth, $"[{name}] flags[\"O\"] must mirror Storage.OwnsData");
                numBit.Should().Be(storageTruth, $"[{name}] flags.num bit 4 must mirror Storage.OwnsData");
            }
        }

        // ---- the Shape struct itself ------------------------------------------------------------

        [TestMethod]
        public void HandBuiltShapes_CarryNoOwnershipClaim()
        {
            // Ownership is a storage fact — a Shape built from dims/strides makes no claim, and the
            // SHARED Shape.Scalar static must never carry the bit (views copy it verbatim).
            new Shape(3, 4).OwnsData.Should().BeFalse("a hand-built shape has no storage behind it");
            new Shape(new long[] { 4 }, new long[] { 1 }, 0, 4).OwnsData.Should().BeFalse();
            Shape.Scalar.OwnsData.Should().BeFalse("the shared Scalar static is also used for view shapes");
            Shape.NewScalar().OwnsData.Should().BeFalse();
            Shape.Vector(5).OwnsData.Should().BeFalse();
        }

        [TestMethod]
        public void WithFlags_OwnDataRoundtrip_PreservesEverythingElse()
        {
            var s = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 2, 14);
            var owned = s.WithFlags(flagsToSet: ArrayFlags.OWNDATA);

            owned.OwnsData.Should().BeTrue();
            owned.size.Should().Be(s.size, "WithFlags must not disturb size");
            owned.GetHashCode().Should().Be(s.GetHashCode(), "WithFlags must not disturb the hash");
            owned.dimensions.Should().BeSameAs(s.dimensions, "dims array is carried by reference");
            owned.strides.Should().BeSameAs(s.strides, "strides array is carried by reference");
            owned.offset.Should().Be(s.offset);
            owned.bufferSize.Should().Be(s.bufferSize);
            owned.IsContiguous.Should().Be(s.IsContiguous, "layout flags are untouched");
            owned.IsWriteable.Should().Be(s.IsWriteable);

            var cleared = owned.WithFlags(flagsToClear: ArrayFlags.OWNDATA);
            cleared.OwnsData.Should().BeFalse();
            cleared._flags.Should().Be(s._flags, "set→clear round-trips to the original flags word");
        }

        [TestMethod]
        public void OwnDataBit_DoesNotAffectShapeEquality()
        {
            // Shape equality/hash are layout-based; two shapes differing ONLY in OWNDATA are the same
            // layout — an owner and its full view compare equal (NumPy: a.shape == a[:].shape).
            var s = new Shape(3, 4);
            var owned = s.WithFlags(flagsToSet: ArrayFlags.OWNDATA);
            owned.Equals(s).Should().BeTrue("ownership is not part of layout identity");
            (owned == s).Should().BeTrue();
            owned.GetHashCode().Should().Be(s.GetHashCode());

            np.arange(10).Shape.Should().Be(np.arange(10)[":"].Shape, "owner and full view share the layout");
        }

        [TestMethod]
        public void FlagsNum_MatchesNumPy_PinnedComposites()
        {
            // The exact flags.num integers NumPy 2.4.2 reports for the layout×ownership composites.
            np.arange(10).flags.num.Should().Be(1287);                       // owning 1-D
            np.ones(new Shape(3, 4)).flags.num.Should().Be(1285);            // owning 2-D
            np.arange(12).reshape(3, 4).flags.num.Should().Be(1281);         // reshape VIEW of 1-D owner
            np.arange(10)["1:5"].flags.num.Should().Be(1283);                // contiguous view
            np.arange(12).reshape(3, 4).T.flags.num.Should().Be(1282);       // F-contiguous view
            np.arange(10).reshape(2, 5).flags.num.Should().Be(1281);         // C-contiguous 2-D view
            np.arange(10)["::2"].flags.num.Should().Be(1280);                // strided view
            np.broadcast_to(np.arange(10), new Shape(3, 10)).flags.num.Should().Be(256); // read-only broadcast
            np.arange(12).reshape(3, 4).diagonal().flags.num.Should().Be(256);           // read-only diagonal
        }
    }
}
