using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    ///     Regression matrix for the unified copy / retype / cast core (NDIter.Copy + NDIter.CopyAs,
    ///     reached by ndarray.copy(), astype(), Clone(), and np.copyto). These are the probes that
    ///     drove the core's design, pinned as always-on tests. They are SELF-VALIDATING — no committed
    ///     NumPy oracle is needed here (the bytes-exact NumPy differential lives in the FuzzMatrix
    ///     astype/copyto corpora); each assertion compares the core against an INDEPENDENT ground truth:
    ///
    ///       Part A  — NDIter.Copy materializes EVERY layout (15 dtypes x 8 layouts) bit-for-bit equal
    ///                 to a raw stride-walk of the source (a reader that does not touch the copy engine).
    ///       Part B/C— casting is layout-invariant: astype(view) == astype(view.copy()) for cross-dtype
    ///                 pairs across all layouts (the strided / F / broadcast cast kernels must agree with
    ///                 the contiguous reference — the structural content of the 169 + 126 cast matrices).
    ///       (a)/(c) — the scalar-broadcast fast fills (same-dtype C/F + cross-dtype) reproduce the
    ///                 element-by-element general path exactly.
    ///       Part D  — overlapping np.copyto (COPY_IF_OVERLAP) equals a copy-source-first reference,
    ///                 including the strided write-before-read case (copyto(a[2:8:2], a[0:6:2])).
    /// </summary>
    [TestClass]
    public class CopyCastCoreParityTests
    {
        private static readonly (NPTypeCode tc, string nm)[] Dtypes =
        {
            (NPTypeCode.Boolean, "bool"), (NPTypeCode.Byte, "u8"), (NPTypeCode.SByte, "i8"),
            (NPTypeCode.Int16, "i16"), (NPTypeCode.UInt16, "u16"), (NPTypeCode.Int32, "i32"),
            (NPTypeCode.UInt32, "u32"), (NPTypeCode.Int64, "i64"), (NPTypeCode.UInt64, "u64"),
            (NPTypeCode.Char, "char"), (NPTypeCode.Half, "f16"), (NPTypeCode.Single, "f32"),
            (NPTypeCode.Double, "f64"), (NPTypeCode.Decimal, "dec"), (NPTypeCode.Complex, "c128"),
        };

        private static NDArray Base(NPTypeCode tc) => np.arange(24).astype(tc).reshape(6, 4);

        private static readonly (string name, Func<NPTypeCode, NDArray> build)[] Layouts =
        {
            ("C-2d",        tc => Base(tc)),
            ("F-transpose", tc => Base(tc).T),
            ("strided",     tc => Base(tc)["::2, ::2"]),
            ("reversed",    tc => Base(tc)["::-1, ::-1"]),
            ("offset",      tc => Base(tc)["2:5, 1:3"]),
            ("broadcast",   tc => np.broadcast_to(np.arange(4).astype(tc).reshape(1, 4), new Shape(6, 4))),
            ("size1",       tc => np.arange(1).astype(tc)),
            ("empty",       tc => np.arange(0).astype(tc).reshape(2, 0, 3)),
        };

        // -- INDEPENDENT ground truth: walk C-order coords, read raw ItemLength bytes via strides.
        //    Deliberately does NOT call copy()/astype/CopyAs so it can validate them.
        private static unsafe string RawLogicalHex(NDArray x)
        {
            var sh = x.Shape;
            int nd = (int)sh.NDim;
            int il = (int)x.Storage.InternalArray.ItemLength;
            byte* basep = x.Storage.Address;
            var dims = sh.dimensions;
            var str = sh.strides;
            long off = sh.offset;
            long size = 1;
            for (int d = 0; d < nd; d++) size *= dims[d];

            var sb = new StringBuilder();
            if (nd == 0)
            {
                for (int b = 0; b < il; b++) sb.Append((basep + off * il)[b].ToString("x2"));
                return sb.ToString();
            }
            var coord = new int[nd];
            for (long i = 0; i < size; i++)
            {
                long eoff = off;
                for (int d = 0; d < nd; d++) eoff += (long)coord[d] * str[d];
                for (int b = 0; b < il; b++) sb.Append((basep + eoff * il)[b].ToString("x2"));
                sb.Append('|');
                for (int d = nd - 1; d >= 0; d--) { if (++coord[d] < dims[d]) break; coord[d] = 0; }
            }
            return sb.ToString();
        }

        private static long[] DimsOf(NDArray a) => (long[])a.Shape.dimensions.Clone();

        // ============================ Part A — materialization (120) ============================

        [TestMethod]
        public void Core_NDIterCopy_MaterializesEveryLayout_AllDtypes()
        {
            var fail = new List<string>();
            int n = 0;
            foreach (var (tc, nm) in Dtypes)
            foreach (var (lname, build) in Layouts)
            {
                n++;
                var src = build(tc);
                string truth = RawLogicalHex(src);
                var dst = new NDArray(tc, new Shape(DimsOf(src)), false);
                NDIter.Copy(dst, src);
                if (RawLogicalHex(dst) != truth)
                    fail.Add($"{nm}/{lname}: core != raw-logical");
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} layout materializations diverged:\n  {string.Join("\n  ", fail)}");
        }

        [TestMethod]
        public void PublicBypasses_CopyCloneCloneData_MatchRawLogical()
        {
            var fail = new List<string>();
            int n = 0;
            foreach (var (tc, nm) in Dtypes)
            foreach (var (lname, build) in Layouts)
            {
                var src = build(tc);
                string truth = RawLogicalHex(src);

                n++;
                if (RawLogicalHex(src.copy()) != truth) fail.Add($"{nm}/{lname}: copy() != raw-logical");
                n++;
                if (RawLogicalHex(new NDArray(src.Storage.Clone())) != truth) fail.Add($"{nm}/{lname}: Clone() != raw-logical");
                n++;
                if (ContigHex(src.Storage.CloneData(), src.size, (int)src.Storage.InternalArray.ItemLength) != truth)
                    fail.Add($"{nm}/{lname}: CloneData() != raw-logical");
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} bypass materializations diverged:\n  {string.Join("\n  ", fail)}");
        }

        private static unsafe string ContigHex(IArraySlice s, long count, int il)
        {
            byte* p = (byte*)s.Address;
            var sb = new StringBuilder();
            for (long i = 0; i < count; i++)
            {
                for (int b = 0; b < il; b++) sb.Append(p[i * il + b].ToString("x2"));
                sb.Append('|');
            }
            return sb.ToString();
        }

        // ===================== Part B/C — cast is layout-invariant (169 + 126) =====================

        private static readonly (NPTypeCode s, NPTypeCode d, string nm)[] CrossPairs =
        {
            (NPTypeCode.Int32, NPTypeCode.Double, "i32->f64"),
            (NPTypeCode.Double, NPTypeCode.Int32, "f64->i32"),
            (NPTypeCode.Single, NPTypeCode.Byte, "f32->u8"),
            (NPTypeCode.Int16, NPTypeCode.Int64, "i16->i64"),
            (NPTypeCode.Double, NPTypeCode.Half, "f64->f16"),
            (NPTypeCode.Int32, NPTypeCode.Boolean, "i32->bool"),
            (NPTypeCode.Byte, NPTypeCode.Single, "u8->f32"),
            (NPTypeCode.Double, NPTypeCode.UInt32, "f64->u32"),
            (NPTypeCode.Complex, NPTypeCode.Double, "c128->f64"),
            (NPTypeCode.Int64, NPTypeCode.Int16, "i64->i16"),
        };

        [TestMethod]
        public void Cast_IsLayoutInvariant_AcrossLayouts()
        {
            // astype(view) must produce the SAME logical values as astype(view.copy()): the strided /
            // F / broadcast cast kernels must agree with the contiguous reference for every pair.
            var fail = new List<string>();
            int n = 0;
            foreach (var (s, d, pnm) in CrossPairs)
            foreach (var (lname, build) in Layouts)
            {
                if (lname == "empty") continue; // no elements to cast
                n++;
                var view = build(s);
                string viaView = RawLogicalHex(view.astype(d));
                string viaContig = RawLogicalHex(view.copy().astype(d));   // copy() => C-contiguous of view's logical values
                if (viaView != viaContig)
                    fail.Add($"{pnm}/{lname}: astype(view) != astype(view.copy())");
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} cast layout-invariance checks diverged:\n  {string.Join("\n  ", fail)}");
        }

        // ===================== (a) + (c) — scalar-broadcast fast fills =====================

        [TestMethod]
        public void ScalarBroadcast_SameType_Fill_CandF_AllDtypes()
        {
            // (a): a scalar-broadcast (ALL strides 0) filled into a whole-buffer C OR F dst must equal
            //      the element-by-element general path (CloneData materialization).
            var fail = new List<string>();
            int n = 0;
            foreach (var (tc, nm) in Dtypes)
            foreach (var dims in new[] { new long[] { 1000 }, new long[] { 6, 4 }, new long[] { 3, 2, 4 } })
            {
                var src = np.broadcast_to(NDArray.Scalar(5, tc), new Shape(dims));
                string truth = RawLogicalHex(new NDArray(src.Storage.CloneData(), new Shape((long[])dims.Clone())));

                n++;
                var cDst = new NDArray(tc, new Shape((long[])dims.Clone()), false);
                NDIter.Copy(cDst, src);
                if (RawLogicalHex(cDst) != truth) fail.Add($"{nm}/[{string.Join(",", dims)}] C-fill != general");

                n++;
                var fDst = src.copy('F'); // F-contiguous whole-buffer dst -> (a) broadened guard fills it
                if (RawLogicalHex(fDst) != truth) fail.Add($"{nm}/[{string.Join(",", dims)}] F-fill != general");
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} scalar-broadcast same-type fills diverged:\n  {string.Join("\n  ", fail)}");
        }

        [TestMethod]
        public unsafe void ScalarBroadcast_CrossType_Fill_MatchesGeneralPath()
        {
            // (c): scalar-broadcast cross-dtype into a whole-buffer dst (the convert-once-then-fill fast
            //      path) must equal the per-element general path. Force the general path with a STRIDED
            //      dst (declines the fast fill), and stress every cross pair with each source edge value.
            var fail = new List<string>();
            int n = 0;
            foreach (var (s, d, pnm) in CrossPairs)
            {
                var srcBase = Base(s).reshape(24); // 24 distinct-ish source values incl. boundaries
                for (int i = 0; i < 24; i++)
                {
                    var one = srcBase[$"{i}:{i + 1}"];                 // (1,) view of value i
                    var bcast = np.broadcast_to(one, new Shape(5));    // scalar-broadcast (stride 0)

                    // FAST path: whole-buffer contiguous dst.
                    var fast = new NDArray(d, new Shape(5), false);
                    NDIter.Copy(fast, bcast);

                    // GENERAL path: strided dst (every other slot of a length-10 buffer) declines the fill.
                    var gbuf = new NDArray(d, new Shape(10), false);
                    var gview = gbuf["::2"];
                    NDIter.Copy(gview, bcast);

                    n++;
                    if (RawLogicalHex(fast) != RawLogicalHex(gview))
                        fail.Add($"{pnm} val#{i}: fast-fill != general");
                }
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} scalar-broadcast cross-type fills diverged:\n  {string.Join("\n  ", fail.GetRange(0, Math.Min(30, fail.Count)))}");
        }

        // ===================== Part D — overlapping copyto (COPY_IF_OVERLAP) =====================

        [TestMethod]
        public void Overlap_Copyto_MatchesCopySourceFirst()
        {
            // np.copyto(dst, src) where dst & src are overlapping views of one buffer must equal the
            // copy-source-to-a-temp-first reference (NumPy COPY_IF_OVERLAP). Includes the strided
            // write-before-read case that the NeedsAssignmentTemp gate originally mishandled.
            var fail = new List<string>();
            int n = 0;
            var specs1d = new (string tag, Func<NDArray, NDArray> dst, Func<NDArray, NDArray> src)[]
            {
                ("shift_fwd", a => a["1:8"],   a => a["0:7"]),
                ("shift_bwd", a => a["0:7"],   a => a["1:8"]),
                ("reverse",   a => a["0:8"],   a => a["::-1"]),
                ("step_wbr",  a => a["2:8:2"], a => a["0:6:2"]),  // write-before-read strided overlap
                ("step_rev",  a => a["0:6:2"], a => a["2:8:2"]),
            };

            foreach (var (tc, nm) in Dtypes)
            {
                if (tc == NPTypeCode.Boolean) continue; // 0/1 values can't distinguish overlap corruption
                foreach (var (tag, dstF, srcF) in specs1d)
                {
                    n++;
                    // overlapping
                    var work = np.arange(8).astype(tc);
                    np.copyto(dstF(work), srcF(work));
                    string got = RawLogicalHex(dstF(work));

                    // reference: copy source first into a separate buffer, then assign (no overlap)
                    var refb = np.arange(8).astype(tc);
                    var temp = srcF(refb).copy();
                    np.copyto(dstF(refb), temp);
                    string want = RawLogicalHex(dstF(refb));

                    if (got != want) fail.Add($"{nm}/{tag}: overlap copyto != copy-first");
                }

                // 2-D in-place transpose (square) + full reversal.
                foreach (var (tag, srcF) in new (string, Func<NDArray, NDArray>)[]
                         {
                             ("transpose_2d", a => a.T),
                             ("reverse_2d",   a => a["::-1, ::-1"]),
                         })
                {
                    n++;
                    var work = np.arange(16).astype(tc).reshape(4, 4);
                    np.copyto(work, srcF(work));
                    string got = RawLogicalHex(work);

                    var refb = np.arange(16).astype(tc).reshape(4, 4);
                    var temp = srcF(refb).copy();
                    np.copyto(refb, temp);
                    string want = RawLogicalHex(refb);

                    if (got != want) fail.Add($"{nm}/{tag}: overlap copyto != copy-first");
                }
            }
            Assert.AreEqual(0, fail.Count, $"{fail.Count}/{n} overlap copyto checks diverged:\n  {string.Join("\n  ", fail)}");
        }
    }
}
