using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NumSharp.Tests.LongIndexing;

/// <summary>
/// Long-indexing oracle: exercises np.* functions on a <b>byte (uint8)</b> array whose element
/// count EXCEEDS <see cref="int.MaxValue"/>, to CONFIRM which functions honour > 2^31 elements
/// and which silently truncate to a 32-bit index/length. The stated goal is that <b>all</b>
/// functions should support it.
///
/// <para><b>Why byte, why this size.</b></para>
/// <para>
/// byte is 1 byte/element, so it is the ONLY dtype whose 2^31-element array fits in the ~2 GB
/// floor needed to cross the int boundary (int32 would be ~9 GB, double ~18 GB per array). The
/// size is <see cref="N"/> = <c>int.MaxValue + 16</c> — the smallest round value safely past 2^31,
/// keeping per-op memory at the minimum that still forces every internal loop / shape / index off
/// the 32-bit path.
/// </para>
///
/// <para><b>Why this is not the committed differential-fuzz corpus.</b></para>
/// <para>
/// The NumPy oracle corpus commits operand bytes; a single &gt; 2 GB case cannot be committed and
/// NumPy generating 2B+-element arrays is impractical. So this gate is a <b>self-checking</b>
/// oracle: the expected answers are known analytically (a handful of sentinel values poked at
/// indices that straddle int.MaxValue), needing no Python at test time — the same "confirm the
/// long path" role, expressed the only way a &gt; 2 GB case can be.
/// </para>
///
/// <para><b>The load-bearing trick — sentinels ABOVE int.MaxValue.</b></para>
/// <para>
/// A truncation to 32 bits only misbehaves at indices &gt;= 2^31. So every sentinel that a check
/// reads back sits at <see cref="Over"/> (= 2^31, the first index no positive int32 can hold) or
/// <see cref="Last"/> (= N-1). Index-returning ops (argmax/argmin/nonzero/flatnonzero/argwhere/
/// where) are the sharpest probes: their RESULT is an index &gt; int.MaxValue, so a 32-bit return
/// is caught even though the result array is tiny.
/// </para>
///
/// <para><b>Memory &amp; CI.</b></para>
/// <para>
/// Class is <c>[HighMemory]</c> (CI-excluded) + <c>[LongIndexing]</c>. Each large array is disposed
/// eagerly (NDArray.Dispose frees the unmanaged buffer synchronously; the finalizer only
/// <i>abandons</i>, so relying on GC would OOM the sweep). Each method guards its peak with
/// <see cref="RequireAvailableMemory"/> and goes Inconclusive rather than OOM a machine that
/// selects the category. Run with: <c>dotnet test --filter "TestCategory=HighMemory"</c>.
/// </para>
/// </summary>
[TestClass]
[LongIndexing]
[HighMemory]
public class ByteHugeArrayOracleTests
{
    /// <summary>Element count, just past 2^31 so every internal index/length must be 64-bit. ~2.0 GB as byte.</summary>
    private const long N = (long)int.MaxValue + 16; // 2,147,483,663

    /// <summary>The first index that a positive int32 cannot represent (2^31). A 32-bit truncation misbehaves at/after here.</summary>
    private const long Over = (long)int.MaxValue + 1; // 2,147,483,648

    /// <summary>The highest valid index (N-1), also &gt; int.MaxValue.</summary>
    private const long Last = N - 1; // 2,147,483,662

    private const long GB = 1024L * 1024 * 1024;

    // ────────────────────────────────────────────────────────────────────────────────────────
    // Bounded-output ops: results are scalars, bools, tiny index arrays, or one byte-wide array.
    // Peak ≈ input (2 GB) + one or two 2 GB byte/bool arrays ≈ 6 GB (mod divisor, greater, where).
    // Guard at 10 GB for headroom over the CLR + GC fragmentation.
    // ────────────────────────────────────────────────────────────────────────────────────────
    [TestMethod]
    public void ByteHugeArray_BoundedOutputOps_SupportBeyondIntMaxValue()
    {
        RequireAvailableMemory(10 * GB);
        var results = new List<(string Op, bool Ok, string Err)>();

        // ── creation ────────────────────────────────────────────────────────────────────────
        RunOp(results, "np.zeros", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            Assert.AreEqual(N, a.size);
            Assert.AreEqual(0, a.GetByte(Over));
            Assert.AreEqual(0, a.GetByte(Last));
        });
        RunOp(results, "np.ones", () =>
        {
            using var a = np.ones(new Shape(N), np.uint8);
            Assert.AreEqual(N, a.size);
            Assert.AreEqual(1, a.GetByte(Last));
        });
        RunOp(results, "np.full", () =>
        {
            using var a = np.full(new Shape(N), (byte)42, np.uint8);
            Assert.AreEqual(N, a.size);
            Assert.AreEqual(42, a.GetByte(Over));
            Assert.AreEqual(42, a.GetByte(Last));
        });
        RunOp(results, "np.empty", () =>
        {
            using var a = np.empty(new Shape(N), np.uint8);
            Assert.AreEqual(N, a.size); // value is uninitialized — size is the truncation signal
        });
        RunOp(results, "np.full_like", () =>
        {
            using var z = np.zeros(new Shape(N), np.uint8);
            using var a = np.full_like(z, (byte)99);
            Assert.AreEqual(N, a.size);
            Assert.AreEqual(99, a.GetByte(Last));
        });
        RunOp(results, "np.copy", () =>
        {
            using var z = np.full(new Shape(N), (byte)77, np.uint8);
            using var a = np.copy(z);
            Assert.AreEqual(N, a.size);
            Assert.AreEqual(77, a.GetByte(Last));
        });

        // ── element get/set past the boundary ────────────────────────────────────────────────
        RunOp(results, "GetByte/SetByte@>2^31", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)123, Over);
            a.SetByte((byte)200, Last);
            Assert.AreEqual(123, a.GetByte(Over));
            Assert.AreEqual(200, a.GetByte(Last));
        });

        // ── reductions (scalar out) ──────────────────────────────────────────────────────────
        RunOp(results, "np.sum", () =>
        {
            using var a = Pokes();               // 1 + 3 + 200 = 204 (two sentinels sit past 2^31)
            using var r = np.sum(a);
            Assert.AreEqual(0, r.ndim);
            // sum(uint8) promotes to a wide integer dtype; read dtype-agnostically (GetDouble would
            // REINTERPRET the raw bytes). A loop truncated at 2^31 would miss a[Over]/a[Last] -> 1, not 204.
            Assert.AreEqual(204.0, Convert.ToDouble(r.GetAtIndex(0)), 0.5);
        });
        RunOp(results, "np.prod", () =>
        {
            using var a = Pokes();               // zeros present -> 0
            using var r = np.prod(a);
            Assert.AreEqual(0.0, Convert.ToDouble(r.GetAtIndex(0)), 0.0);
        });
        RunOp(results, "np.max", () =>
        {
            using var a = Pokes();
            using var r = np.max(a);
            Assert.AreEqual(200, r.GetByte(0));
        });
        RunOp(results, "np.min", () =>
        {
            using var a = Pokes();
            using var r = np.min(a);
            Assert.AreEqual(0, r.GetByte(0));
        });
        RunOp(results, "np.mean", () =>
        {
            using var a = Pokes();
            using var r = np.mean(a);
            Assert.AreEqual(0, r.ndim);
            Assert.AreEqual(204.0 / N, r.GetDouble(0), 1e-6); // depends on the true 64-bit count
        });
        RunOp(results, "np.ptp", () =>
        {
            using var a = Pokes();
            using var r = np.ptp(a);
            Assert.AreEqual(200.0, Convert.ToDouble(r.GetAtIndex(0)), 0.0);
        });
        RunOp(results, "np.any", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)1, Last);            // sole nonzero past 2^31
            Assert.IsTrue(np.any(a));
        });
        RunOp(results, "np.all", () =>
        {
            using var a = np.ones(new Shape(N), np.uint8);
            a.SetByte((byte)0, Last);            // sole zero past 2^31
            Assert.IsFalse(np.all(a));
        });
        RunOp(results, "np.count_nonzero", () =>
        {
            using var a = Pokes();               // exactly 3 nonzeros, one at Over, one at Last
            Assert.AreEqual(3L, np.count_nonzero(a));
        });

        // ── arg / search: the RESULT is itself an index > int.MaxValue ────────────────────────
        RunOp(results, "np.argmax", () =>
        {
            using var a = Pokes();               // unique max (200) at Last
            Assert.AreEqual(Last, np.argmax(a));
        });
        RunOp(results, "np.argmin", () =>
        {
            using var a = np.full(new Shape(N), (byte)255, np.uint8);
            a.SetByte((byte)0, Last);            // unique min at Last
            Assert.AreEqual(Last, np.argmin(a));
        });
        RunOp(results, "np.nanargmax", () =>
        {
            using var a = Pokes();               // byte has no NaN -> plain argmax
            Assert.AreEqual(Last, np.nanargmax(a));
        });
        RunOp(results, "np.nonzero", () =>
        {
            using var a = Pokes();               // nonzeros at {0, Over, Last}
            var nz = np.nonzero(a);
            Assert.AreEqual(1, nz.Length);
            Assert.AreEqual(3L, nz[0].size);
            Assert.AreEqual(Over, nz[0].GetInt64(1));
            Assert.AreEqual(Last, nz[0].GetInt64(2));
        });
        RunOp(results, "np.flatnonzero", () =>
        {
            using var a = Pokes();
            var f = np.flatnonzero(a);
            Assert.AreEqual(3L, f.size);
            Assert.AreEqual(Last, f.GetInt64(2));
        });
        RunOp(results, "np.argwhere", () =>
        {
            using var a = Pokes();
            using var w = np.argwhere(a);        // (3, 1) int64
            Assert.AreEqual(Last, w.GetInt64(2, 0));
        });
        RunOp(results, "np.where(cond)", () =>
        {
            using var a = Pokes();
            var w = np.where(a);                 // 1-arg where == nonzero
            Assert.AreEqual(Last, w[0].GetInt64(2));
        });

        // ── element-wise (byte out; value checked past the boundary) ──────────────────────────
        RunOp(results, "np.add", () =>
        {
            using var a = Elem();                // a[Over]=20, a[Last]=10
            using var r = np.add(a, a);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(20, r.GetByte(Last));
            Assert.AreEqual(40, r.GetByte(Over));
        });
        RunOp(results, "np.subtract", () =>
        {
            using var a = Elem();
            using var r = np.subtract(a, a);
            Assert.AreEqual(0, r.GetByte(Last));
        });
        RunOp(results, "np.multiply", () =>
        {
            using var a = Elem();
            using var r = np.multiply(a, (byte)3);
            Assert.AreEqual(30, r.GetByte(Last));  // 10*3
        });
        RunOp(results, "np.square", () =>
        {
            using var a = Elem();
            using var r = np.square(a);
            Assert.AreEqual(100, r.GetByte(Last)); // 10^2
        });
        RunOp(results, "np.mod", () =>
        {
            using var a = Elem();
            using var b = np.full(new Shape(N), (byte)7, np.uint8); // NDArray divisor stays uint8
            using var r = np.mod(a, b);                             // (np.mod(NDArray,float) would promote to float)
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(3, r.GetByte(Last));   // 10 % 7
        });
        RunOp(results, "np.invert", () =>
        {
            using var a = Elem();
            using var r = np.invert(a);
            Assert.AreEqual(245, r.GetByte(Last)); // ~10 & 0xFF
        });
        RunOp(results, "np.left_shift", () =>
        {
            using var a = Elem();
            using var r = np.left_shift(a, 1);
            Assert.AreEqual(20, r.GetByte(Last));
        });
        RunOp(results, "np.right_shift", () =>
        {
            using var a = Elem();
            using var r = np.right_shift(a, 1);
            Assert.AreEqual(5, r.GetByte(Last));
        });
        RunOp(results, "np.clip", () =>
        {
            using var a = Elem();
            using var r = np.clip(a, (byte)0, (byte)5);
            Assert.AreEqual(5, r.GetByte(Last));   // 10 clipped to 5
        });
        RunOp(results, "np.maximum", () =>
        {
            using var a = Elem();
            using var b = np.full(new Shape(N), (byte)5, np.uint8);
            using var r = np.maximum(a, b);
            Assert.AreEqual(10, r.GetByte(Last));
        });
        RunOp(results, "np.minimum", () =>
        {
            using var a = Elem();
            using var b = np.full(new Shape(N), (byte)5, np.uint8);
            using var r = np.minimum(a, b);
            Assert.AreEqual(5, r.GetByte(Last));
        });
        RunOp(results, "bitwise & | ^", () =>
        {
            using var a = np.full(new Shape(N), (byte)0b1111_0000, np.uint8);
            using var b = np.full(new Shape(N), (byte)0b1010_1010, np.uint8);
            using var rAnd = a & b;
            using var rOr = a | b;
            using var rXor = a.TensorEngine.BitwiseXor(a, b);
            Assert.AreEqual(0b1010_0000, rAnd.GetByte(Last));
            Assert.AreEqual(0b1111_1010, rOr.GetByte(Last));
            Assert.AreEqual(0b0101_1010, rXor.GetByte(Last));
        });

        // ── comparison (bool out; value checked past the boundary) ────────────────────────────
        RunOp(results, "np.equal", () =>
        {
            using var a = Elem();
            using var r = np.equal(a, a);
            Assert.AreEqual(N, r.size);
            Assert.IsTrue(r.GetBoolean(Last));
        });
        RunOp(results, "np.greater", () =>
        {
            using var a = Elem();
            using var z = np.zeros(new Shape(N), np.uint8);
            using var r = np.greater(a, z);
            Assert.IsTrue(r.GetBoolean(Last));     // 10 > 0
        });

        // ── selection / ternary where ─────────────────────────────────────────────────────────
        RunOp(results, "np.where(cond,x,y)", () =>
        {
            using var mask = np.zeros(new Shape(N), NPTypeCode.Boolean);
            mask.SetBoolean(true, Last);
            using var r = np.where(mask, (byte)7, (byte)9); // scalar x/y broadcast -> no extra 2 GB arrays
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(7, r.GetByte(Last));   // mask true at Last -> x
            Assert.AreEqual(9, r.GetByte(Over));   // mask false -> y
        });
        RunOp(results, "np.take", () =>
        {
            using var a = Pokes();
            using var idx = np.array(new long[] { 0, Over, Last });
            using var r = np.take(a, idx);
            Assert.AreEqual(3L, r.size);
            Assert.AreEqual(3, r.GetByte(1));      // a[Over]
            Assert.AreEqual(200, r.GetByte(2));    // a[Last]
        });

        // ── shape / view (full-size out; value checked past the boundary) ─────────────────────
        RunOp(results, "np.reshape", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)5, Last);
            using var r = np.reshape(a, new Shape(1L, N));
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(2, r.ndim);
            Assert.AreEqual(5, r.GetByte(0, Last));
        });
        RunOp(results, "np.ravel", () =>
        {
            using var a = np.zeros(new Shape(1, N), np.uint8);
            a.SetByte((byte)5, 0, Last);
            using var r = np.ravel(a);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(5, r.GetByte(Last));
        });
        RunOp(results, "ndarray.flatten", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)5, Last);
            using var r = a.flatten();
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(5, r.GetByte(Last));
        });
        RunOp(results, "np.expand_dims", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            using var r = np.expand_dims(a, 0);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(2, r.ndim);
        });
        RunOp(results, "np.atleast_1d/2d/3d", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            using var r1 = np.atleast_1d(a);
            using var r2 = np.atleast_2d(a);
            using var r3 = np.atleast_3d(a);
            Assert.AreEqual(N, r1.size);
            Assert.AreEqual(N, r2.size);
            Assert.AreEqual(N, r3.size);
        });
        RunOp(results, "np.broadcast_to", () =>
        {
            using var one = np.full(new Shape(1L), (byte)77, np.uint8);
            using var r = np.broadcast_to(one, new Shape(N));
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(77, r.GetByte(Over));
            Assert.AreEqual(77, r.GetByte(Last));
        });
        RunOp(results, "np.roll", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)7, 0);
            using var r = np.roll(a, Over);        // element at 0 moves to index Over (> 2^31)
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(7, r.GetByte(Over));
        });
        RunOp(results, "np.flip", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)7, Last);
            using var r = np.flip(a);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(7, r.GetByte(0));      // a[Last] -> r[0]
        });

        // ── sort (byte out, full size) — flat/axis paths over > 2^31 elements ─────────────────
        RunOp(results, "np.sort", () =>
        {
            using var a = Pokes();                 // max 200 at Last, zeros elsewhere
            using var r = np.sort(a, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(0, r.GetByte(0));       // smallest first
            Assert.AreEqual(200, r.GetByte(Last));  // largest last
        });
        RunOp(results, "np.partition", () =>
        {
            using var a = Pokes();
            using var r = np.partition(a, new int[] { 0 }, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(0, r.GetByte(0));       // kth=0 -> smallest in place
        });

        Report(results, "BoundedOutputOps");
    }

    // ────────────────────────────────────────────────────────────────────────────────────────
    // Scan + cast wide-output ops: the RESULT is int64/double at FULL size (~17 GB) while the input
    // is a single 2 GB byte array, so peak ≈ 19 GB. These ride the standard long-based Direct
    // scan/cast kernels (loop counters + count + Shape.Vector all long — verified 64-bit end to
    // end), NOT the bespoke sort/select cores; confirmed on the real 2.1e9 array. Guard 24 GB.
    // ────────────────────────────────────────────────────────────────────────────────────────
    [TestMethod]
    public void ByteHugeArray_ScanAndCastOps_SupportBeyondIntMaxValue()
    {
        RequireAvailableMemory(24 * GB);
        var results = new List<(string Op, bool Ok, string Err)>();

        // np.cumsum flat — uint8 -> int64 running total at full size.
        RunOp(results, "np.cumsum(axis=null)", () =>
        {
            using var a = np.ones(new Shape(N), np.uint8);
            using var r = np.cumsum(a, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(1L, r.GetInt64(0));
            Assert.AreEqual(Over + 1, r.GetInt64(Over)); // running sum crosses 2^31 mid-array
            Assert.AreEqual(N, r.GetInt64(N - 1));       // sum of N ones = N (> 2^31)
        });

        // np.cumprod flat.
        RunOp(results, "np.cumprod(axis=null)", () =>
        {
            using var a = np.ones(new Shape(N), np.uint8);
            using var r = np.cumprod(a, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(1L, r.GetInt64(Over));
            Assert.AreEqual(1L, r.GetInt64(N - 1));
        });

        // astype to wider dtypes — full-size int64 / double output, value read back past 2^31.
        RunOp(results, "astype(int64)", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)123, Last);
            a.SetByte((byte)45, Over);
            using var r = a.astype(NPTypeCode.Int64);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(45L, r.GetInt64(Over));
            Assert.AreEqual(123L, r.GetInt64(Last));
        });
        RunOp(results, "astype(double)", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)123, Last);
            using var r = a.astype(NPTypeCode.Double);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(123.0, r.GetDouble(Last), 0.0);
        });

        Report(results, "ScanAndCastOps");
    }

    // ────────────────────────────────────────────────────────────────────────────────────────
    // argsort / argpartition over > 2^31 elements — the int64-INDEX-returning sort/select family.
    // Validates the 64-bit sort/select core (RadixSort/QuickSelect) AND the output-shape (int)a.size
    // fix. Their footprint is enormous: argsort(byte) widens keys to 4 bytes and carries two int64
    // columns, so peak ≈ 71 GB at 2.1e9 (int64 output 17 + key/tmp 17 + idx/it 34 + input 2). The
    // 72 GB guard reflects that — this runs only on a very-large-memory host. sort/partition (byte
    // output, ~4 GB) already prove the same cores on the real 2.1e9 array in BoundedOutputOps.
    // ────────────────────────────────────────────────────────────────────────────────────────
    [TestMethod]
    public void ByteHugeArray_ArgSortArgPartition_SupportBeyondIntMaxValue()
    {
        RequireAvailableMemory(72 * GB);
        var results = new List<(string Op, bool Ok, string Err)>();

        // np.argsort flat — validates AxisSort.ArgSort's Shape.Vector(a.size) (was new Shape((int)a.size)).
        RunOp(results, "np.argsort(axis=null)", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)200, Last);            // unique max
            using var r = np.argsort(a, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(Last, r.GetInt64(N - 1)); // ascending -> max's index last, and it is > 2^31
        });

        // np.argpartition flat, int[] kth — validates AxisPartition.ArgPartition (int[] overload).
        RunOp(results, "np.argpartition(int[],axis=null)", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)200, Last);
            using var r = np.argpartition(a, new int[] { 0 }, axis: null);
            Assert.AreEqual(N, r.size);            // shape must not truncate
        });

        // np.argpartition flat, NDArray kth (value > int.MaxValue) — validates the NDArray-kth overload.
        RunOp(results, "np.argpartition(kth>2^31,axis=null)", () =>
        {
            using var a = np.zeros(new Shape(N), np.uint8);
            a.SetByte((byte)200, Last);
            using var kth = np.array(new long[] { N - 1 });
            using var r = np.argpartition(a, kth, axis: null);
            Assert.AreEqual(N, r.size);
            Assert.AreEqual(Last, r.GetInt64(N - 1)); // largest partitioned into last position
        });

        Report(results, "ArgSortArgPartition");
    }

    // ── helpers ───────────────────────────────────────────────────────────────────────────────

    /// <summary>Zeros of size N with three nonzero sentinels: a[0]=1, a[Over]=3, a[Last]=200 (unique max).</summary>
    private static NDArray Pokes()
    {
        var a = np.zeros(new Shape(N), np.uint8);
        a.SetByte((byte)1, 0);
        a.SetByte((byte)3, Over);
        a.SetByte((byte)200, Last);
        return a;
    }

    /// <summary>Zeros of size N with a[Over]=20 and a[Last]=10, for element-wise value checks past the boundary.</summary>
    private static NDArray Elem()
    {
        var a = np.zeros(new Shape(N), np.uint8);
        a.SetByte((byte)20, Over);
        a.SetByte((byte)10, Last);
        return a;
    }

    private static void RunOp(List<(string Op, bool Ok, string Err)> results, string name, Action body)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            body();
            sw.Stop();
            Console.WriteLine($"  OK    {name} ({sw.ElapsedMilliseconds} ms)");
            results.Add((name, true, null));
        }
        catch (Microsoft.VisualStudio.TestTools.UnitTesting.AssertInconclusiveException)
        {
            throw; // never swallow an Inconclusive guard
        }
        catch (Exception ex)
        {
            sw.Stop();
            Console.WriteLine($"  FAIL  {name} ({sw.ElapsedMilliseconds} ms): {ex.GetType().Name}: {ex.Message}");
            results.Add((name, false, $"{ex.GetType().Name}: {ex.Message}"));
        }
        finally
        {
            ForceGC();
        }
    }

    private static void Report(List<(string Op, bool Ok, string Err)> results, string group)
    {
        int failed = results.Count(r => !r.Ok);
        Console.WriteLine($"\n=== {group}: {results.Count - failed}/{results.Count} support > int.MaxValue ===");
        foreach (var r in results.Where(r => !r.Ok))
            Console.WriteLine($"  NOT SUPPORTED: {r.Op} -> {r.Err}");
        Assert.AreEqual(0, failed,
            $"{failed} op(s) do NOT support > int.MaxValue byte arrays: "
            + string.Join(", ", results.Where(r => !r.Ok).Select(r => r.Op)));
    }

    /// <summary>Goes Inconclusive (rather than OOM) when the GC reports less usable memory than the peak the method needs.</summary>
    private static void RequireAvailableMemory(long bytes)
    {
        long avail = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (avail > 0 && avail < bytes)
            Assert.Inconclusive(
                $"Needs ~{bytes / (double)GB:F0} GB of usable memory; GC reports ~{avail / (double)GB:F1} GB. "
                + "Run on a host with more RAM to confirm > int.MaxValue support.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ForceGC()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
    }
}
