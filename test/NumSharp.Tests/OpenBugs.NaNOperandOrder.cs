using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in WHICH NaN an arithmetic result carries when both operands are NaN: the paths the float
    ///     add / multiply fix (commit 3169579a, 2026-10-03) left open. Each test asserts NumPy 2.4.2 win-amd64's
    ///     answer — dtype, shape and the raw bits of every element — so it FAILS while the bug lives and passes once
    ///     it is fixed. Remove <c>[OpenBugs]</c> from a test when its path is fixed, move its cases into the
    ///     <c>nan_order</c> oracle tier (<c>test/oracle/gen_oracle.py</c> <c>gen_nan_order</c>) and delete the test.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     When both inputs are NaN the result is one of the two NaNs, and WHICH one is a property of the compiled
    ///     loop: x86 <c>addpd</c> / <c>mulpd</c> return their first source operand (quieted), and a compiler may
    ///     swap the operands of a commutative operation. NumPy's MSVC build keeps the SECOND operand's NaN for a
    ///     float32 / float64 add or multiply on every loop path except its AVX2 scalar-broadcast loop (one input
    ///     with stride 0, the other input and the output contiguous, more than two vectors): there, blocks of two
    ///     vectors (8 float64 / 16 float32 elements) keep the BROADCAST operand's NaN and the scalar tail keeps the
    ///     ARRAY operand's (the "region rule"), restarting at every inner-loop call. NumSharp follows that rule since
    ///     3169579a (<c>DirectILKernelGenerator.NaNOperandOrder.cs</c>, gated by the <c>nan_order</c> tier and
    ///     <c>Math/NaNOperandOrderTests.cs</c>).
    ///     </para>
    ///     <para>
    ///     The cases below are where NumSharp still differs. Every candidate path was replayed against NumPy's raw
    ///     bits (a 458-case probe, 2026-10-03): the 272 that differ are grouped into the seven tests below, the 186
    ///     that match went into the <c>nan_order</c> tier (float16 scans, complex128 subtract, the buffered N-D
    ///     broadcasts, the cast and <c>where=</c> layouts that already split per call, float32 trees too short for a
    ///     region block, <c>dtype=</c> loops). Most of the open paths are cases where NumPy makes MORE inner-loop
    ///     calls than NumSharp's kernels do — per row, per buffer fill, per mask run, per tree node — so the region
    ///     rule restarts where NumSharp's single call keeps the second operand's NaN.
    ///     </para>
    ///     <para>
    ///     The oracle tiers compare NaN as one token except where a case carries <c>params.nan_bits</c>, so these
    ///     bits are visible only here and in <c>nan_order.jsonl</c>. Linux NumPy (GCC) orders the operands
    ///     differently on the same CPU, so like the tier these pins are a win-amd64 contract. <c>np.sum</c> /
    ///     <c>np.prod</c> are deliberately absent: their NaN choice follows the reduction order, an accepted
    ///     difference of NumSharp's multi-accumulator SIMD reductions.
    ///     </para>
    ///     <para>
    ///     Every expected string is NumPy 2.4.2's output for the same call (same operand bits, same layout, same
    ///     arguments), probed on 2026-10-03, in <see cref="Runs"/>'s format: <c>dtype (shape) tokens…</c>, one hex
    ///     token per element in logical C order (complex: <c>real:imaginary</c>), a run of identical tokens written
    ///     <c>token*count</c>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsNaNOperandOrderTests : TestClass
    {
        /// <summary>Operand a's float64 NaN: negative, payload 1.</summary>
        private const ulong A64 = 0xfff8000000000001;

        /// <summary>Operand b's float64 NaN: positive, payload 2.</summary>
        private const ulong B64 = 0x7ff8000000000002;

        /// <summary>Operand a's float32 NaN: negative, payload 1.</summary>
        private const ulong A32 = 0xffc00001;

        /// <summary>Operand b's float32 NaN: positive, payload 2.</summary>
        private const ulong B32 = 0x7fc00002;

        /// <summary>Operand a's complex128 parts: real and imaginary NaNs, both negative, payloads 0x01 / 0x11.</summary>
        private static readonly (ulong Re, ulong Im) CA = (0xfff8000000000001, 0xfff8000000000011);

        /// <summary>Operand b's complex128 parts: real and imaginary NaNs, both positive, payloads 0x02 / 0x22.</summary>
        private static readonly (ulong Re, ulong Im) CB = (0x7ff8000000000002, 0x7ff8000000000022);

        /// <summary>The float64 quiet-NaN base the scan inputs add their per-element payload to.</summary>
        private const ulong P64 = 0x7ff8000000000000;

        /// <summary>The float32 quiet-NaN base the scan inputs add their per-element payload to.</summary>
        private const ulong P32 = 0x7fc00000;

        // ======================================================================================================
        //  1. cumsum / cumprod keep the NEW element's NaN at every step
        // ======================================================================================================

        /// <summary>
        ///     <c>np.cumsum</c> / <c>np.cumprod</c> over NaNs that each carry their own payload must keep NumPy's NaN
        ///     at every step: for float32 / float64 the NEW element's (output i carries input i's payload, on every
        ///     layout and axis), for complex128 a per-component mix. NumSharp keeps the accumulator's NaN, so every
        ///     output after the first carries the first element's payload.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>add.accumulate</c> / <c>multiply.accumulate</c> call the binary inner loop as
        ///     <c>op(out[i-1], in[i])</c>, so the float32 / float64 rule of the elementwise loops (the second operand's
        ///     NaN) gives the new element's NaN. complex128 is MSVC's scalar complex loop: cumsum keeps the
        ///     accumulator's REAL NaN and the new element's IMAGINARY NaN; cumprod puts the new element's real NaN in
        ///     both parts. Output 0 is a COPY of input 0 in NumPy; NumSharp's complex cumprod multiplies it by the
        ///     identity instead, so even output 0 loses its imaginary payload (<c>…01:…01</c> for <c>…01:…101</c>).
        ///     </para>
        ///     <para>
        ///     Root cause: the scan kernels (<c>DirectILKernelGenerator.Scan.cs</c>) fold <c>acc op x</c> and the
        ///     hardware returns the first source's NaN. The fix is the scan counterpart of 3169579a: encode the order
        ///     explicitly (<c>NaNOperandOrder.NumPyUfunc</c> emitters) instead of trusting the IL operand order.
        ///     float16 scans already match on every probed layout and axis and are gated by <c>nan_order.jsonl</c>.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void CumsumCumprod_KeepNumPysNaNAtEveryStep()
        {
            var fails = new List<string>();
            Check(fails, "cumsum f64 (9,)", () => np.cumsum(Payloads(NPTypeCode.Double, P64, 9)),
                "float64 (9,) 7ff8000000000001 7ff8000000000002 7ff8000000000003 7ff8000000000004 7ff8000000000005 " +
                "7ff8000000000006 7ff8000000000007 7ff8000000000008 7ff8000000000009");
            Check(fails, "cumprod f64 (9,)", () => np.cumprod(Payloads(NPTypeCode.Double, P64, 9)),
                "float64 (9,) 7ff8000000000001 7ff8000000000002 7ff8000000000003 7ff8000000000004 7ff8000000000005 " +
                "7ff8000000000006 7ff8000000000007 7ff8000000000008 7ff8000000000009");
            Check(fails, "cumsum f32 (9,)", () => np.cumsum(Payloads(NPTypeCode.Single, P32, 9)),
                "float32 (9,) 7fc00001 7fc00002 7fc00003 7fc00004 7fc00005 7fc00006 7fc00007 7fc00008 7fc00009");
            Check(fails, "cumprod f32 (9,)", () => np.cumprod(Payloads(NPTypeCode.Single, P32, 9)),
                "float32 (9,) 7fc00001 7fc00002 7fc00003 7fc00004 7fc00005 7fc00006 7fc00007 7fc00008 7fc00009");
            Check(fails, "cumsum f64 (4,5) axis 0", () => np.cumsum(Payloads(NPTypeCode.Double, P64, 4, 5), 0),
                "float64 (4,5) 7ff8000000000001 7ff8000000000002 7ff8000000000003 7ff8000000000004 7ff8000000000005 " +
                "7ff8000000000006 7ff8000000000007 7ff8000000000008 7ff8000000000009 7ff800000000000a " +
                "7ff800000000000b 7ff800000000000c 7ff800000000000d 7ff800000000000e 7ff800000000000f " +
                "7ff8000000000010 7ff8000000000011 7ff8000000000012 7ff8000000000013 7ff8000000000014");
            Check(fails, "cumsum f64 (4,5) axis 1", () => np.cumsum(Payloads(NPTypeCode.Double, P64, 4, 5), 1),
                "float64 (4,5) 7ff8000000000001 7ff8000000000002 7ff8000000000003 7ff8000000000004 7ff8000000000005 " +
                "7ff8000000000006 7ff8000000000007 7ff8000000000008 7ff8000000000009 7ff800000000000a " +
                "7ff800000000000b 7ff800000000000c 7ff800000000000d 7ff800000000000e 7ff800000000000f " +
                "7ff8000000000010 7ff8000000000011 7ff8000000000012 7ff8000000000013 7ff8000000000014");
            Check(fails, "cumprod f64 (5,4).T axis 0", () => np.cumprod(Payloads(NPTypeCode.Double, P64, 5, 4).T, 0),
                "float64 (4,5) 7ff8000000000001 7ff8000000000005 7ff8000000000009 7ff800000000000d 7ff8000000000011 " +
                "7ff8000000000002 7ff8000000000006 7ff800000000000a 7ff800000000000e 7ff8000000000012 " +
                "7ff8000000000003 7ff8000000000007 7ff800000000000b 7ff800000000000f 7ff8000000000013 " +
                "7ff8000000000004 7ff8000000000008 7ff800000000000c 7ff8000000000010 7ff8000000000014");
            Check(fails, "cumsum f64 (9,)[::-1]", () => np.cumsum(Payloads(NPTypeCode.Double, P64, 9)["::-1"]),
                "float64 (9,) 7ff8000000000009 7ff8000000000008 7ff8000000000007 7ff8000000000006 7ff8000000000005 " +
                "7ff8000000000004 7ff8000000000003 7ff8000000000002 7ff8000000000001");
            Check(fails, "cumsum c128 (3,)", () => np.cumsum(ComplexPayloads(3)),
                "complex128 (3,) 7ff8000000000001:7ff8000000000101 7ff8000000000001:7ff8000000000102 " +
                "7ff8000000000001:7ff8000000000103");
            Check(fails, "cumprod c128 (3,)", () => np.cumprod(ComplexPayloads(3)),
                "complex128 (3,) 7ff8000000000001:7ff8000000000101 7ff8000000000002:7ff8000000000002 " +
                "7ff8000000000003:7ff8000000000003");
            AssertNoMismatch(fails, "NumPy's accumulate calls op(out[i-1], in[i]) and keeps that loop's NaN");
        }

        // ======================================================================================================
        //  2. complex128 add / multiply / divide
        // ======================================================================================================

        /// <summary>
        ///     complex128 <c>np.add</c> / <c>np.multiply</c> / <c>np.divide</c> of two all-NaN operands (a = negative
        ///     NaNs, b = positive NaNs, distinct payloads per part) must return NumPy's NaN in each part. NumSharp
        ///     differs on every layout for multiply and divide and on every array layout for add; complex128
        ///     subtract already matches (gated by <c>nan_order.jsonl</c>).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy (probed): add keeps b's NaN in both parts for contiguous elements 0-1, a's for 2-3, b's for 4-5, …
        ///     in MEMORY order — MSVC's 2×-unrolled complex loop alternates the operand order by vector pair — and the
        ///     scalar-broadcast loop keeps the broadcast operand's NaN; multiply returns a's REAL NaN in both parts;
        ///     divide returns a's real NaN in the real part and b's real NaN in the imaginary part, also for a 0-d
        ///     pair and a strided pair.
        ///     </para>
        ///     <para>
        ///     Root cause: the complex kernels compute through <c>Complex.op_Addition</c>,
        ///     <c>NDComplexMath.Multiply</c> (fused like NumPy's <c>simd_cmul</c>, finite bits exact) and
        ///     <c>ComplexDivideNumPy</c> (Smith, finite bits exact); their NaN choice is whatever the emitted
        ///     instruction order gives. A fix keeps those finite formulations and selects NumPy's NaN on the cold
        ///     path where a part is NaN.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Complex128_AddMultiplyDivide_KeepNumPysNaNPerPart()
        {
            var fails = new List<string>();
            Check(fails, "add (6,)", () => np.add(ComplexBits(CA, 6), ComplexBits(CB, 6)),
                "complex128 (6,) 7ff8000000000002:7ff8000000000022*2 fff8000000000001:fff8000000000011*2 " +
                "7ff8000000000002:7ff8000000000022*2");
            Check(fails, "add (17,)", () => np.add(ComplexBits(CA, 17), ComplexBits(CB, 17)),
                "complex128 (17,) 7ff8000000000002:7ff8000000000022*2 fff8000000000001:fff8000000000011*2 " +
                "7ff8000000000002:7ff8000000000022*2 fff8000000000001:fff8000000000011*2 " +
                "7ff8000000000002:7ff8000000000022*2 fff8000000000001:fff8000000000011*2 " +
                "7ff8000000000002:7ff8000000000022*2 fff8000000000001:fff8000000000011*2 " +
                "7ff8000000000002:7ff8000000000022");
            Check(fails, "add (5,4).T", () => np.add(ComplexBits(CA, 5, 4).T, ComplexBits(CB, 5, 4).T),
                "complex128 (4,5) 7ff8000000000002:7ff8000000000022*10 fff8000000000001:fff8000000000011*10");
            Check(fails, "add (9,) + 0-d", () => np.add(ComplexBits(CA, 9), ComplexBits(CB)),
                "complex128 (9,) 7ff8000000000002:7ff8000000000022*9");
            Check(fails, "multiply (6,)", () => np.multiply(ComplexBits(CA, 6), ComplexBits(CB, 6)),
                "complex128 (6,) fff8000000000001:fff8000000000001*6");
            Check(fails, "multiply 0-d x 0-d", () => np.multiply(ComplexBits(CA), ComplexBits(CB)),
                "complex128 () fff8000000000001:fff8000000000001");
            Check(fails, "multiply 0-d x (9,)", () => np.multiply(ComplexBits(CA), ComplexBits(CB, 9)),
                "complex128 (9,) fff8000000000001:fff8000000000001*9");
            Check(fails, "divide (6,)", () => np.divide(ComplexBits(CA, 6), ComplexBits(CB, 6)),
                "complex128 (6,) fff8000000000001:7ff8000000000002*6");
            Check(fails, "divide 0-d x 0-d", () => np.divide(ComplexBits(CA), ComplexBits(CB)),
                "complex128 () fff8000000000001:7ff8000000000002");
            Check(fails, "divide (18,)[::2]", () => np.divide(ComplexBits(CA, 18)["::2"], ComplexBits(CB, 18)["::2"]),
                "complex128 (9,) fff8000000000001:7ff8000000000002*9");
            AssertNoMismatch(fails, "NumPy's MSVC complex loops choose the NaN per part and per unrolled vector pair");
        }

        // ======================================================================================================
        //  3. N-D broadcasts NumPy runs unbuffered: the region rule per row
        // ======================================================================================================

        /// <summary>
        ///     Some N-D broadcasts reach NumPy's scalar-broadcast loop once per ROW — an outer product with two rows,
        ///     <c>(2,1) op (1,n)</c>, and a <c>(m,n)</c> / <c>(m,1)</c> pair whose rows are longer than half the
        ///     buffer (<c>n &gt; 4096</c>) — so every row gets the region rule. NumSharp keeps the second operand's
        ///     NaN on every element.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's buffering setup leaves these pairs unbuffered and hands the inner loop one row at a time: the
        ///     row's broadcast operand has stride 0 and the other operand and the output are contiguous, which is
        ///     exactly the AVX2 scalar-broadcast loop. Each row splits into blocks (the broadcast operand's NaN) and a
        ///     tail (the array operand's), so <c>(2,1)+(1,9)</c> is <c>a*8 b</c> per row and a row length that is a
        ///     multiple of the block (<c>(1,64)</c>) is all blocks. The probe found the same shapes with three rows
        ///     (<c>(3,1)+(1,64)</c>) and with rows of at most 4096 elements already matching: NumPy buffers those
        ///     into one contiguous call, which NumSharp reproduces.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>DefaultEngine.NumPyBroadcastNaNRule</c> enables the split only when the whole iteration
        ///     coalesces to one 1-D run; the N-D kernel then keeps the second operand's NaN. A fix needs NumPy's
        ///     buffering decision (<c>npyiter_find_buffering_setup</c>) to know which pairs run unbuffered per row.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void UnbufferedNDBroadcast_RunsTheRegionRulePerRow()
        {
            var fails = new List<string>();
            Check(fails, "add f64 (2,1)+(1,64)",
                () => np.add(Bits(NPTypeCode.Double, A64, 2, 1), Bits(NPTypeCode.Double, B64, 1, 64)),
                "float64 (2,64) fff8000000000001*128");
            Check(fails, "multiply f32 (2,1)+(1,64)",
                () => np.multiply(Bits(NPTypeCode.Single, A32, 2, 1), Bits(NPTypeCode.Single, B32, 1, 64)),
                "float32 (2,64) ffc00001*128");
            Check(fails, "add f64 (2,1)+(1,9)",
                () => np.add(Bits(NPTypeCode.Double, A64, 2, 1), Bits(NPTypeCode.Double, B64, 1, 9)),
                "float64 (2,9) fff8000000000001*8 7ff8000000000002 fff8000000000001*8 7ff8000000000002");
            Check(fails, "multiply f64 (2,1)+(1,17)",
                () => np.multiply(Bits(NPTypeCode.Double, A64, 2, 1), Bits(NPTypeCode.Double, B64, 1, 17)),
                "float64 (2,17) fff8000000000001*16 7ff8000000000002 fff8000000000001*16 7ff8000000000002");
            Check(fails, "add f32 (2,5000)+(2,1)",
                () => np.add(Bits(NPTypeCode.Single, A32, 2, 5000), Bits(NPTypeCode.Single, B32, 2, 1)),
                "float32 (2,5000) 7fc00002*4992 ffc00001*8 7fc00002*4992 ffc00001*8");
            Check(fails, "add f64 (3,4097)+(3,1)",
                () => np.add(Bits(NPTypeCode.Double, A64, 3, 4097), Bits(NPTypeCode.Double, B64, 3, 1)),
                "float64 (3,4097) 7ff8000000000002*4096 fff8000000000001 7ff8000000000002*4096 fff8000000000001 " +
                "7ff8000000000002*4096 fff8000000000001");
            Check(fails, "multiply f64 (3,1)+(3,5000)",
                () => np.multiply(Bits(NPTypeCode.Double, A64, 3, 1), Bits(NPTypeCode.Double, B64, 3, 5000)),
                "float64 (3,5000) fff8000000000001*15000");
            AssertNoMismatch(fails, "NumPy runs these broadcasts one row per inner-loop call, the region rule per row");
        }

        // ======================================================================================================
        //  4. A 0-d against an N-D view NumPy buffers: the region rule per buffer fill
        // ======================================================================================================

        /// <summary>
        ///     A 0-d NaN against an N-D view that does not coalesce to one run — a row slice <c>x[:, :k]</c>, reversed
        ///     rows or columns, a 3-D slice — runs NumPy's scalar-broadcast loop once per BUFFER FILL, so the region
        ///     rule splits each fill. NumSharp keeps the second operand's NaN on every element.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy copies the view into its buffer in iteration order — whole rows, up to <c>np.getbufsize()</c> =
        ///     8192 elements a fill — and calls the inner loop once per fill with the 0-d operand at stride 0: a
        ///     <c>(6,9)</c> view is one fill of 54 (<c>b*48 a*6</c>), the <c>(100,100)</c> slice of a <c>(100,128)</c>
        ///     array two fills of 81 and 19 rows (8100 = 8096 + 4, 1900 = 1896 + 4). A view that coalesces (a column
        ///     stride <c>x[:, ::2]</c>), a C-contiguous array and an F-contiguous one already match and are gated by
        ///     the tier.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>DefaultEngine.NumPyBroadcastNaNRule</c> splits only a coalesced 1-D run; the N-D
        ///     broadcast kernel walks the view in place and has no buffer fills to restart the rule at. A fix splits
        ///     at NumPy's fill boundaries (whole rows per <c>np.getbufsize()</c>).
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void ZeroDimAgainstABufferedNDView_RunsTheRegionRulePerBufferFill()
        {
            var fails = new List<string>();
            Check(fails, "add f64 (6,20)[:, :9] + 0-d",
                () => np.add(Bits(NPTypeCode.Double, A64, 6, 20)[":, :9"], Bits(NPTypeCode.Double, B64)),
                "float64 (6,9) 7ff8000000000002*48 fff8000000000001*6");
            Check(fails, "add f64 0-d + (6,20)[:, :9]",
                () => np.add(Bits(NPTypeCode.Double, A64), Bits(NPTypeCode.Double, B64, 6, 20)[":, :9"]),
                "float64 (6,9) fff8000000000001*48 7ff8000000000002*6");
            Check(fails, "add f32 (6,20)[:, :17] + 0-d",
                () => np.add(Bits(NPTypeCode.Single, A32, 6, 20)[":, :17"], Bits(NPTypeCode.Single, B32)),
                "float32 (6,17) 7fc00002*96 ffc00001*6");
            Check(fails, "multiply f64 (6,50)[::-1] + 0-d",
                () => np.multiply(Bits(NPTypeCode.Double, A64, 6, 50)["::-1, :"], Bits(NPTypeCode.Double, B64)),
                "float64 (6,50) 7ff8000000000002*296 fff8000000000001*4");
            Check(fails, "multiply f32 0-d + (6,50)[:, ::-1]",
                () => np.multiply(Bits(NPTypeCode.Single, A32), Bits(NPTypeCode.Single, B32, 6, 50)[":, ::-1"]),
                "float32 (6,50) ffc00001*288 7fc00002*12");
            Check(fails, "add f64 (3,4,20)[:, :, :17] + 0-d",
                () => np.add(Bits(NPTypeCode.Double, A64, 3, 4, 20)[":, :, :17"], Bits(NPTypeCode.Double, B64)),
                "float64 (3,4,17) 7ff8000000000002*200 fff8000000000001*4");
            Check(fails, "add f64 0-d + (100,128)[:, :100]",
                () => np.add(Bits(NPTypeCode.Double, A64), Bits(NPTypeCode.Double, B64, 100, 128)[":, :100"]),
                "float64 (100,100) fff8000000000001*8096 7ff8000000000002*4 fff8000000000001*1896 7ff8000000000002*4");
            AssertNoMismatch(fails, "NumPy buffers the view and runs the broadcast loop once per buffer fill");
        }

        // ======================================================================================================
        //  5. A reversed cast operand: the region rule per buffer fill
        // ======================================================================================================

        /// <summary>
        ///     A reversed (negative-stride) float32 array against a size-1 float64 NaN is cast through NumPy's buffer,
        ///     and the scalar-broadcast loop runs once per buffer fill, so the region rule applies — from 9 elements
        ///     on, not only past one buffer. NumSharp keeps the second operand's NaN.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy (probed): <c>(9,)[::-1] + (1,)</c> is one fill (<c>b*8 a</c>, the tail at the end of the logical
        ///     order), 8203 elements are fills of 8192 and 11, and with the size-1 operand first (the broadcast
        ///     operand's NaN in the blocks) 9000 elements are blocks only (8192 and 808 are multiples of 8). The
        ///     forward layouts (contiguous, strided) already split per fill in NumSharp and are gated by the tier; so is
        ///     <c>f32 (9000,)[::-1] + f64 (1,)</c>, which matches only because its block and second-operand answers
        ///     coincide.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>DefaultEngine.NumPyBroadcastNaNRule</c> declines an operand with a flipped
        ///     (negative-stride) axis, so the cast route keeps the second operand's NaN. A fix lets a cast copy of a
        ///     reversed operand take the same per-fill split as a forward one.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void ReversedCastOperand_RunsTheRegionRulePerBufferFill()
        {
            var fails = new List<string>();
            Check(fails, "add f32 (9,)[::-1] + f64 (1,)",
                () => np.add(Bits(NPTypeCode.Single, A32, 9)["::-1"], Bits(NPTypeCode.Double, B64, 1)),
                "float64 (9,) 7ff8000000000002*8 fff8000020000000");
            Check(fails, "add f64 (1,) + f32 (9,)[::-1]",
                () => np.add(Bits(NPTypeCode.Double, A64, 1), Bits(NPTypeCode.Single, B32, 9)["::-1"]),
                "float64 (9,) fff8000000000001*8 7ff8000040000000");
            Check(fails, "multiply f32 (8203,)[::-1] + f64 (1,)",
                () => np.multiply(Bits(NPTypeCode.Single, A32, 8203)["::-1"], Bits(NPTypeCode.Double, B64, 1)),
                "float64 (8203,) 7ff8000000000002*8200 fff8000020000000*3");
            Check(fails, "add f64 (1,) + f32 (9000,)[::-1]",
                () => np.add(Bits(NPTypeCode.Double, A64, 1), Bits(NPTypeCode.Single, B32, 9000)["::-1"]),
                "float64 (9000,) fff8000000000001*9000");
            AssertNoMismatch(fails, "NumPy casts the reversed operand through its buffer and splits every fill");
        }

        // ======================================================================================================
        //  6. where= together with a cast: the region rule per mask run
        // ======================================================================================================

        /// <summary>
        ///     <c>where=</c> together with a cast (a float32 operand into a float64 loop, or a float64 loop into a
        ///     float32 <c>out</c>) runs NumPy's inner loop once per run of True mask elements, so the region rule
        ///     restarts per run exactly as it does without the cast. NumSharp keeps the second operand's NaN.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The mask has True runs of 12, 1, 20 and 4 elements (<see cref="RunsMask"/>): NumPy splits the run of 12
        ///     into 8 + 4 and the run of 20 into 16 + 4, and runs of 1 and 4 never reach the SIMD loop (the second
        ///     operand's NaN); masked-off slots keep the out array's prior zeros. A float64 NaN narrowed into a float32
        ///     <c>out</c> keeps only its sign (<c>7fc00000</c> / <c>ffc00000</c>). An all-True mask over 9000 elements
        ///     with the size-1 operand first gives blocks only (fills of 8192 and 808).
        ///     </para>
        ///     <para>
        ///     Root cause: <c>DefaultEngine.NumPyBroadcastNaNRule</c> splits per mask run or per buffer chunk, but not
        ///     when <c>where=</c> and a cast meet; that route then keeps the second operand's NaN.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void WhereWithACast_RunsTheRegionRulePerMaskRun()
        {
            var fails = new List<string>();
            Check(fails, "add f32 (40,) + f64 (1,), out f64, where runs", () =>
                {
                    var o = np.zeros(new Shape(40), NPTypeCode.Double);
                    np.add(Bits(NPTypeCode.Single, A32, 40), Bits(NPTypeCode.Double, B64, 1), o, RunsMask(40));
                    return o;
                },
                "float64 (40,) 7ff8000000000002*8 fff8000020000000*4 0000000000000000 7ff8000000000002 " +
                "0000000000000000 7ff8000000000002*16 fff8000020000000*4 0000000000000000 7ff8000000000002*4");
            Check(fails, "multiply f64 (1,) + f32 (40,), out f64, where runs", () =>
                {
                    var o = np.zeros(new Shape(40), NPTypeCode.Double);
                    np.multiply(Bits(NPTypeCode.Double, A64, 1), Bits(NPTypeCode.Single, B32, 40), o, RunsMask(40));
                    return o;
                },
                "float64 (40,) fff8000000000001*8 7ff8000040000000*4 0000000000000000 7ff8000040000000 " +
                "0000000000000000 fff8000000000001*16 7ff8000040000000*4 0000000000000000 7ff8000040000000*4");
            Check(fails, "add f64 (40,) + f64 0-d, out f32, where runs", () =>
                {
                    var o = np.zeros(new Shape(40), NPTypeCode.Single);
                    np.add(Bits(NPTypeCode.Double, A64, 40), Bits(NPTypeCode.Double, B64), o, RunsMask(40));
                    return o;
                },
                "float32 (40,) 7fc00000*8 ffc00000*4 00000000 7fc00000 00000000 7fc00000*16 ffc00000*4 00000000 " +
                "7fc00000*4");
            Check(fails, "add f64 (1,) + f32 (9000,), out f64, where all", () =>
                {
                    var o = np.zeros(new Shape(9000), NPTypeCode.Double);
                    np.add(Bits(NPTypeCode.Double, A64, 1), Bits(NPTypeCode.Single, B32, 9000), o, np.ones(new Shape(9000), NPTypeCode.Boolean));
                    return o;
                },
                "float64 (9000,) fff8000000000001*9000");
            AssertNoMismatch(fails, "NumPy calls the masked loop once per True run, cast or not");
        }

        // ======================================================================================================
        //  7. np.evaluate trees deeper than one node with a NaN broadcast input
        // ======================================================================================================

        /// <summary>
        ///     An <c>np.evaluate</c> tree deeper than one node whose broadcast input (a 0-d or a size-1 array) is NaN
        ///     must give NumPy's node-by-node answer: the node that meets the broadcast input runs NumPy's
        ///     scalar-broadcast loop and its region rule. NumSharp's fused kernel keeps that node's second operand's
        ///     NaN.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy evaluates <c>(a * k) + c</c> as <c>np.add(np.multiply(a, k), c)</c>: with k a NaN 0-d, the
        ///     multiply's blocks keep k's NaN and its tail a's, and the add with a finite <c>c</c> keeps that NaN. A
        ///     float32 tree of 9 elements has no region block (16 lanes) and already matches; it is gated by the tier.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>DefaultEngine.Evaluate</c> delegates a SINGLE node with a 0-d operand to the direct
        ///     broadcast kernel (its region split), but a deeper tree runs one fused kernel whose nodes follow the
        ///     second-operand rule. A fix carries the region split into the fused kernel for the nodes that read a
        ///     broadcast input.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void EvaluateTreeWithANaNBroadcastInput_RunsTheRegionRuleInThatNode()
        {
            var fails = new List<string>();
            Check(fails, "f64 (a*k)+c, k 0-d, n=9", () =>
                {
                    NDExpr a = Bits(NPTypeCode.Double, A64, 9), k = Bits(NPTypeCode.Double, B64), c = Twos(NPTypeCode.Double, 9);
                    return np.evaluate(a * k + c);
                },
                "float64 (9,) 7ff8000000000002*8 fff8000000000001");
            Check(fails, "f64 (k+a)*c, k (1,), n=17", () =>
                {
                    NDExpr a = Bits(NPTypeCode.Double, A64, 17), k = Bits(NPTypeCode.Double, B64, 1), c = Twos(NPTypeCode.Double, 17);
                    return np.evaluate((k + a) * c);
                },
                "float64 (17,) 7ff8000000000002*16 fff8000000000001");
            Check(fails, "f32 (a+k)-c, k 0-d, n=33", () =>
                {
                    NDExpr a = Bits(NPTypeCode.Single, A32, 33), k = Bits(NPTypeCode.Single, B32), c = Twos(NPTypeCode.Single, 33);
                    return np.evaluate(a + k - c);
                },
                "float32 (33,) 7fc00002*32 ffc00001");
            Check(fails, "f32 (k*a)+c, k (1,), n=17", () =>
                {
                    NDExpr a = Bits(NPTypeCode.Single, A32, 17), k = Bits(NPTypeCode.Single, B32, 1), c = Twos(NPTypeCode.Single, 17);
                    return np.evaluate(k * a + c);
                },
                "float32 (17,) 7fc00002*16 ffc00001");
            AssertNoMismatch(fails, "NumPy evaluates the tree node by node; the broadcast node runs the region rule");
        }

        // ======================================================================================================
        //  Helpers
        // ======================================================================================================

        /// <summary>
        ///     Builds a fresh C-contiguous array whose every element holds the raw <paramref name="bits"/>.
        /// </summary>
        /// <param name="type">The element type: Half, Single or Double.</param>
        /// <param name="bits">The raw element bits, truncated to the element width.</param>
        /// <param name="shape">The dimensions; none gives a 0-d array.</param>
        /// <returns>The new array.</returns>
        /// <remarks>
        ///     Writes through a pointer, never through a float value, so the NaN payload and sign reach the array
        ///     exactly as given (a float store can quiet a signalling NaN).
        /// </remarks>
        private static unsafe NDArray Bits(NPTypeCode type, ulong bits, params long[] shape)
        {
            var nd = np.zeros(new Shape(shape), type);
            byte* p = (byte*)nd.Address;
            for (long i = 0; i < nd.size; i++)
            {
                switch (nd.itemsize)
                {
                    case 8: ((ulong*)p)[i] = bits; break;
                    case 4: ((uint*)p)[i] = (uint)bits; break;
                    default: ((ushort*)p)[i] = (ushort)bits; break;
                }
            }

            return nd;
        }

        /// <summary>
        ///     Builds a fresh C-contiguous float array whose element <c>i</c> (in C order) holds the raw bits
        ///     <c>baseBits + i + 1</c>: with a quiet-NaN base, every element is a NaN with its own payload, so each
        ///     output element of a scan names the input element whose NaN it carries.
        /// </summary>
        /// <param name="type">The element type: Single or Double.</param>
        /// <param name="baseBits">The raw bits the payload counter is added to (a quiet NaN with payload 0).</param>
        /// <param name="shape">The dimensions.</param>
        /// <returns>The new array.</returns>
        private static unsafe NDArray Payloads(NPTypeCode type, ulong baseBits, params long[] shape)
        {
            var nd = np.zeros(new Shape(shape), type);
            byte* p = (byte*)nd.Address;
            for (long i = 0; i < nd.size; i++)
            {
                ulong v = baseBits + (ulong)i + 1;
                if (nd.itemsize == 8)
                    ((ulong*)p)[i] = v;
                else
                    ((uint*)p)[i] = (uint)v;
            }

            return nd;
        }

        /// <summary>
        ///     Builds a fresh C-contiguous complex128 array whose every element holds the raw real and imaginary bits
        ///     of <paramref name="parts"/>.
        /// </summary>
        /// <param name="parts">The raw bits of the real part and of the imaginary part.</param>
        /// <param name="shape">The dimensions; none gives a 0-d array.</param>
        /// <returns>The new array.</returns>
        private static unsafe NDArray ComplexBits((ulong Re, ulong Im) parts, params long[] shape)
        {
            var nd = np.zeros(new Shape(shape), NPTypeCode.Complex);
            ulong* p = (ulong*)nd.Address;
            for (long i = 0; i < nd.size; i++)
            {
                p[2 * i] = parts.Re;
                p[2 * i + 1] = parts.Im;
            }

            return nd;
        }

        /// <summary>
        ///     Builds a fresh C-contiguous complex128 array whose element <c>i</c> holds a real NaN with payload
        ///     <c>i + 1</c> and an imaginary NaN with payload <c>0x100 + i + 1</c> (both positive), so every output
        ///     part of a scan names the input part whose NaN it carries.
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <returns>The new array.</returns>
        private static unsafe NDArray ComplexPayloads(params long[] shape)
        {
            var nd = np.zeros(new Shape(shape), NPTypeCode.Complex);
            ulong* p = (ulong*)nd.Address;
            for (long i = 0; i < nd.size; i++)
            {
                p[2 * i] = P64 + (ulong)i + 1;
                p[2 * i + 1] = P64 + 0x100 + (ulong)i + 1;
            }

            return nd;
        }

        /// <summary>Builds a fresh 1-D array of <paramref name="n"/> elements, each the finite value 2.0.</summary>
        /// <param name="type">The element type: Single or Double.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The new array; a finite operand passes the other operand's NaN through unchanged.</returns>
        private static NDArray Twos(NPTypeCode type, long n) => np.full(new Shape(n), 2.0, type);

        /// <summary>
        ///     The <c>where=</c> mask the NumPy probe used: True runs of 12, 1, 20 and <c>n - 36</c> elements
        ///     (<c>[0,12)</c>, <c>[13,14)</c>, <c>[15,35)</c>, <c>[36,n)</c>), separated by single False slots.
        /// </summary>
        /// <param name="n">The mask length; at least 37.</param>
        /// <returns>A fresh 1-D bool array.</returns>
        private static NDArray RunsMask(int n)
        {
            var runs = new bool[n];
            foreach (var (start, stop) in new[] { (0, 12), (13, 14), (15, 35), (36, n) })
                for (int i = start; i < stop; i++)
                    runs[i] = true;
            return np.array(runs);
        }

        /// <summary>
        ///     Runs one case and records a line when its <see cref="Runs"/> string differs from NumPy's. An exception
        ///     is recorded as a mismatch too, so one throwing case cannot hide the state of the others in the same
        ///     test — a partial fix then shows exactly which cases still differ.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test.</param>
        /// <param name="expected">NumPy 2.4.2's result in <see cref="Runs"/>'s format.</param>
        private static void Check(List<string> mismatches, string name, Func<NDArray> call, string expected)
        {
            string actual;
            try
            {
                actual = Runs(call());
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            if (actual != expected)
                mismatches.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>
        ///     Fails the test when <see cref="Check"/> recorded any mismatch, listing EVERY recorded line, so a partial
        ///     fix shows how much of a path it already repaired.
        /// </summary>
        /// <param name="mismatches">The lines <see cref="Check"/> collected; empty means every case matched NumPy.</param>
        /// <param name="because">The NumPy rule the cases check, shown in the failure message.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, when <paramref name="mismatches"/>
        /// is not empty.</exception>
        private static void AssertNoMismatch(List<string> mismatches, string because)
        {
            string.Join(Environment.NewLine, mismatches).Should().BeEmpty(because);
        }

        /// <summary>
        ///     Renders a result as <c>dtype (shape) tokens…</c>: the NumPy dtype name, the shape in tuple form
        ///     (<c>()</c>, <c>(9,)</c>, <c>(2,64)</c>), then one token per element in logical C order — the raw bits in
        ///     hex (float64 16 digits, float32 8, float16 4; complex128 <c>real:imaginary</c>) — with a run of
        ///     identical tokens written <c>token*count</c>. Raw bits are the point: a value comparison, and every
        ///     ordinary oracle tier, treats all NaNs as equal.
        /// </summary>
        /// <param name="r">The result to render: Half, Single, Double or Complex.</param>
        /// <returns>The rendering.</returns>
        /// <exception cref="NotSupportedException">When <paramref name="r"/> has another dtype.</exception>
        private static unsafe string Runs(NDArray r)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // A C-order copy walked through its pointer: logical C order for every layout, and no float load or
            // store that could touch a NaN's bits. A 0-d result has one element, so a reshape view of it is enough;
            // the offset term covers a view that does not start at its buffer's first element.
            using var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            byte* p = (byte*)flat.Address + flat.Shape.offset * flat.itemsize;
            var tokens = new string[flat.size];
            for (long i = 0; i < flat.size; i++)
            {
                tokens[i] = flat.typecode switch
                {
                    NPTypeCode.Complex => $"{((ulong*)p)[2 * i]:x16}:{((ulong*)p)[2 * i + 1]:x16}",
                    NPTypeCode.Double => ((ulong*)p)[i].ToString("x16"),
                    NPTypeCode.Single => ((uint*)p)[i].ToString("x8"),
                    NPTypeCode.Half => ((ushort*)p)[i].ToString("x4"),
                    _ => throw new NotSupportedException($"Runs renders float and complex results only, not {flat.typecode}."),
                };
            }

            var sb = new StringBuilder($"{r.dtype.name} {shape}");
            for (long i = 0; i < tokens.Length;)
            {
                long j = i;
                while (j < tokens.Length && tokens[j] == tokens[i])
                    j++;
                sb.Append(' ').Append(tokens[i]);
                if (j - i > 1)
                    sb.Append('*').Append(j - i);
                i = j;
            }

            return sb.ToString();
        }
    }
}
