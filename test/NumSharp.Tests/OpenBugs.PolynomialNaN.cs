using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in WHICH NaN the <c>numpy.polynomial</c> integral kernels carry, element by element, when a
    ///     computation produces a NaN (an <c>inf - inf</c> in low precision, a NaN in the integration constants, or
    ///     a non-finite scale). Each test asserts NumPy 2.4.2 win-amd64's answer as the raw bits of every element, so
    ///     it FAILS while the bug lives and passes once the package's own IL step-table kernels pin NumPy's NaN order
    ///     per statement. Remove <c>[OpenBugs]</c> from a test when its path is fixed; the cases then belong in the
    ///     <c>polycalc</c> oracle tier with <c>params.nan_bits</c> (<c>test/oracle/gen_oracle.py</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The library-wide float add / multiply fix (commit 3169579a, 2026-10-03,
    ///     <c>DirectILKernelGenerator.NaNOperandOrder.cs</c>) made the HOUSE kernels keep NumPy's operand's NaN when
    ///     both operands are NaN — the second operand for an array-array ufunc, the first for subtract / divide, the
    ///     broadcast-block rule for a scalar-broadcast loop. The <c>numpy.polynomial</c> package does NOT reuse those
    ///     emitters for its own arithmetic: U3's evaluation (<c>ILKernelGenerator.Polynomial.Emitter.cs</c>) and U4's
    ///     calculus (<c>DirectILKernelGenerator.PolyCalculus.cs</c>) emit their recurrences from per-basis STEP TABLES
    ///     whose binary operations trust the IL operand order, which RyuJIT may swap for a commutative op. So where a
    ///     recurrence step meets two NaNs, the package returns the operand the JIT happened to put first, not the one
    ///     NumPy's statement keeps.
    ///     </para>
    ///     <para>
    ///     The value is a NaN either way — every ordinary oracle tier (and <c>np.allclose</c>) treats all NaNs as one
    ///     token — so these divergences are visible only at the raw-bit level, which is why they are pinned here rather
    ///     than gated by a differential-fuzz tier. They are a win-amd64 contract: NumPy's choice is a property of its
    ///     MSVC build, and a Linux (GCC) NumPy orders the same operands differently on the same CPU.
    ///     </para>
    ///     <para>
    ///     The three cases below were found by a raw-NaN replay of the <c>numpy.polynomial</c> fuzz corpus against
    ///     NumPy 2.4.2 (2026-10-05, plan <c>docs/plans/numpy-polynomial.md</c> section 12). Each is faithful to a
    ///     corpus case (same element bits, same layout, same arguments) and was confirmed to diverge on the build this
    ///     file ships with; each exercises a different NaN source — a float16 integral that OVERFLOWS to
    ///     <c>inf - inf</c>, an integral whose integration-constant vector CONTAINS a NaN, and an integral whose SCALE
    ///     is <c>+inf</c> — across both integral functions (<c>lagint</c>, <c>hermeint</c>) and two dtypes
    ///     (float16, float64).
    ///     </para>
    ///     <para>
    ///     The <c>scl</c> sign is load-bearing and a footgun: <c>lagint(c, scl=nan)</c> with a NEGATIVE NaN
    ///     (<c>double.NaN</c>, <c>0xfff8…</c>) already MATCHES NumPy, and only a POSITIVE NaN scale (<c>0x7ff8…</c>)
    ///     diverges — so a pin must fix the exact NaN bits on both sides, never pass <c>double.NaN</c> and compare
    ///     against a positive-NaN reference. The cases here therefore reproduce the corpus arguments exactly rather
    ///     than a hand-simplified call that would silently stop diverging.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsPolynomialNaNTests : TestClass
    {
        /// <summary>
        ///     The three integral cases where the step-table kernels keep the wrong NaN. Each is run through
        ///     <see cref="Check"/>, so a partial fix shows exactly which cases still differ instead of stopping at the
        ///     first failure.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while any case still differs from
        /// NumPy — the expected state until the step tables pin NumPy's per-statement NaN order.</exception>
        [TestMethod, OpenBugs]
        public void PolynomialIntegral_StepTableKeepsWrongNaN()
        {
            // The two integral functions whose recurrences are the step-table kernels under test.
            var G = np.polynomial.laguerre;
            var E = np.polynomial.hermite_e;
            var fails = new List<string>();

            // lagint of a flat float16 series that OVERFLOWS during integration (m=2): the finite coefficients grow
            // past float16's range, so the recurrence forms inf - inf and the NaN's sign is the step order's. NumPy
            // puts a negative NaN then a positive NaN in the first two outputs; NumSharp swaps them.
            Check(fails, "lagint f16 overflow (m=2)",
                () => G.lagint(F16(0xb8d5, 0xc3c7, 0x6650, 0x8b55, 0xe03a), m: 2, scl: -1099511627776L, axis: -1, lbnd: -8.0),
                "float16 (7,) fe00 7e00*2 fc00 fe00 7e00 fc00");

            // hermeint whose integration-constant vector k CONTAINS a NaN (k[1] = 0x7ff8…): the correction
            // tmp[0] += k[i] - hermeval(lbnd, tmp) carries that NaN into the step order. NumPy keeps a negative NaN at
            // output 1 where NumSharp keeps a positive one.
            Check(fails, "hermeint f16 NaN in k (m=3)",
                () => E.hermeint(F16(0x64d5, 0x4700, 0xbc89, 0x7c00, 0x4600)["1:-1"], m: 3, scl: -1L, axis: 0,
                    k: new object[] { FromBits64(0xbf44b60d4beab8ee), FromBits64(0x7ff8000000000000), FromBits64(0xbf844082036edaf5) },
                    lbnd: FromBits64(0xbf32158ccdd0379e)),
                "float16 (6,) fe00*3 bcab 2a0c fc00");

            // lagint whose SCALE is +inf (scl = 0x7ff0…): every finite coefficient becomes ±inf, so the recurrence
            // forms inf - inf throughout. A 3-D transposed int8 series integrated along the last axis (m=2); NumSharp
            // disagrees with NumPy on the sign of eight of the twenty output NaNs.
            Check(fails, "lagint f64 scl=+inf 3-D transposed (m=2)",
                () => G.lagint(np.transpose(np.array(new sbyte[] { -93, -41, -40, -1, -1, -54, 40, 53, -51, -1, -68, 84 }).reshape(3, 2, 2), new[] { 2, 1, 0 }),
                    m: 2, scl: FromBits64(0x7ff0000000000000), axis: -1,
                    k: new object[] { FromBits64(0x3f4869474e06d418), FromBits64(0x7e37e43c8800759c) },
                    lbnd: FromBits64(0xbf96316ec28a8daa)),
                "float64 (2,2,5) 7ff8000000000000 fff8000000000000*2 7ff8000000000000 fff0000000000000 fff8000000000000 " +
                "7ff8000000000000 fff0000000000000 7ff0000000000000 fff0000000000000 7ff8000000000000 fff8000000000000*2 " +
                "7ff8000000000000 fff0000000000000 7ff8000000000000*2 fff8000000000000 7ff8000000000000 7ff0000000000000");

            AssertNoMismatch(fails,
                "the numpy.polynomial step-table integral kernels must keep NumPy 2.4.2's NaN per statement " +
                "(DirectILKernelGenerator.PolyCalculus.cs / ILKernelGenerator.Polynomial.Emitter.cs); " +
                "see plan docs/plans/numpy-polynomial.md section 12");
        }

        /// <summary>
        ///     Builds a fresh float16 array from raw bit patterns, so a NaN or an exact overflow-prone coefficient is
        ///     reproduced bit for bit rather than rounded through a decimal literal.
        /// </summary>
        /// <param name="bits">One IEEE 754 binary16 bit pattern per element, in C order.</param>
        /// <returns>The new 1-D array.</returns>
        private static NDArray F16(params ushort[] bits)
        {
            var h = new Half[bits.Length];
            // UInt16BitsToHalf reinterprets the bits, so a signalling or payload-bearing NaN survives unchanged — a
            // (Half)double cast would quiet or re-sign it.
            for (int i = 0; i < bits.Length; i++)
                h[i] = BitConverter.UInt16BitsToHalf(bits[i]);
            return np.array(h);
        }

        /// <summary>Reinterprets raw bits as a double, preserving a NaN's sign and payload.</summary>
        /// <param name="bits">An IEEE 754 binary64 bit pattern.</param>
        /// <returns>The double with exactly those bits.</returns>
        private static double FromBits64(ulong bits) => BitConverter.Int64BitsToDouble((long)bits);

        /// <summary>
        ///     Runs one case and records a line when its <see cref="Runs"/> rendering differs from NumPy's, catching
        ///     an exception as a mismatch too so one throwing case cannot hide the state of the others.
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
        ///     Fails the test when <see cref="Check"/> recorded any mismatch, listing EVERY recorded line so a partial
        ///     fix shows which cases it repaired.
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
        ///     Renders a result as <c>dtype (shape) tokens…</c>: the NumPy dtype name, the shape in tuple form, then
        ///     one token per element in logical C order — the raw bits in hex (float64 16 digits, float32 8, float16
        ///     4) — with a run of identical tokens written <c>token*count</c>. Raw bits are the point: a value
        ///     comparison treats every NaN as equal and so cannot see these divergences.
        /// </summary>
        /// <param name="r">The result to render: Half, Single or Double.</param>
        /// <returns>The rendering.</returns>
        /// <exception cref="NotSupportedException">When <paramref name="r"/> has another dtype.</exception>
        private static unsafe string Runs(NDArray r)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // A C-order copy walked through its pointer: logical C order for every layout, and no float load or store
            // that could re-sign a NaN. The offset term covers a view that does not start at its buffer's first
            // element; a 0-d result reshapes to one element.
            using var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            byte* p = (byte*)flat.Address + flat.Shape.offset * flat.itemsize;
            var tokens = new string[flat.size];
            for (long i = 0; i < flat.size; i++)
            {
                tokens[i] = flat.typecode switch
                {
                    NPTypeCode.Double => ((ulong*)p)[i].ToString("x16"),
                    NPTypeCode.Single => ((uint*)p)[i].ToString("x8"),
                    NPTypeCode.Half => ((ushort*)p)[i].ToString("x4"),
                    _ => throw new NotSupportedException($"Runs renders float results only, not {flat.typecode}."),
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
