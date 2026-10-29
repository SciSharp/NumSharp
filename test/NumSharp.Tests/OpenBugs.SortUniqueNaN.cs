using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in how <c>np.unique</c> / <c>np.unique_values</c> and <c>np.sort</c> treat a float NaN's exact
    ///     bits versus NumPy 2.4.2. Each test asserts NumPy's raw bytes, so it FAILS while NumSharp's NaN policy
    ///     differs and passes once it matches. Remove <c>[OpenBugs]</c> from a test when its path is fixed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy's default float32 / float64 sort (x86-simd-sort, behind <c>kind='quicksort'</c> and
    ///     <c>'heapsort'</c>) rewrites every NaN of a line longer than one element — negative, signalling or
    ///     payload-bearing — to <c>std::numeric_limits&lt;T&gt;::quiet_NaN()</c>, the POSITIVE <c>0x7ff8000000000000</c>
    ///     / <c>0x7fc00000</c>. NumSharp's sort does the same since commit 7026b3e5 (<c>AxisSort.SortLineF32/F64</c>,
    ///     the F5 fix). Two NaN-policy gaps remain, both library-wide:
    ///     </para>
    ///     <list type="number">
    ///         <item>
    ///         <b><c>np.unique</c> / <c>np.unique_values</c> do NOT canonicalize.</b> NumPy's unique SORTS first, so
    ///         a surviving float32 / float64 NaN comes back as the canonical positive quiet NaN with no payload.
    ///         NumSharp's unique path preserves the input NaN's sign and payload (its sort-then-mask core keeps the
    ///         original bits, and only the set routines — <c>union1d</c> / <c>setxor1d</c> / <c>setdiff1d</c> — run
    ///         <c>CanonicalizeSetOpNaN</c> afterwards; plain <c>unique</c> does not). So a negative or payload NaN fed
    ///         to <c>np.unique</c> comes back unchanged here where NumPy returns <c>0x7ff8…</c>.
    ///         </item>
    ///         <item>
    ///         <b><c>np.sort</c> ignores <c>kind</c>.</b> NumPy's <c>kind='stable'</c> and <c>'mergesort'</c> do NOT
    ///         canonicalize — they keep every NaN's exact bits (sign and payload). NumSharp ignores the <c>kind</c>
    ///         argument entirely and always runs the one sort core, which canonicalizes, so a stable / mergesort call
    ///         loses the NaN bits NumPy preserves.
    ///         </item>
    ///     </list>
    ///     <para>
    ///     The value is a NaN either way, so every ordinary oracle tier (and <c>np.allclose</c>) treats these as
    ///     equal; the divergence is visible only at the raw-bit level, which is why it is pinned here. NumPy's
    ///     canonical quiet NaN is a portable constant (not a host-libm artifact), so unlike the win-amd64 NaN-operand
    ///     pins these are platform-independent. The float16 and complex128 dtypes are deliberately absent: NumPy's own
    ///     unique and sort keep those NaNs' bits, and so does NumSharp, so there is nothing to pin.
    ///     </para>
    ///     <para>
    ///     Every expected string is NumPy 2.4.2's output for the same call (same operand bits), probed on 2026-10-05,
    ///     in <see cref="Runs"/>'s format: <c>dtype (shape) tokens…</c>, one hex token per element in logical C order,
    ///     a run of identical tokens written <c>token*count</c>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsSortUniqueNaNTests : TestClass
    {
        /// <summary>float64 negative quiet NaN (no payload).</summary>
        private const ulong NegNaN64 = 0xfff8000000000000;

        /// <summary>float64 positive quiet NaN carrying payload 0xabc, to show the payload is stripped.</summary>
        private const ulong PayloadNaN64 = 0x7ff8000000000abc;

        /// <summary>float64 bits of 1.0 and 2.0 — finite neighbours that sort below the NaN.</summary>
        private const ulong One64 = 0x3ff0000000000000, Two64 = 0x4000000000000000;

        /// <summary>float32 negative quiet NaN, and the bits of 1.0f / 2.0f.</summary>
        private const uint NegNaN32 = 0xffc00000, One32 = 0x3f800000, Two32 = 0x40000000;

        /// <summary>
        ///     <c>np.unique</c> and <c>np.unique_values</c> must return NumPy's canonical positive quiet NaN
        ///     (<c>0x7ff8000000000000</c> / <c>0x7fc00000</c>) for a surviving float NaN, whatever the input NaN's sign
        ///     or payload, because NumPy's unique sorts first. NumSharp keeps the input bits, so each case below fails.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while NumSharp's unique preserves
        /// the input NaN bits — the expected state until its unique path canonicalizes float NaN like NumPy's.</exception>
        [TestMethod, OpenBugs]
        public void Unique_KeepsInputNaNBits_InsteadOfCanonicalQuietNaN()
        {
            var fails = new List<string>();

            // A NEGATIVE float64 NaN: NumPy's sorted unique returns +0x7ff8…; NumSharp keeps 0xfff8….
            Check(fails, "unique f64 negative NaN",
                () => (NDArray)np.unique(F64(NegNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000000");
            Check(fails, "unique_values f64 negative NaN",
                () => np.unique_values(F64(NegNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000000");

            // A float64 NaN carrying a PAYLOAD: NumPy strips it to the canonical 0x7ff8000000000000; NumSharp keeps it.
            Check(fails, "unique f64 payload NaN",
                () => (NDArray)np.unique(F64(PayloadNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000000");
            Check(fails, "unique_values f64 payload NaN",
                () => np.unique_values(F64(PayloadNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000000");

            // float32 behaves the same: a negative NaN becomes the canonical +0x7fc00000.
            Check(fails, "unique f32 negative NaN",
                () => (NDArray)np.unique(F32(NegNaN32, One32, Two32)),
                "float32 (3,) 3f800000 40000000 7fc00000");
            Check(fails, "unique_values f32 negative NaN",
                () => np.unique_values(F32(NegNaN32, One32, Two32)),
                "float32 (3,) 3f800000 40000000 7fc00000");

            AssertNoMismatch(fails,
                "np.unique / np.unique_values must canonicalize a surviving float NaN to NumPy's positive quiet NaN " +
                "(NumPy's unique sorts first); NumSharp keeps the input NaN bits — see Manipulation/np.setops.cs " +
                "CanonicalizeSetOpNaN and plan docs/plans/numpy-polynomial.md section 12");
        }

        /// <summary>
        ///     <c>np.sort</c> with <c>kind='stable'</c> or <c>'mergesort'</c> must PRESERVE every NaN's exact bits
        ///     (NumPy's stable kinds do not canonicalize), where NumSharp ignores <c>kind</c> and always canonicalizes.
        ///     The same array sorted with the default (quicksort) kind already matches NumPy on both sides and so is
        ///     not pinned here.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while NumSharp ignores
        /// <c>kind</c> — the expected state until a stable / mergesort sort keeps NaN bits like NumPy's.</exception>
        [TestMethod, OpenBugs]
        public void Sort_IgnoresKind_CanonicalizesWhereStableKindsPreserveNaNBits()
        {
            var fails = new List<string>();

            // NumPy's stable / mergesort leave the negative NaN (0xfff8…) and the payload NaN (0x7ff8…abc) untouched
            // after the finite values; NumSharp rewrites both to the canonical 0x7ff8000000000000.
            foreach (var kind in new[] { "stable", "mergesort" })
                Check(fails, $"sort f64 kind={kind}",
                    () => np.sort(F64(One64, NegNaN64, Two64, PayloadNaN64), kind: kind),
                    "float64 (4,) 3ff0000000000000 4000000000000000 fff8000000000000 7ff8000000000abc");

            AssertNoMismatch(fails,
                "np.sort(kind='stable'|'mergesort') must keep every NaN's exact bits (NumPy's stable kinds do not " +
                "canonicalize); NumSharp ignores kind and canonicalizes — see Sorting_Searching_Counting/np.sort.cs " +
                "and plan docs/plans/numpy-polynomial.md section 12");
        }

        /// <summary>Builds a fresh float64 array from raw bit patterns, preserving each NaN's sign and payload.</summary>
        /// <param name="bits">One IEEE 754 binary64 bit pattern per element, in C order.</param>
        /// <returns>The new 1-D array.</returns>
        private static NDArray F64(params ulong[] bits)
        {
            var d = new double[bits.Length];
            // Int64BitsToDouble reinterprets the bits, so a payload-bearing or negative NaN survives — a literal would
            // not express one and a cast could quiet it.
            for (int i = 0; i < bits.Length; i++)
                d[i] = BitConverter.Int64BitsToDouble((long)bits[i]);
            return np.array(d);
        }

        /// <summary>Builds a fresh float32 array from raw bit patterns, preserving each NaN's sign and payload.</summary>
        /// <param name="bits">One IEEE 754 binary32 bit pattern per element, in C order.</param>
        /// <returns>The new 1-D array.</returns>
        private static NDArray F32(params uint[] bits)
        {
            var f = new float[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                f[i] = BitConverter.Int32BitsToSingle((int)bits[i]);
            return np.array(f);
        }

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
        ///     one token per element in logical C order — the raw bits in hex (float64 16 digits, float32 8) — with a
        ///     run of identical tokens written <c>token*count</c>. Raw bits are the point: a value comparison treats
        ///     every NaN as equal and so cannot see these divergences.
        /// </summary>
        /// <param name="r">The result to render: Single or Double.</param>
        /// <returns>The rendering.</returns>
        /// <exception cref="NotSupportedException">When <paramref name="r"/> has another dtype.</exception>
        private static unsafe string Runs(NDArray r)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // A C-order copy walked through its pointer: logical C order for every layout, and no float load or store
            // that could re-sign a NaN. The offset term covers a view that does not start at its buffer's first element.
            using var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            byte* p = (byte*)flat.Address + flat.Shape.offset * flat.itemsize;
            var tokens = new string[flat.size];
            for (long i = 0; i < flat.size; i++)
            {
                tokens[i] = flat.typecode switch
                {
                    NPTypeCode.Double => ((ulong*)p)[i].ToString("x16"),
                    NPTypeCode.Single => ((uint*)p)[i].ToString("x8"),
                    _ => throw new NotSupportedException($"Runs renders float32 / float64 results only, not {flat.typecode}."),
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
