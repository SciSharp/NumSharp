using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Pins of how <c>np.unique</c> / <c>np.unique_values</c> and <c>np.sort</c> treat a float NaN's exact bits
    ///     versus NumPy 2.5.4, former open bugs, now regression guards. Each test asserts NumPy's raw bytes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy 2.5's default float32 / float64 sort (x86-simd-sort <c>fa944ef</c>, vendored since 2.5.2, behind
    ///     <c>kind='quicksort'</c> and <c>'heapsort'</c>) keeps every NaN's bits — negative, signalling or payload-bearing —
    ///     and leaves the NaNs at the end in the order its <c>move_nans_to_end_of_array</c> swap pass puts them. NumPy 2.4's
    ///     rewrote each to the POSITIVE quiet NaN, which NumSharp mirrored until the 2.5.4 pin
    ///     (<c>AxisSort.SimdNaNTail32/64</c> now run that swap pass). The NaN-policy pins below:
    ///     </para>
    ///     <list type="number">
    ///         <item>
    ///         <b><c>np.unique</c> / <c>np.unique_values</c> keep the sorted NaN's bits.</b> NumPy's <c>_unique1d</c>
    ///         sorts in place when neither an index nor an inverse is asked, so a surviving float32 / float64 NaN is the
    ///         first NaN of the sorted tail with its bits (every NaN of the tail, in that order, under
    ///         <c>equal_nan=False</c>); NumSharp's NaN partition is that swap pass, statement for statement. The index and
    ///         inverse routes argsort and gather, which keeps the bits too. (NumSharp wrote NumPy 2.4's canonical NaN on the
    ///         value routes and in the set routines until the 2.5.4 pin.)
    ///         </item>
    ///         <item>
    ///         FIXED (strings phase 2, P2.2): <b><c>np.sort</c> ignored <c>kind</c>.</b> NumPy's <c>kind='stable'</c> and
    ///         <c>'mergesort'</c> do NOT canonicalize — they keep every NaN's exact bits (sign and payload), in input order.
    ///         The kind now reaches the float line sort, which keeps them for the stable kind; the test stays as a
    ///         regression guard (no longer <c>[OpenBugs]</c>).
    ///         </item>
    ///     </list>
    ///     <para>
    ///     The value is a NaN either way, so every ordinary oracle tier (and <c>np.allclose</c>) treats these as
    ///     equal; the divergence is visible only at the raw-bit level, which is why it is pinned here. The swap pass's order
    ///     is x86-simd-sort's (NumPy's sort on an AVX2 / AVX-512 host; another host's NumPy sorts NaNs its own way). The
    ///     float16 and complex128 dtypes are deliberately absent: NumPy's own unique and sort keep those NaNs' bits by a
    ///     comparison sort, and so does NumSharp.
    ///     </para>
    ///     <para>
    ///     Every expected string is NumPy 2.5.4's output for the same call (same operand bits), probed on 2026-11-09,
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
        ///     <c>np.unique</c>'s values-only and counts-only routes, <c>np.unique_values</c> and <c>np.unique_counts</c>
        ///     return the first NaN of NumPy's sorted tail WITH its bits (NumPy 2.5's in-place sort keeps them; NumPy 2.4's
        ///     wrote the positive quiet NaN); under <c>equal_nan=False</c> every NaN of the tail, in the swap pass's order. A
        ///     one-element array is not sorted, and the index route argsorts and gathers: their NaNs keep their bits too.
        ///     Each case is NumPy 2.5.4's output.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, when a route rewrites a NaN NumPy keeps,
        ///     or keeps the wrong one (a regression).</exception>
        [TestMethod]
        public void Unique_ValueRoutes_KeepTheSortedNaNsBits()
        {
            var fails = new List<string>();
            // The counts-only route and np.unique_counts sort in place too (neither an index nor an inverse is asked).
            Check(fails, "unique return_counts f64 negative NaN",
                () => np.unique(F64(NegNaN64, One64, Two64), return_counts: true).values,
                "float64 (3,) 3ff0000000000000 4000000000000000 fff8000000000000");
            Check(fails, "unique_counts f64 payload NaN",
                () => np.unique_counts(F64(PayloadNaN64, One64, Two64)).values,
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000abc");
            // [payload, -nan, 1.0]: the swap pass moves the payload NaN to the end and leaves -nan where the scans meet.
            Check(fails, "unique equal_nan=False f64 two NaNs",
                () => (NDArray)np.unique(F64(PayloadNaN64, NegNaN64, One64), equal_nan: false),
                "float64 (3,) 3ff0000000000000 fff8000000000000 7ff8000000000abc");
            // One element: NumPy does not sort a line of one element, so the NaN keeps its bits.
            Check(fails, "unique f64 one negative NaN",
                () => (NDArray)np.unique(F64(NegNaN64)),
                "float64 (1,) fff8000000000000");
            Check(fails, "unique f32 one negative NaN",
                () => (NDArray)np.unique(F32(NegNaN32)),
                "float32 (1,) ffc00000");
            // The index route argsorts and gathers: the NaN keeps its bits.
            Check(fails, "unique return_index f64 negative NaN",
                () => np.unique(F64(NegNaN64, One64, Two64), return_index: true).values,
                "float64 (3,) 3ff0000000000000 4000000000000000 fff8000000000000");

            // A NEGATIVE float64 NaN keeps its sign (NumPy 2.4 returned +0x7ff8…).
            Check(fails, "unique f64 negative NaN",
                () => (NDArray)np.unique(F64(NegNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 fff8000000000000");
            Check(fails, "unique_values f64 negative NaN",
                () => np.unique_values(F64(NegNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 fff8000000000000");
            // A float64 NaN carrying a PAYLOAD keeps it (NumPy 2.4 stripped it to 0x7ff8000000000000).
            Check(fails, "unique f64 payload NaN",
                () => (NDArray)np.unique(F64(PayloadNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000abc");
            Check(fails, "unique_values f64 payload NaN",
                () => np.unique_values(F64(PayloadNaN64, One64, Two64)),
                "float64 (3,) 3ff0000000000000 4000000000000000 7ff8000000000abc");
            // float32 behaves the same: a negative NaN stays 0xffc00000.
            Check(fails, "unique f32 negative NaN",
                () => (NDArray)np.unique(F32(NegNaN32, One32, Two32)),
                "float32 (3,) 3f800000 40000000 ffc00000");
            Check(fails, "unique_values f32 negative NaN",
                () => np.unique_values(F32(NegNaN32, One32, Two32)),
                "float32 (3,) 3f800000 40000000 ffc00000");
            AssertNoMismatch(fails,
                "np.unique's values-only and counts-only routes, np.unique_values and np.unique_counts keep the surviving " +
                "NaN's bits in the order NumPy 2.5's in-place sort leaves the NaN tail (x86-simd-sort's swap pass); a " +
                "one-element array and the index route keep the input bits too — see " +
                "Backends/Default/Manipulation/Default.Unique.Kwargs.cs PartitionNaN_DoubleKeysOnly");
        }

        /// <summary>
        ///     <c>np.sort</c> with <c>kind='stable'</c> / <c>'mergesort'</c> / <c>stable=True</c>, and <c>ndarray.sort</c> with
        ///     the stable kind, PRESERVE every NaN's exact bits in input order; the default kind preserves them too since NumPy
        ///     2.5, in the order x86-simd-sort's swap pass leaves them (<c>AxisSort.SimdNaNTail32/64</c>; NumPy 2.4 wrote the
        ///     positive quiet NaN). Guards both kinds, float64 and float32.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, when a sort rewrites a NaN or leaves the
        ///     NaN tail in another order (a regression).</exception>
        [TestMethod]
        public void Sort_StableKinds_KeepEveryNaNsBits()
        {
            var fails = new List<string>();

            // NumPy's stable / mergesort leave the negative NaN (0xfff8…) and the payload NaN (0x7ff8…abc) untouched,
            // in input order, after the finite values; its default kind keeps both, in the swap pass's order.
            foreach (var kind in new[] { "stable", "mergesort" })
                Check(fails, $"sort f64 kind={kind}",
                    () => np.sort(F64(One64, NegNaN64, Two64, PayloadNaN64), kind: kind),
                    "float64 (4,) 3ff0000000000000 4000000000000000 fff8000000000000 7ff8000000000abc");
            Check(fails, "sort f64 stable=True",
                () => np.sort(F64(One64, NegNaN64, Two64, PayloadNaN64), stable: true),
                "float64 (4,) 3ff0000000000000 4000000000000000 fff8000000000000 7ff8000000000abc");
            Check(fails, "sort f64 default",
                () => np.sort(F64(One64, NegNaN64, Two64, PayloadNaN64)),
                "float64 (4,) 3ff0000000000000 4000000000000000 7ff8000000000abc fff8000000000000");
            // float32: np.sort([1, -nan, 2, nan(0x123)], kind='stable') keeps 0xffc00000 then 0x7fc00123; ndarray.sort too.
            Check(fails, "sort f32 kind=stable",
                () => np.sort(F32(One32, NegNaN32, Two32, 0x7fc00123), kind: "stable"),
                "float32 (4,) 3f800000 40000000 ffc00000 7fc00123");
            Check(fails, "ndarray.sort f32 kind=stable",
                () =>
                {
                    var a = F32(One32, NegNaN32, Two32, 0x7fc00123);
                    a.sort(kind: "stable");
                    return a;
                },
                "float32 (4,) 3f800000 40000000 ffc00000 7fc00123");
            Check(fails, "sort f32 default",
                () => np.sort(F32(One32, NegNaN32, Two32, 0x7fc00123)),
                "float32 (4,) 3f800000 40000000 7fc00123 ffc00000");

            AssertNoMismatch(fails,
                "np.sort(kind='stable'|'mergesort') must keep every NaN's exact bits in input order, and the default kind " +
                "in x86-simd-sort's swap-pass order (NumPy 2.5) — see Backends/Default/Sorting/AxisSort.cs SimdNaNTail32 / " +
                "SimdNaNTail64");
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
        /// <param name="expected">NumPy 2.5.4's result in <see cref="Runs"/>'s format.</param>
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
