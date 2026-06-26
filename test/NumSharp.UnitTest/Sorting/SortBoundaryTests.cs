using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Sorting
{
    /// <summary>
    /// Stresses the radix line kernel across the insertion→radix crossover (4-byte keys switch at
    /// n=80, 8-byte at n=120) and at sizes well into the radix regime, with heavy ties so the
    /// stability contract (equal keys keep ascending index order — NumPy 'stable') is actually
    /// exercised. Each case is checked against a deterministic C# reference (stable sort / stable
    /// argsort) rather than hand-pinned values, so it covers line lengths the curated NumPy-value
    /// tests don't reach.
    /// </summary>
    [TestClass]
    public class SortBoundaryTests
    {
        // line lengths bracketing both thresholds (79/80/81 and 119/120/121) plus radix-regime sizes
        private static readonly int[] Lengths = { 1, 2, 16, 79, 80, 81, 119, 120, 121, 257, 1000, 5000 };

        private static int[] RandInts(int n, int seed, int lo, int hi)
        {
            var r = new Random(seed);
            var a = new int[n];
            for (int i = 0; i < n; i++) a[i] = r.Next(lo, hi);
            return a;
        }

        [TestMethod]
        public void Sort_Int32_MatchesStableReference_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                // narrow value range => many ties => stresses stability + the histogram trivial-pass skip
                var data = RandInts(n, 1234 + n, 0, Math.Max(2, n / 4));
                var expected = (int[])data.Clone();
                Array.Sort(expected);

                var got = np.sort(np.array(data)).ToArray<int>();
                got.Should().BeEquivalentTo(expected, o => o.WithStrictOrdering(),
                    $"np.sort(int32) of a length-{n} line must equal the reference ascending sort");
            }
        }

        [TestMethod]
        public void Sort_Int64_MatchesStableReference_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                var r = new Random(77 + n);
                var data = new long[n];
                for (int i = 0; i < n; i++) data[i] = ((long)r.Next(-5, 5) << 40) | (uint)r.Next();
                var expected = (long[])data.Clone();
                Array.Sort(expected);

                var got = np.sort(np.array(data)).ToArray<long>();
                got.Should().BeEquivalentTo(expected, o => o.WithStrictOrdering(),
                    $"np.sort(int64) of a length-{n} line must equal the reference ascending sort");
            }
        }

        [TestMethod]
        public void Sort_UInt8_And_Int16_MatchReference_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                var r = new Random(9 + n);
                var b = new byte[n]; for (int i = 0; i < n; i++) b[i] = (byte)r.Next(0, 256);
                var eb = (byte[])b.Clone(); Array.Sort(eb);
                np.sort(np.array(b)).ToArray<byte>().Should().BeEquivalentTo(eb, o => o.WithStrictOrdering(),
                    $"np.sort(uint8) length-{n}");

                var s = new short[n]; for (int i = 0; i < n; i++) s[i] = (short)r.Next(-400, 400);
                var es = (short[])s.Clone(); Array.Sort(es);
                np.sort(np.array(s)).ToArray<short>().Should().BeEquivalentTo(es, o => o.WithStrictOrdering(),
                    $"np.sort(int16) length-{n}");
            }
        }

        [TestMethod]
        public void Sort_Float64_MatchesReference_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                var r = new Random(303 + n);
                var data = new double[n];
                for (int i = 0; i < n; i++) data[i] = Math.Round((r.NextDouble() * 20 - 10), 2); // ties via rounding
                var expected = (double[])data.Clone();
                Array.Sort(expected);

                var got = np.sort(np.array(data)).ToArray<double>();
                got.Should().BeEquivalentTo(expected, o => o.WithStrictOrdering(),
                    $"np.sort(float64) of a length-{n} line must equal the reference ascending sort");
            }
        }

        [TestMethod]
        public void Argsort_Int32_IsStable_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                var data = RandInts(n, 555 + n, 0, Math.Max(2, n / 8)); // heavy ties
                // reference STABLE argsort: order by (value, original index)
                var refIdx = Enumerable.Range(0, n)
                    .OrderBy(i => data[i]).ThenBy(i => i)
                    .Select(i => (long)i).ToArray();

                var got = np.argsort(np.array(data)).ToArray<long>();
                got.Should().BeEquivalentTo(refIdx, o => o.WithStrictOrdering(),
                    $"np.argsort(int32) of a length-{n} line must be the STABLE permutation (ties in ascending index order)");

                // and it must actually index the data in sorted order
                var taken = got.Select(g => data[g]).ToArray();
                var sortedVals = (int[])data.Clone(); Array.Sort(sortedVals);
                taken.Should().BeEquivalentTo(sortedVals, o => o.WithStrictOrdering());
            }
        }

        [TestMethod]
        public void Argsort_Int64_IsStable_AcrossThreshold()
        {
            foreach (int n in Lengths)
            {
                var r = new Random(8080 + n);
                var data = new long[n];
                for (int i = 0; i < n; i++) data[i] = r.Next(0, Math.Max(2, n / 6)); // heavy ties, 8-byte path
                var refIdx = Enumerable.Range(0, n)
                    .OrderBy(i => data[i]).ThenBy(i => i)
                    .Select(i => (long)i).ToArray();

                var got = np.argsort(np.array(data)).ToArray<long>();
                got.Should().BeEquivalentTo(refIdx, o => o.WithStrictOrdering(),
                    $"np.argsort(int64) of a length-{n} line must be the STABLE permutation");
            }
        }

        [TestMethod]
        public void Sort_2D_PerLine_AcrossThreshold()
        {
            // (rows, L) with L straddling the 4-byte threshold: every row sorted independently along axis=-1
            foreach (int L in new[] { 80, 81, 121, 300 })
            {
                int rows = 37;
                var r = new Random(4242 + L);
                var data = new int[rows * L];
                for (int i = 0; i < data.Length; i++) data[i] = r.Next(-1000, 1000);
                var a = np.array(data).reshape(rows, L);
                var s = np.sort(a, -1);
                for (int row = 0; row < rows; row++)
                {
                    var expected = new int[L];
                    Array.Copy(data, row * L, expected, 0, L);
                    Array.Sort(expected);
                    var got = new int[L];
                    for (int c = 0; c < L; c++) got[c] = s.GetInt32(row, c);
                    got.Should().BeEquivalentTo(expected, o => o.WithStrictOrdering(),
                        $"row {row} of a ({rows},{L}) sort along axis=-1");
                }
            }
        }
    }
}
