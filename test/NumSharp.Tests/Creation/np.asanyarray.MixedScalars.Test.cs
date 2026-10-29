using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.asanyarray(object[])</c> over boxed C# values of mixed dtypes against NumPy 2.4.2's array coercion of the
    ///     same scalars (<c>np.array([np.int8(1), np.uint8(1), ...])</c>): the committed corpus
    ///     <c>Creation/corpus/asanyarray_mixed.numpy242.jsonl</c> (2,917 orderings written by
    ///     <c>test/oracle/gen_asanyarray_mixed_oracle.py</c>), plus pins for the defects it replaced.
    /// </summary>
    /// <remarks>
    ///     NumPy promotes the running dtype with each element in order, so the result depends on the order for 18 sets of
    ///     the 13 shared dtypes. The replaced promotion folded from the right and dropped every other pair from four types
    ///     on, and the conversion used the BCL's <see cref="Convert"/>, which cannot read a <see cref="Half"/>: 12,020 of the
    ///     19,032 orderings of 2-4 dtypes gave another dtype or threw InvalidCastException; now none does.
    /// </remarks>
    [TestClass]
    public class np_asanyarray_mixed_scalars_tests
    {
        /// <summary>One corpus case: the element dtypes in order and NumPy's result dtype, in kind + item size spelling.</summary>
        private sealed record Case(string[] T, string R);

        /// <summary>The corpus, read once.</summary>
        private static readonly Lazy<List<Case>> Corpus = new(Load);

        /// <summary>Reads the corpus copied beside the test assembly.</summary>
        /// <returns>The cases, in file order.</returns>
        /// <exception cref="FileNotFoundException">The corpus was not copied to the test output (rebuild the test project).</exception>
        private static List<Case> Load()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Creation", "corpus", "asanyarray_mixed.numpy242.jsonl");
            var cases = new List<Case>();
            foreach (var line in File.ReadLines(path))
            {
                if (line.Length == 0)
                    continue;
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                cases.Add(new Case(root.GetProperty("t").EnumerateArray().Select(e => e.GetString()).ToArray(), root.GetProperty("r").GetString()));
            }
            return cases;
        }

        /// <summary>The boxed C# value of the corpus dtype <paramref name="code"/> (NumSharp maps each C# type onto its own dtype).</summary>
        /// <param name="code">The dtype, kind + item size.</param>
        /// <returns>A boxed 1 of that dtype's C# type.</returns>
        /// <exception cref="ArgumentException">A spelling the corpus does not use.</exception>
        private static object One(string code) => code switch
        {
            "b1" => true, "i1" => (sbyte)1, "u1" => (byte)1, "i2" => (short)1, "u2" => (ushort)1, "i4" => 1, "u4" => 1u,
            "i8" => 1L, "u8" => 1UL, "f2" => (Half)1, "f4" => 1f, "f8" => 1d, "c16" => new Complex(1, 0),
            _ => throw new ArgumentException(code),
        };

        /// <summary>Every corpus ordering gives NumPy 2.4.2's dtype, and every element converts (all are 1, so every result element is 1).</summary>
        [TestMethod]
        public void Corpus_MatchesNumPy242()
        {
            var cases = Corpus.Value;
            cases.Count.Should().BeGreaterThan(2500);
            var problems = new List<string>();
            foreach (var c in cases)
            {
                try
                {
                    var a = np.asanyarray(c.T.Select(One).ToArray());
                    string got = $"{a.dtype.kind}{a.dtype.itemsize}";
                    if (got != c.R)
                        problems.Add($"[{string.Join(",", c.T)}]: NumPy {c.R}, NumSharp {got}");
                    else if (!np.all(a == 1))
                        problems.Add($"[{string.Join(",", c.T)}]: an element did not convert to 1");
                }
                catch (Exception e)
                {
                    problems.Add($"[{string.Join(",", c.T)}]: NumPy {c.R}, NumSharp threw {e.GetType().Name}");
                }
            }

            problems.Should().BeEmpty();
        }

        /// <summary>A float16 among wider types converts (the BCL's Convert threw InvalidCastException: Half is not IConvertible).</summary>
        [TestMethod]
        public void Float16AmongWiderTypes_Converts()
        {
            var a = np.asanyarray(new object[] { (short)3, (Half)2.5 });
            a.dtype.Should().Be(np.float32);
            a.GetSingle(0).Should().Be(3f);
            a.GetSingle(1).Should().Be(2.5f);

            var b = np.asanyarray(new object[] { (Half)0.5, 7L });
            b.dtype.Should().Be(np.float64);
            b.GetDouble(0).Should().Be(0.5);
            b.GetDouble(1).Should().Be(7.0);

            var c = np.asanyarray(new object[] { (Half)1.5, new Complex(0, 2) });
            c.dtype.Should().Be(np.complex128);
            c.GetComplex(0).Should().Be(new Complex(1.5, 0));
        }

        /// <summary>
        ///     The promotion follows element order, as NumPy's does: int8 then uint8 promote to int16 before float16 joins
        ///     (float32), while float16 first absorbs both (float16).
        /// </summary>
        [TestMethod]
        public void Promotion_FollowsElementOrder()
        {
            np.asanyarray(new object[] { (sbyte)1, (byte)1, (Half)1 }).dtype.Should().Be(np.float32);
            np.asanyarray(new object[] { (Half)1, (sbyte)1, (byte)1 }).dtype.Should().Be(np.float16);
        }

        /// <summary>
        ///     Four distinct dtypes are all counted. The old fold promoted the last two, then overwrote that with the promotion
        ///     of the first two, so the result ignored the last two: int16 for the first case, uint8 for the second.
        /// </summary>
        [TestMethod]
        public void FourDistinctDtypes_AreAllCounted()
        {
            np.asanyarray(new object[] { (sbyte)1, (short)1, 1d, true }).dtype.Should().Be(np.float64);
            np.asanyarray(new object[] { true, (byte)1, (sbyte)1, 1f }).dtype.Should().Be(np.float32);
        }
    }
}
