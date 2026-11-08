using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.asanyarray(object[])</c> over boxed C# values of mixed types against NumPy 2.4.2's array coercion of the
    ///     Python list the C# list stands for, each value read through the boundary map (a <see cref="bool"/> is a Python
    ///     bool, every C# integer primitive a Python int, <see cref="float"/> and <see cref="double"/> Python floats,
    ///     <see cref="Complex"/> a Python complex, <see cref="Half"/> a NumPy float16 scalar): the committed corpus
    ///     <c>Creation/corpus/asanyarray_mixed.numpy242.jsonl</c> (2,619 orderings written by
    ///     <c>test/oracle/gen_asanyarray_mixed_oracle.py</c>), plus pins for the defects it replaced.
    /// </summary>
    /// <remarks>
    ///     NumPy promotes the running dtype with each element in order, but under the boundary map no ordering of these
    ///     types changes the result: array coercion discovers a Python int as int64, which widens a float16 to float64 in
    ///     either order, and a bool beside a float16 stays float16 in either order. The element order still decides the
    ///     dtype of a list of NumPy scalars, which C# spells only with <see cref="Half"/> (and NumSharp's own <c>char</c> /
    ///     <c>decimal</c>). The conversion must read every element: the BCL's <see cref="Convert"/> cannot read a
    ///     <see cref="Half"/> (it is not <see cref="IConvertible"/>), which made mixed lists throw InvalidCastException.
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

        /// <summary>
        ///     The boxed C# 1 of the type NumSharp stores as the corpus dtype <paramref name="code"/>; the coercion reads it
        ///     through the boundary map, as the generator's Python value does (a C# sbyte is the Python int 1, not np.int8).
        /// </summary>
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

        /// <summary>
        ///     A float16 among wider types converts (the BCL's Convert threw InvalidCastException: Half is not IConvertible).
        ///     A C# short is a Python int, discovered as int64, so <c>[3, np.float16(2.5)]</c> is float64 in NumPy.
        /// </summary>
        [TestMethod]
        public void Float16AmongWiderTypes_Converts()
        {
            var a = np.asanyarray(new object[] { (short)3, (Half)2.5 });
            a.dtype.Should().Be(np.float64);
            a.GetDouble(0).Should().Be(3.0);
            a.GetDouble(1).Should().Be(2.5);

            var b = np.asanyarray(new object[] { (Half)0.5, 7L });
            b.dtype.Should().Be(np.float64);
            b.GetDouble(0).Should().Be(0.5);
            b.GetDouble(1).Should().Be(7.0);

            var c = np.asanyarray(new object[] { (Half)1.5, new Complex(0, 2) });
            c.dtype.Should().Be(np.complex128);
            c.GetComplex(0).Should().Be(new Complex(1.5, 0));
        }

        /// <summary>
        ///     The C# integers of a list are Python ints in either order: <c>[1, 1, np.float16(1)]</c> and
        ///     <c>[np.float16(1), 1, 1]</c> are both float64 in NumPy (a Python int is discovered as int64), where reading a
        ///     C# sbyte and byte as np.int8 / np.uint8 would give float32 in the first order and float16 in the second. A bool
        ///     beside a float16 stays float16 in either order.
        /// </summary>
        [TestMethod]
        public void Integers_ArePythonInts_InEitherOrder()
        {
            np.asanyarray(new object[] { (sbyte)1, (byte)1, (Half)1 }).dtype.Should().Be(np.float64);
            np.asanyarray(new object[] { (Half)1, (sbyte)1, (byte)1 }).dtype.Should().Be(np.float64);
            np.asanyarray(new object[] { true, (Half)1 }).dtype.Should().Be(np.float16);
            np.asanyarray(new object[] { (Half)1, true }).dtype.Should().Be(np.float16);
        }

        /// <summary>
        ///     Four distinct types are all counted: a fold that promoted the last two and then overwrote that with the
        ///     promotion of the first two ignored the last two. A C# float is a Python float, so the second list is float64 as
        ///     <c>[True, 1, 1, 1.0]</c> is in NumPy.
        /// </summary>
        [TestMethod]
        public void FourDistinctDtypes_AreAllCounted()
        {
            np.asanyarray(new object[] { (sbyte)1, (short)1, 1d, true }).dtype.Should().Be(np.float64);
            np.asanyarray(new object[] { true, (byte)1, (sbyte)1, 1f }).dtype.Should().Be(np.float64);
        }
    }
}
