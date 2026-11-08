using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    ///     <c>np.find_common_type</c> against NumPy 1.26.4, the last NumPy that has it (deprecated in 1.25, removed in 2.0):
    ///     the committed corpus <c>Logic/corpus/find_common_type.numpy1264.jsonl</c> (11,291 cases written by
    ///     <c>test/oracle/gen_find_common_type_oracle.py</c> from NumPy 1.26.4), plus pins for the defects it replaced.
    /// </summary>
    /// <remarks>
    ///     The replaced engine compared NPTypeCode ENUM values where NumPy compares safe casts, read NumPy 2's NEP 50
    ///     array-scalar table for a higher-kind scalar, dropped every other pair from four dtypes on, sliced the dtype list at
    ///     a candidate index (a negative length: OverflowException), and read decimal's character as 'q' (int64). Over the
    ///     full matrix of the 13 shared dtypes (66,443 cases) it differed from NumPy 1.26.4 on 15,696; the port differs on 0.
    /// </remarks>
    [TestClass]
    public class np_find_common_type_numpy1264_tests
    {
        /// <summary>One corpus case: array dtypes, scalar dtypes, NumPy's result (null for None), in kind + item size spelling.</summary>
        private sealed record Case(string[] A, string[] S, string R);

        /// <summary>The corpus, read once.</summary>
        private static readonly Lazy<List<Case>> Corpus = new(Load);

        /// <summary>Reads the corpus copied beside the test assembly.</summary>
        /// <returns>The cases, in file order.</returns>
        /// <exception cref="FileNotFoundException">The corpus was not copied to the test output (rebuild the test project).</exception>
        private static List<Case> Load()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Logic", "corpus", "find_common_type.numpy1264.jsonl");
            var cases = new List<Case>();
            foreach (var line in File.ReadLines(path))
            {
                if (line.Length == 0)
                    continue;
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                string[] Codes(string p) => root.GetProperty(p).EnumerateArray().Select(e => e.GetString()).ToArray();
                var r = root.GetProperty("r");
                cases.Add(new Case(Codes("a"), Codes("s"), r.ValueKind == JsonValueKind.Null ? null : r.GetString()));
            }
            return cases;
        }

        /// <summary>A result in the corpus's kind + item size spelling, or null for none.</summary>
        /// <param name="d">The dtype, or null.</param>
        /// <returns>The spelling.</returns>
        private static string Code(DType d) => d is null ? null : $"{d.kind}{d.itemsize}";

        /// <summary>Every corpus case gives NumPy 1.26.4's answer (the corpus is not empty and covers the sections the generator writes).</summary>
        [TestMethod]
        public void Corpus_MatchesNumPy1264()
        {
            var cases = Corpus.Value;
            cases.Count.Should().BeGreaterThan(11000);
            cases.Count(c => c.S.Length == 0).Should().BeGreaterThan(2000, "array types alone");
            cases.Count(c => c.A.Length == 0).Should().BeGreaterThan(2000, "scalar types alone");
            cases.Count(c => c.A.Length > 0 && c.S.Length > 0).Should().BeGreaterThan(4000, "array x scalar types");

            var problems = new List<string>();
            foreach (var c in cases)
            {
                string got;
                try
                {
                    got = Code(np.find_common_type(c.A.Select(x => np.dtype(x)).ToArray(), c.S.Select(x => np.dtype(x)).ToArray()));
                }
                catch (Exception e)
                {
                    got = e.GetType().Name;
                }
                if (got != c.R)
                    problems.Add($"a=[{string.Join(",", c.A)}] s=[{string.Join(",", c.S)}]: NumPy {c.R ?? "None"}, NumSharp {got ?? "None"}");
            }

            problems.Should().BeEmpty();
        }

        /// <summary>The fold this replaced dropped every other pair from four dtypes on: (int8, int8, float64, int8) came back int8.</summary>
        [TestMethod]
        public void FourArrayTypes_AreAllCounted()
        {
            np.find_common_type(new[] { np.int8, np.int8, np.float64, np.int8 }).Should().Be(np.float64);
            np.find_common_type(new[] { np.@bool, np.@bool, np.@bool, np.int8 }).Should().Be(np.int8);
        }

        /// <summary>
        ///     A maximum is the FIRST candidate every input casts to safely, not a pairwise promotion: int8 and uint8 both cast
        ///     to float16, so (float16, int8, uint8) is float16 where promote_types(promote_types(int8, uint8), float16) is
        ///     float32.
        /// </summary>
        [TestMethod]
        public void Maximum_IsTheCandidateSearch()
        {
            np.find_common_type(new[] { np.float16, np.int8, np.uint8 }).Should().Be(np.float16);
            np.find_common_type(new[] { np.float32, np.int8, np.uint16 }).Should().Be(np.float32);
        }

        /// <summary>
        ///     A scalar of a higher kind is NumPy's _find_common_coerce, not NEP 50's weak-scalar rule: uint8 arrays with an
        ///     int16 scalar give int16 (the NEP 50 table kept uint8), and int64 arrays with a float32 scalar give float64.
        /// </summary>
        [TestMethod]
        public void HigherKindScalar_IsFindCommonCoerce()
        {
            np.find_common_type(new[] { np.uint8 }, new[] { np.int16 }).Should().Be(np.int16);
            np.find_common_type(new[] { np.uint16 }, new[] { np.int16 }).Should().Be(np.int32);
            np.find_common_type(new[] { np.int64 }, new[] { np.float32 }).Should().Be(np.float64);
            np.find_common_type(new[] { np.float32 }, new[] { np.complex128 }).Should().Be(np.complex128);
            // A scalar of the same or a lower kind never moves the array maximum (np.result_type(int8, int64) is int64).
            np.find_common_type(new[] { np.int8 }, new[] { np.int64 }).Should().Be(np.int8);
            np.find_common_type(new[] { np.int8 }, new[] { np.uint64 }).Should().Be(np.int8);
            np.find_common_type(new[] { np.float32 }, new[] { np.int64, np.float64 }).Should().Be(np.float32);
        }

        /// <summary>
        ///     The NDArray form takes 0-d arrays as the scalar types and reaches _find_common_coerce; it used to crash with
        ///     OverflowException for (uint16 array, int16 0-d) because the search sliced the dtype list at the candidate index.
        /// </summary>
        [TestMethod]
        public void NDArrayForm_SplitsZeroDimArraysAsScalars()
        {
            var arrays = new[] { np.zeros(new Shape(3), np.uint16), NDArray.Scalar((short)1) };
            np._FindCommonType(arrays).Should().Be(NPTypeCode.Int32);
            var bytes = new[] { np.zeros(new Shape(3), np.uint8), NDArray.Scalar((sbyte)1) };
            np._FindCommonType(bytes).Should().Be(NPTypeCode.Int16);
        }

        /// <summary>
        ///     NumSharp's own dtypes have no place in NumPy's candidate list and are promoted with NumSharp's table: decimal
        ///     wins over every real type, so it is never ranked as int64 (the old character 'q') and an integer array with a
        ///     decimal scalar gives decimal; char promotes as uint16.
        /// </summary>
        [TestMethod]
        public void NumSharpOnlyDtypes_UseThePromotionTable()
        {
            np.find_common_type(new[] { np.@decimal, np.float64 }).Should().Be(np.@decimal);
            np.find_common_type(new[] { np.int32 }, new[] { np.@decimal }).Should().Be(np.@decimal);
            np.find_common_type(new[] { np.uint64 }, new[] { np.@decimal }).Should().Be(np.@decimal);
            np.find_common_type(new[] { np.@char, np.int16 }).Should().Be(np.int32);
            np.find_common_type(new[] { np.@char, np.@char, np.@char, np.float64 }).Should().Be(np.float64);
        }

        /// <summary>No input is NumPy's None (null), and a dtype string NumSharp cannot store is refused by the parser.</summary>
        [TestMethod]
        public void NoInput_IsNone()
        {
            np.find_common_type(Array.Empty<DType>(), Array.Empty<DType>()).Should().BeNull();
            np.find_common_type((DType[])null, null).Should().BeNull();
            Action complex64 = () => np.find_common_type(new[] { "f4" }, new[] { "c8" });
            complex64.Should().Throw<NotSupportedException>();
        }
    }
}
