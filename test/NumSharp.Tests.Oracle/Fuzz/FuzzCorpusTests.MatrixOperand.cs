using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The matrix-operand tiers (<c>test/oracle/gen_matrix_operand_oracle.py</c>): existing cases of the corpus
    ///     re-emitted with one two-dimensional operand typed as a <c>numpy.matrix</c>, and NumPy 2.4.2's answer
    ///     recomputed. They hold every public <c>np.*</c> function to the CLASS and the SHAPE NumPy returns for a
    ///     matrix operand: a matrix from the ufuncs and from the functions that call the operand's own method
    ///     (<c>np.sum(m, 0)</c> is a 1 x N matrix), a plain array from the functions that strip the class
    ///     (<c>np.ravel</c>, <c>np.where</c>), an error from the ones that cannot keep the result in two axes
    ///     (<c>np.stack</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The operand reaches the function through an <see cref="NDArray"/> reference (the registry's operands are
    ///     typed <see cref="NDArray"/>), so these tiers gate the run-time rule of the public functions: the
    ///     <see cref="NDMatrix"/> members, chosen at compile time, are the subject of <see cref="Matrix"/>.
    ///     </para>
    ///     <para>
    ///     The variants keep their source's op key, params and valueclass, so a VALUE divergence the source tier
    ///     documents stays documented here. The class is never excused (the comparator has no registry call for
    ///     it) and neither is the shape (<see cref="MisalignedRegistry"/> returns null for a shape divergence of a
    ///     case whose layout starts with <c>matrix@</c>). Known divergences are carved by the generator into
    ///     <c>gaps/matrix_operand*_gaps.jsonl</c> under named gap ids and replayed by the <c>*_KnownGaps</c> tests.
    ///     </para>
    ///     <para>
    ///     <c>matrix_operand_host.jsonl</c> holds the variants of the host-libm tiers and replays through
    ///     <see cref="RunHostLibmCorpus"/> like them.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>The portable matrix-operand variants: NumPy's values, class and shape, or its error.</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MatrixOperand() => RunCorpus("matrix_operand.jsonl");

        /// <summary>The matrix-operand variants of the host-libm tiers (hard-gated on win-amd64, inconclusive
        ///     elsewhere).</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MatrixOperandHost() => RunHostLibmCorpus("matrix_operand_host.jsonl");

        /// <summary>The carved gaps of <see cref="MatrixOperand"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void MatrixOperand_KnownGaps() => RunGapCorpus("matrix_operand_gaps.jsonl");

        /// <summary>The carved gaps of <see cref="MatrixOperandHost"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void MatrixOperandHost_KnownGaps() => RunGapCorpus("matrix_operand_host_gaps.jsonl");

        /// <summary>
        ///     The boundary's safety contract, measured over the whole corpus: no method of the library itself
        ///     reaches the matrix rule of a public function. Every case of every ordinary tier that has a
        ///     two-dimensional operand is replayed with those operands typed as matrices while
        ///     <see cref="NDMatrixBoundary.AuditInternalCallers"/> is on, and the set of library callers the
        ///     audit records must stay empty.
        /// </summary>
        /// <exception cref="AssertFailedException">A library method called a public function that has a matrix
        ///     rule, with a matrix operand, outside an open boundary region.</exception>
        /// <remarks>
        ///     <para>
        ///     Why it matters: a public function with a matrix rule returns NumPy's matrix shape (a 1 x N row
        ///     where the plain function returns N elements) to whoever calls it at depth zero. The USER is meant
        ///     to; library code is written for the plain result. The three ways library code gets there are an
        ///     <see cref="NDArray"/> member or operator applied to a matrix object (it must keep the plain
        ///     meaning), an engine method that calls back up into <c>np.*</c>, and a public function WITHOUT a
        ///     rule that calls one WITH a rule on its own operand. The fix for a finding is a region
        ///     (<c>NDMatrixBoundary.PlainFor</c> / <c>Inside</c>) or a rule for the calling function.
        ///     </para>
        ///     <para>
        ///     The outcomes of the replays are not compared (most of these calls have no NumPy answer recorded
        ///     for a matrix operand); only the call paths are observed. The cases of the matrix tier itself are
        ///     included: their operands are matrices already, and they drive every <see cref="NDMatrix"/> member.
        ///     </para>
        /// </remarks>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MatrixOperand_LibraryNeverReachesAMatrixRule()
        {
            var ignoredFailures = new System.Collections.Generic.List<string>();
            var ignoredDocumented = new System.Collections.Generic.Dictionary<string, int>();
            int replayed = 0;
            NDMatrixBoundary.InternalCallers.Clear();
            NDMatrixBoundary.AuditInternalCallers = true;
            try
            {
                foreach (string file in System.Linq.Enumerable.OrderBy(MinCases.Keys, k => k, System.StringComparer.Ordinal))
                {
                    using var cases = FuzzCorpus.Open(file);
                    foreach (var c in cases)
                    {
                        bool any = false;
                        foreach (var operand in c.Operands)
                        {
                            if (operand?.Shape is { Length: 2 })
                            {
                                operand.Matrix = true;
                                any = true;
                            }
                        }
                        if (!any)
                            continue;
                        replayed++;
                        ReplayOne(c, ignoredFailures, ignoredDocumented);
                        ignoredFailures.Clear();
                    }
                }

                // The numpy.ma tiers: the masked operands are built from matrix data (np.ma.masked_array(matrix,
                // mask)), so every path a raw array takes into the module is walked with a matrix.
                foreach (string file in System.Linq.Enumerable.OrderBy(MaMinCases.Keys, k => k, System.StringComparer.Ordinal))
                {
                    using var cases = FuzzCorpus.Open(file);
                    foreach (var c in cases)
                    {
                        bool any = false;
                        foreach (var operand in c.Operands)
                        {
                            if (operand?.Shape is { Length: 2 })
                            {
                                operand.Matrix = true;
                                any = true;
                            }
                        }
                        if (!any)
                            continue;
                        replayed++;
                        ReplayMaOne(c, ignoredFailures, ignoredDocumented);
                        ignoredFailures.Clear();
                    }
                }

                // The index tiers: the indexers of an NDArray reference keep the plain-array meaning for a matrix
                // object, so a 2-D base, 2-D index arrays and a 2-D value are handed over as matrices.
                foreach (string file in new[] { "index_curated.jsonl", "index_random_20240626.jsonl" })
                {
                    foreach (var c in IndexOracleTests.LoadLines(file))
                    {
                        try
                        {
                            NDArray b = IndexOracleTests.Base(c.GetProperty("base").GetString());
                            if (b.ndim != 2)
                                continue;
                            replayed++;
                            object[] idx = IndexOracleTests.BuildIndex(c.GetProperty("tokens"));
                            for (int i = 0; i < idx.Length; i++)
                                if (idx[i] is NDArray ia && ia.ndim == 2)
                                    idx[i] = (NDArray)np.asmatrix(ia);
                            if (c.GetProperty("op").GetString() == "get")
                            {
                                NDArray matrixBase = np.asmatrix(b);
                                _ = matrixBase[idx];
                            }
                            else
                            {
                                NDArray matrixBase = np.asmatrix(b.copy());
                                NDArray value = IndexOracleTests.BuildValue(c.GetProperty("value"));
                                matrixBase[idx] = value.ndim == 2 ? np.asmatrix(value) : value;
                            }
                        }
                        catch (System.Exception)
                        {
                            // The outcome is not the subject (most of these cases raise in NumPy too): only the call
                            // paths the audit records are.
                        }
                    }
                }
            }
            finally
            {
                NDMatrixBoundary.AuditInternalCallers = false;
            }

            var found = System.Linq.Enumerable.ToList(
                System.Linq.Enumerable.OrderByDescending(NDMatrixBoundary.InternalCallers, kv => kv.Value));
            System.Console.WriteLine($"[matrix boundary audit] {replayed} cases replayed with matrix operands, " +
                                     $"{found.Count} library call paths reached a matrix rule");
            DumpFailures("matrix-boundary-audit",
                System.Linq.Enumerable.ToList(System.Linq.Enumerable.Select(found, kv => $"{kv.Value}\t{kv.Key}")));
            Assert.IsTrue(replayed >= MinBoundaryAuditCases,
                $"only {replayed} cases replayed (floor {MinBoundaryAuditCases}): the corpus shrank or failed to copy");
            Assert.AreEqual(0, found.Count,
                "library methods reached the matrix rule of a public function at depth zero:\n" +
                string.Join("\n", System.Linq.Enumerable.Select(System.Linq.Enumerable.Take(found, 60), kv => $"  {kv.Value,6} x {kv.Key}")));
        }

        /// <summary>
        ///     The fewest cases <see cref="MatrixOperand_LibraryNeverReachesAMatrixRule"/> must replay: a floor
        ///     under the measured count, so an emptied or uncopied corpus cannot pass vacuously.
        /// </summary>
        private const int MinBoundaryAuditCases = 40000;
    }
}
