using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     End-to-end text parity from pandas: a pandas text column (the <c>'str'</c> dtype, <c>object</c>, the nullable
    ///     <c>'string'</c> without a missing value, a DataFrame of text columns) crosses into NumSharp through the bridge's
    ///     pandas adapter as a <c>StringDType()</c> copy, NumSharp computes text operations over it, and LIVE numpy's answer
    ///     over the column's own numpy projection (<c>np.asarray(column.to_numpy(), dtype=StringDType())</c>) is the
    ///     expected one — so the adapter's projection, the copy and NumSharp's operation are proved in one assertion.
    /// </summary>
    /// <remarks>
    ///     The crossing contract itself (what pandas containers share, copy or refuse) is
    ///     <see cref="PandasInteropEdgeCaseTests"/>'; this class holds what a program does with the text once it crossed.
    ///     pandas is an ecosystem package, so the class runs in the <c>ecosystem</c> environment only.
    /// </remarks>
    [TestClass]
    [PythonEcosystem]
    public class TextPandasLiveParityTests : TextLiveParityHarness
    {
        /// <summary>
        ///     The pandas sources (name → Python expression): mixed-case ASCII with spaces, accented and non-BMP text, the
        ///     empty string, an embedded NUL, a text past the 15-byte short-string cell, and digits — in each pandas text
        ///     dtype that crosses, and as a two-column DataFrame.
        /// </summary>
        private static readonly (string Name, string Expr)[] PandasSources =
        {
            ("'str' Series", "pd.Series(_P, dtype='str')"),
            ("object Series", "pd.Series(_P, dtype=object)"),
            ("'string' Series (no NA)", "pd.Series(_P, dtype='string')"),
            ("DataFrame of two text columns", "pd.DataFrame({'a': _P, 'b': _P[::-1]})"),
            ("reversed 'str' Series", "pd.Series(_P, dtype='str').iloc[::-1]"),
        };

        /// <summary>The text operations each pandas source is put through (an elementwise <c>numpy.strings</c> mix).</summary>
        private static readonly TextOp[] Ops =
        {
            E("upper", "np.strings.upper(x)", x => np.strings.upper(x)),
            E("lower", "np.strings.lower(x)", x => np.strings.lower(x)),
            E("str_len", "np.strings.str_len(x)", x => np.strings.str_len(x)),
            Lit("find 'a'", "np.strings.find(x, {0})", "a", (x, s) => np.strings.find(x, s)),
            Lit("startswith 'Pa'", "np.strings.startswith(x, {0})", "Pa", (x, s) => np.strings.startswith(x, s)),
            Lit2("replace a A", "np.strings.replace(x, {0}, {1})", "a", "A", (x, a, b) => np.strings.replace(x, a, b)),
            E("strip", "np.strings.strip(x)", x => np.strings.strip(x)),
            E("slice 1:4", "np.strings.slice(x, 1, 4)", x => np.strings.slice(x, 1, 4)),
            E2("add partner", "np.strings.add(x, p)", (x, p) => np.strings.add(x, p)),
            E2("less partner", "x < p", (x, p) => x < p),
            W("sort", "np.sort(x, axis=None)", (x, _) => np.sort(x, axis: null)),
            W("unique", "np.unique(x)", (x, _) => np.unique(x).values),
            W("astype U12", "x.astype('U12')", (x, _) => x.astype(np.dtype("U12"))),
        };

        /// <summary>
        ///     Every pandas text source × every operation: NumSharp over the adapter's copy equals numpy over the column's
        ///     numpy projection (dtype, shape and values with NA identity, or numpy's exception class and text), and the
        ///     crossings lease nothing.
        /// </summary>
        [TestMethod]
        public void PandasTextColumns_ThroughNumSharpsTextOps_MatchNumpy()
        {
            PandasTestGate.Require(Scope);
            DefineSources("_P = ['Panda', 'a b ', ' x', 'Gr' + chr(0xfc) + 'n', chr(0x1F600) + 'emoji', '', 'nul' + chr(0) + 'in', " +
                          "'a text past the sixteen byte cell', '0042', 'MiXeD']");
            int imports = NDArrayPythonInterop.LiveImports;
            var failures = new List<string>();
            int cells = 0;
            foreach (var (name, expr) in PandasSources)
            {
                // numpy's twin of the column: its own numpy projection as text, C-ordered like the bridge's copy.
                PyExec($"col = {expr}\nx = np.asarray(col.to_numpy(), dtype=StringDType()).copy(order='C')\np = x[::-1] if x.ndim else x");
                NDArray xs;
                using (Gil())
                using (PyObject col = Scope.Get("col"))
                    xs = col.ToNDArray(requireGIL: false);
                using (xs)
                {
                    xs.dtype.Should().Be(np.dtypes.StringDType(), $"{name}: a pandas text column crosses as StringDType()");
                    NDArray ps = xs.ndim > 0 ? xs["::-1"] : xs;
                    foreach (TextOp op in Ops)
                    {
                        cells++;
                        string verdict = RunCell(op, TextClass.StringDType, xs, ps, zeroD: false);
                        if (verdict.Length > 0)
                            failures.Add($"pandas | {name} | {op.Name}: {verdict}");
                    }

                    if (!ReferenceEquals(ps, xs))
                        ps.Dispose();
                }
            }

            PyExec("col = x = p = None");
            AssertNoFailures("pandas", cells, failures);
            NDArrayPythonInterop.LiveImports.Should().Be(imports, "a pandas text column crosses as a copy: nothing is leased");
        }
    }
}
