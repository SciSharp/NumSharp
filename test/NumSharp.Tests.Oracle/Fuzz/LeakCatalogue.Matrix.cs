using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for the <see cref="NDMatrix"/> members the matrix tier (<c>matrix.jsonl</c>) does not
    ///     reach through an op key of the same name: the object-protocol members (<c>Clone</c>, <c>Equals</c>,
    ///     <c>GetHashCode</c>, <c>GetEnumerator</c>), the indexers, the implicit scalar conversions, the logical-not
    ///     operator (a NumSharp spelling with no Python dunder) and the three inherited methods the tier leaves to
    ///     their ndarray cases (<c>argpartition</c>, <c>choose</c>, <c>getfield</c>, <c>to_device</c>).
    /// </summary>
    /// <remarks>
    ///     Each entry builds its matrix over a fixture array and releases it before it returns: a matrix view
    ///     holds a counted reference on the fixture's buffer, which the result disposer would not find.
    /// </remarks>
    internal static partial class LeakCatalogue
    {
        /// <summary>NDMatrix members outside the matrix tier's op keys.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddMatrix(List<LeakCase> l)
        {
            E(l, "NDMatrix.Clone", "matrix view", f =>
            {
                using var m = np.asmatrix(f.M);
                return m.Clone();
            });
            E(l, "NDMatrix.Equals", "against its own data", f =>
            {
                using var m = np.asmatrix(f.M);
                return Box(m.Equals((object)f.M));
            });
            T(l, "NDMatrix.GetHashCode", "raises by design (an array is mutable, so unhashable, as in NumPy)", f =>
            {
                using var m = np.asmatrix(f.M);
                return Box(m.GetHashCode());
            });
            E(l, "NDMatrix.GetEnumerator", "rows as (1, N) matrices", f =>
            {
                using var m = np.asmatrix(f.M);
                long elements = 0;
                foreach (NDMatrix row in m)
                {
                    elements += row.size;
                    row.Dispose();
                }

                return Box(elements);
            });
            E(l, "NDMatrix.Item", "get: row, element, column, index array, mask", f =>
            {
                using var m = np.asmatrix(f.M);
                using var mask = m > 5;
                return new object[] { m[1], m[1, 2], m[":, 1"], m[new object[] { f.Idx }], m[new object[] { (NDArray)mask }] };
            });
            E(l, "NDMatrix.Item", "set: row, element, column, index array on an owned copy", f =>
            {
                var m = np.matrix(f.M);
                m[0] = 1.5;
                m[1, 2] = 2.5;
                m[":, 1"] = 3.5;
                m[new object[] { f.Idx }] = 4.5;
                return m;
            });
            E(l, "NDMatrix.argpartition", "last axis", f =>
            {
                using var m = np.asmatrix(f.M);
                return m.argpartition(1);
            });
            E(l, "NDMatrix.choose", "two choices", f =>
            {
                using var picks = np.array(new long[,] { { 0, 1 }, { 1, 0 } });
                using var m = np.asmatrix(picks);
                using var zeros = np.zeros(new Shape(2, 2));
                using var ones = np.ones(new Shape(2, 2));
                return m.choose(new[] { zeros, ones });
            });
            E(l, "NDMatrix.getfield", "the whole element", f =>
            {
                using var m = np.asmatrix(f.M);
                return m.getfield(np.float64);
            });
            E(l, "NDMatrix.to_device", "cpu returns the matrix itself", f =>
            {
                using var m = np.asmatrix(f.M);
                return Box(ReferenceEquals(m.to_device("cpu"), m));
            });
            E(l, "NDMatrix.op_Implicit", "a C# number as the 0-d scalar stand-in", f =>
            {
                NDMatrix scalar = 2.5;
                return scalar;
            });
            E(l, "NDMatrix.op_LogicalNot", "bool matrix", f =>
            {
                using var m = np.asmatrix(f.M);
                return !m;
            });
        }
    }
}
