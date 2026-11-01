using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="meshgrid(NDArray[], string, bool, bool)"/>: a tuple of N coordinate
        ///     grids (NumPy returns a Python tuple). C# cannot return a variadic tuple, so this value stands
        ///     in for it — it converts implicitly to <see cref="NDArray"/><c>[]</c>, <c>Deconstruct</c>s
        ///     (<c>var (xx, yy) = np.meshgrid(x, y);</c>) and indexes (<c>[k]</c>).
        /// </summary>
        public readonly struct MeshgridResult : INDArrayCarrier
        {
            private readonly NDArray[] _grids;

            internal MeshgridResult(NDArray[] grids) => _grids = grids ?? Array.Empty<NDArray>();

            /// <summary>Number of coordinate grids (one per input vector).</summary>
            public int Length => _grids?.Length ?? 0;

            /// <summary>The k-th coordinate grid.</summary>
            public NDArray this[int index] => (_grids ?? Array.Empty<NDArray>())[index];

            /// <summary>Returns the coordinate grids as an <see cref="NDArray"/><c>[]</c>.</summary>
            public NDArray[] ToArray() => _grids ?? Array.Empty<NDArray>();

            /// <summary>Exposes the coordinate grids — the tuple NumPy's <c>meshgrid</c> returns.</summary>
            public static implicit operator NDArray[](MeshgridResult result) => result.ToArray();

            /// <summary>Deconstructs a two-grid result: <c>var (xx, yy) = np.meshgrid(x, y);</c></summary>
            public void Deconstruct(out NDArray item1, out NDArray item2)
            {
                EnsureArity(2);
                item1 = _grids[0];
                item2 = _grids[1];
            }

            /// <summary>Deconstructs a three-grid result.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3)
            {
                EnsureArity(3);
                item1 = _grids[0];
                item2 = _grids[1];
                item3 = _grids[2];
            }

            /// <summary>Deconstructs a four-grid result.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3, out NDArray item4)
            {
                EnsureArity(4);
                item1 = _grids[0];
                item2 = _grids[1];
                item3 = _grids[2];
                item4 = _grids[3];
            }

            private void EnsureArity(int n)
            {
                int have = Length;
                if (have != n)
                    throw new InvalidOperationException(
                        $"np.meshgrid produced {have} grids; cannot deconstruct into {n}. Use indexing or the " +
                        "NDArray[] conversion.");
            }

            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(_grids);
        }

        /// <summary>
        ///     Return coordinate matrices from two coordinate vectors — see
        ///     <see cref="meshgrid(NDArray[], string, bool, bool)"/>.
        /// </summary>
        public static MeshgridResult meshgrid(NDArray x1, NDArray x2,
            string indexing = "xy", bool sparse = false, bool copy = true)
        {
            // A numpy.char.chararray operand: the array overload's rule (each grid keeps its own input's class), decided
            // before PlainFor's region would hide the chararray.
            if (NDCharArrayBoundary.Sees(x1, x2))
                return meshgrid(new[] { x1, x2 }, indexing, sparse, copy);
            using var plain = NDMatrixBoundary.PlainFor(x1, x2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return meshgrid(new[] { x1, x2 }, indexing, sparse, copy);
        }

        /// <summary>
        ///     Return coordinate matrices from three coordinate vectors — see
        ///     <see cref="meshgrid(NDArray[], string, bool, bool)"/>.
        /// </summary>
        public static MeshgridResult meshgrid(NDArray x1, NDArray x2, NDArray x3,
            string indexing = "xy", bool sparse = false, bool copy = true)
        {
            // A numpy.char.chararray operand: the array overload's rule, decided before PlainFor's region hides it.
            if (NDCharArrayBoundary.Sees(x1, x2, x3))
                return meshgrid(new[] { x1, x2, x3 }, indexing, sparse, copy);
            using var plain = NDMatrixBoundary.PlainFor(x1, x2, x3);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return meshgrid(new[] { x1, x2, x3 }, indexing, sparse, copy);
        }

        /// <summary>
        ///     Return a tuple of coordinate matrices from N coordinate vectors — make N-D coordinate arrays
        ///     for vectorized evaluation of N-D fields over an N-D grid.
        /// </summary>
        /// <param name="xi">
        ///     The coordinate vectors. Each is flattened to 1-D, so a higher-rank input is read in C-order.
        ///     For the two- and three-vector cases the <c>x1, x2[, x3]</c> overloads let them be passed
        ///     directly (<c>np.meshgrid(x, y)</c>); use this overload for four or more.
        /// </param>
        /// <param name="indexing">
        ///     <c>"xy"</c> (Cartesian, default) or <c>"ij"</c> (matrix). For two inputs of length M and N the
        ///     outputs are <c>(N, M)</c> under <c>"xy"</c> and <c>(M, N)</c> under <c>"ij"</c>; the two
        ///     conventions swap the first two axes. Has no effect for a single input.
        /// </param>
        /// <param name="sparse">
        ///     If true, grid <c>i</c> keeps the open-mesh shape <c>(1, …, Ni, …, 1)</c> instead of the full
        ///     <c>(N1, …, Nn)</c> — these broadcast to the same dense result. Default false.
        /// </param>
        /// <param name="copy">
        ///     If true (default) each grid is an independent C-contiguous array. If false the dense grids are
        ///     returned as broadcast VIEWS (non-contiguous, and multiple elements may alias one memory
        ///     location — copy before writing).
        /// </param>
        /// <returns>
        ///     N grids as a <see cref="MeshgridResult"/> (NumPy's tuple): implicit to <see cref="NDArray"/>
        ///     <c>[]</c>, deconstructable, and indexable. The grid of a <c>numpy.char.chararray</c> input is a chararray
        ///     (NumPy keeps each input's class), the grid of a plain input a plain array.
        /// </returns>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.meshgrid</c> (<c>numpy/lib/_function_base_impl.py</c>). Each input
        ///     is reshaped to its open-mesh axis; <c>"xy"</c> then swaps the placement of the first two;
        ///     unless <paramref name="sparse"/> the grids are broadcast to the full shape; unless
        ///     <paramref name="copy"/> is false they are then materialized. Each grid PRESERVES its input's
        ///     dtype (grids are not promoted to a common type). The companion open-mesh builder is
        ///     <see cref="ix_"/>; the indexing-notation forms are <see cref="mgrid"/> / <see cref="ogrid"/>.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        /// </remarks>
        /// <example>
        /// <code>
        /// var (xx, yy) = np.meshgrid(np.arange(3), np.arange(2));  // xx,yy shape (2,3), 'xy'
        /// var (i, j)   = np.meshgrid(a, b, indexing: "ij");        // shape (len a, len b)
        /// NDArray[] g  = np.meshgrid(a, b, sparse: true);          // (1,M) and (N,1)
        /// </code>
        /// </example>
        /// <exception cref="ArgumentNullException"><paramref name="xi"/> is null, or one of its vectors is (the parameter is
        ///     then named <c>xi[i]</c>, after the indexing check).</exception>
        /// <exception cref="ValueError"><paramref name="indexing"/> is neither <c>"xy"</c> nor <c>"ij"</c> (<c>Valid values
        ///     for `indexing` are 'xy' and 'ij'.</c>).</exception>
        public static MeshgridResult meshgrid(NDArray[] xi,
            string indexing = "xy", bool sparse = false, bool copy = true)
        {
            // A numpy.char.chararray operand: NumPy builds grid i from np.asanyarray(xi[i]) (reshape,
            // broadcast_arrays(subok=True), copy), so each grid keeps ITS input's class — a chararray input gives a
            // chararray grid (owning with copy=True, a view without), a plain one a plain grid (NDCharArrayBoundary).
            if (NDCharArrayBoundary.Sees(xi))
                return new MeshgridResult(NDCharArrayBoundary.KeptPerOperand((xi, indexing, sparse, copy),
                    static s => meshgrid(s.xi, s.indexing, s.sparse, s.copy).ToArray(), xi));
            using var plain = NDMatrixBoundary.PlainFor(xi);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (xi is null)
                throw new ArgumentNullException(nameof(xi));

            // NumPy's statements (the indexing check, the open-mesh reshapes, the broadcast, the copies) are the array
            // object's; the tuple NumPy returns is this result struct.
            return new MeshgridResult(NDArray.meshgrid(xi, indexing, sparse, copy));
        }
    }
}
