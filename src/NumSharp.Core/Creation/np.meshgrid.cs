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
            => meshgrid(new[] { x1, x2 }, indexing, sparse, copy);

        /// <summary>
        ///     Return coordinate matrices from three coordinate vectors — see
        ///     <see cref="meshgrid(NDArray[], string, bool, bool)"/>.
        /// </summary>
        public static MeshgridResult meshgrid(NDArray x1, NDArray x2, NDArray x3,
            string indexing = "xy", bool sparse = false, bool copy = true)
            => meshgrid(new[] { x1, x2, x3 }, indexing, sparse, copy);

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
        ///     <c>[]</c>, deconstructable, and indexable.
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
        public static MeshgridResult meshgrid(NDArray[] xi,
            string indexing = "xy", bool sparse = false, bool copy = true)
        {
            if (xi is null)
                throw new ArgumentNullException(nameof(xi));
            if (indexing != "xy" && indexing != "ij")
                throw new ValueError("Valid values for `indexing` are 'xy' and 'ij'.");

            int ndim = xi.Length;
            if (ndim == 0)
                return new MeshgridResult(Array.Empty<NDArray>());

            var output = new NDArray[ndim];

            // Reshape each input to its open-mesh axis (size at that axis, 1 elsewhere) in ONE pass.
            // 'ij' (matrix) puts input i at axis i; 'xy' (Cartesian) swaps the first two — input 0 → axis 1,
            // input 1 → axis 0 — for ndim > 1.
            bool swapXY = indexing == "xy" && ndim > 1;
            for (int i = 0; i < ndim; i++)
            {
                NDArray a = xi[i] ?? throw new ArgumentNullException($"xi[{i}]",
                    "meshgrid coordinate vectors must not be null.");
                int axis = swapXY ? (i == 0 ? 1 : i == 1 ? 0 : i) : i;
                output[i] = a.reshape(OpenMeshShape(ndim, axis, a.size));
            }

            // Dense grids: broadcast each open-mesh axis across the full shape.
            if (!sparse)
                output = broadcast_arrays(output);

            // Materialize to independent C-contiguous arrays unless views were requested.
            if (copy)
                for (int i = 0; i < ndim; i++)
                    output[i] = output[i].copy();

            return new MeshgridResult(output);
        }

        /// <summary>Shape that is 1 in every axis but <paramref name="axis"/>, which holds <paramref name="size"/>.</summary>
        private static long[] OpenMeshShape(int ndim, int axis, long size)
        {
            var shape = new long[ndim];
            for (int d = 0; d < ndim; d++)
                shape[d] = 1L;
            shape[axis] = size;
            return shape;
        }
    }
}
