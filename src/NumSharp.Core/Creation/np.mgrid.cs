using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The result of an <see cref="mgrid"/> index expression: a <b>dense</b> mesh — one array. NumPy's
        ///     <c>mgrid</c> is a single <c>ndarray</c> (a 1-D array for one slice; a stacked
        ///     <c>(N, *sizes)</c> array for several), and the idiom <c>X, Y = np.mgrid[…]</c> works by
        ///     iterating that array's first axis. This value carries the array and reproduces both spellings:
        ///     it converts implicitly to the bare <see cref="NDArray"/>, and it <c>Deconstruct</c>s
        ///     (<c>var (x, y) = np.mgrid["0:5", "0:3"];</c>) or indexes (<c>[k]</c>) into the per-axis grids
        ///     along that first axis.
        /// </summary>
        public readonly struct MGridResult
        {
            private readonly NDArray _grid;

            internal MGridResult(NDArray grid) => _grid = grid;

            /// <summary>The dense grid array — a 1-D array for one slice, else the stacked <c>(N, *sizes)</c>.</summary>
            public NDArray Grid => _grid;

            /// <summary>The number of per-axis grids stacked along the first axis (0 for a scalar/empty grid).</summary>
            public int Length => _grid is null || _grid.ndim == 0 ? 0 : (int)_grid.shape[0];

            /// <summary>The k-th per-axis grid — <c>grid[k]</c> along the first axis.</summary>
            public NDArray this[int index] => _grid[index];

            /// <summary>Yields the whole dense grid — the array NumPy's <c>mgrid[…]</c> returns.</summary>
            public static implicit operator NDArray(MGridResult result) => result._grid;

            /// <summary>Deconstructs a two-axis grid: <c>var (x, y) = np.mgrid["0:5", "0:3"];</c></summary>
            public void Deconstruct(out NDArray item1, out NDArray item2)
            {
                EnsureArity(2);
                item1 = _grid[0];
                item2 = _grid[1];
            }

            /// <summary>Deconstructs a three-axis grid.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3)
            {
                EnsureArity(3);
                item1 = _grid[0];
                item2 = _grid[1];
                item3 = _grid[2];
            }

            /// <summary>Deconstructs a four-axis grid.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3, out NDArray item4)
            {
                EnsureArity(4);
                item1 = _grid[0];
                item2 = _grid[1];
                item3 = _grid[2];
                item4 = _grid[3];
            }

            private void EnsureArity(int n)
            {
                int have = Length;
                if (have != n)
                    throw new InvalidOperationException(
                        $"np.mgrid produced a grid of {have} axes; cannot deconstruct into {n}. A single-slice " +
                        "mgrid is a bare 1-D array — use it as an NDArray, not a tuple.");
            }
        }

        /// <summary>
        ///     An instance which returns a <b>dense</b> ("fleshed out") multi-dimensional "meshgrid" when
        ///     indexed, so that every returned axis-grid has the SAME shape. The number of stacked grids and
        ///     their dimensionality equal the number of indexing slices. If the step length is not a complex
        ///     number, the stop is NOT inclusive; a <b>complex</b> step (e.g. <c>"…:5j"</c>) instead specifies
        ///     the number of points, with the stop INCLUSIVE (i.e. <see cref="linspace"/>).
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.mgrid</c> (the <c>sparse=False</c> instance of
        ///     <c>numpy.lib.index_tricks.nd_grid</c>). It is the DENSE twin of <see cref="ogrid"/> and shares
        ///     its whole grammar and dtype/edge-case behaviour (see <see cref="OGridClass"/> and the shared
        ///     <c>NdGridLines</c> builder): slices are written as strings (<c>np.mgrid["0:5", "0:3"]</c> ≙
        ///     <c>np.mgrid[0:5, 0:3]</c>); one string may carry comma-separated slices; <see cref="Slice"/>
        ///     objects work; there are NO directives. The mesh shares ONE dtype (int64 unless a float literal
        ///     or imaginary step appears, then float64); a multi-slice mesh requires an explicit stop on every
        ///     slice.
        ///     <para>
        ///     <b>Shape.</b> A single slice returns a bare 1-D array; N slices return the stacked array of
        ///     shape <c>(N, size_0, …, size_{N-1})</c> — <c>result[k]</c> is the k-th coordinate broadcast
        ///     across the whole grid. It equals <c>stack([broadcast_to(o, full) for o in ogrid[…]], axis:0)</c>
        ///     (NumPy builds it as <c>indices(sizes)</c> then rescales each layer — same values, same layout).
        ///     </para>
        ///     <para>
        ///     <b>Return.</b> The <see cref="MGridResult"/> is NumPy's single array: it converts implicitly to
        ///     <see cref="NDArray"/> and <c>Deconstruct</c>s / indexes into the per-axis grids.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
        /// </remarks>
        /// <example>
        /// <code>
        /// NDArray x        = np.mgrid["-1:1:5j"];      // array([-1., -0.5, 0., 0.5, 1.])
        /// var (rows, cols) = np.mgrid["0:5", "0:3"];   // each shape (5, 3)
        /// NDArray stacked  = np.mgrid["0:5", "0:3"];   // shape (2, 5, 3)
        /// </code>
        /// </example>
        public sealed class MGridClass
        {
            internal MGridClass() { }

            /// <summary>
            ///     Expands the slice expression into a dense mesh. See <see cref="MGridClass"/> for how a
            ///     Python slice literal is spelled in C#.
            /// </summary>
            /// <param name="key">
            ///     Slice-expression strings (a colon-bearing <c>"start:stop[:step]"</c>, or several such
            ///     comma-separated), and/or <see cref="Slice"/> objects.
            /// </param>
            public MGridResult this[params object[] key] => new MGridResult(Build(key));

            private static NDArray Build(object[] key)
            {
                // Shared nd_grid parsing (same grammar/errors as ogrid).
                var specs = ParseGridSpecs(key, "mgrid");
                int n = specs.Count;

                // mgrid[()] -> shape (0,), matching NumPy's indices([]) tail.
                if (n == 0)
                    return new NDArray(NPTypeCode.Int64, new Shape(0), false);

                // A single slice is a bare 1-D array (NumPy's except-branch), NOT a stacked grid.
                if (n == 1)
                    return specs[0].Materialize();

                NPTypeCode typ = GridMeshDtype(specs, "mgrid");

                // NumPy builds the dense grid as indices(sizes, typ) then rescales each layer. The sizes
                // are ceil((stop-start)/step) (imaginary: the point count) and MAY be negative — indices
                // then raises "negative dimensions are not allowed", exactly as NumPy's mgrid does (its
                // ogrid twin instead clamps via arange). A zero real step is a divide-by-zero.
                int[] sizes = new int[n];
                for (int k = 0; k < n; k++)
                    sizes[k] = checked((int)MeshAxisSize(specs[k]));

                NDArray grid = indices(sizes, typ);   // (n, size_0, …, size_{n-1})

                // Rescale each NON-trivial axis by overwriting its layer with the 1-D line broadcast
                // across the whole grid. A "pure" axis (start 0, step 1, real) already equals indices'
                // coordinates, so it is skipped — the common np.mgrid[0:a, 0:b] case is a single fused
                // indices fill with no post-pass, and this reuses SliceSpec's exact arange/linspace
                // values instead of a dtype-branching multiply-add.
                long[] full = new long[n];
                for (int k = 0; k < n; k++)
                    full[k] = sizes[k];
                var fullShape = new Shape(full);

                for (int k = 0; k < n; k++)
                {
                    var spec = specs[k];
                    if (spec.Start == 0d && spec.Step == 1d && !spec.ImaginaryStep)
                        continue;

                    NDArray line = spec.Materialize();
                    if (line.typecode != typ)
                        line = line.astype(typ);

                    long[] lineShape = new long[n];
                    for (int d = 0; d < n; d++)
                        lineShape[d] = 1L;
                    lineShape[k] = line.size;

                    copyto(grid[k], broadcast_to(line.reshape(lineShape), fullShape));
                }

                return grid;
            }

            /// <summary>
            ///     NumPy's per-axis size for a dense grid: <c>ceil((stop-start)/step)</c> for a real step
            ///     (the result MAY be negative, which <see cref="indices"/> rejects), or the point count
            ///     <c>int(abs(step))</c> for an imaginary one. A zero real step is a divide-by-zero,
            ///     matching NumPy's <c>ZeroDivisionError</c>.
            /// </summary>
            private static long MeshAxisSize(AxisConcatenator.SliceSpec spec)
            {
                if (spec.ImaginaryStep)
                    return (long)Math.Abs(spec.Step);
                if (spec.Step == 0d)
                    throw new DivideByZeroException("mgrid slice step cannot be zero.");
                return (long)Math.Ceiling((spec.Stop - spec.Start) / spec.Step);
            }
        }

        /// <summary>
        ///     Returns a dense multi-dimensional "meshgrid" when indexed — see <see cref="MGridClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html</remarks>
        public static MGridClass mgrid { get; } = new MGridClass();
    }
}
