using System;
using System.Numerics;
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
        public readonly struct MGridResult : INDArrayCarrier
        {
            private readonly NDArray _grid;

            internal MGridResult(NDArray grid) => _grid = grid;

            /// <summary>The dense grid array — a 1-D array for one slice, else the stacked <c>(N, *sizes)</c>.</summary>
            public NDArray Grid => _grid;

            /// <summary>The number of per-axis grids stacked along the first axis (0 for a scalar/empty grid).</summary>
            public int Length => _grid is null || _grid.ndim == 0 ? 0 : (int)_grid.Shape.dimensions[0];

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

            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(_grid);
        }

        /// <summary>
        ///     An instance which returns a <b>dense</b> ("fleshed out") multi-dimensional "meshgrid" when
        ///     indexed, so that every returned axis-grid has the SAME shape. The number of stacked grids and
        ///     their dimensionality equal the number of indexing slices. If the step length is not a complex
        ///     number, the stop is NOT inclusive; a <b>complex</b> step (e.g. <c>"…:5j"</c>) instead specifies
        ///     the number of points, with the stop INCLUSIVE: point <c>i</c> is <c>i * step + start</c> with
        ///     <c>step = (stop - start) / (N - 1)</c>, as NumPy computes it (so the last point can miss the stop
        ///     by a rounding: <c>np.mgrid["0.3:0.9:7j"]</c> ends at 0.9000000000000001, where
        ///     <c>linspace</c> puts it on 0.9).
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.mgrid</c> (the <c>sparse=False</c> instance of
        ///     <c>numpy.lib.index_tricks.nd_grid</c>). It is the DENSE twin of <see cref="ogrid"/> and shares
        ///     its whole grammar and dtype/edge-case behaviour (see <see cref="OGridClass"/>; both run
        ///     NumPy's <c>nd_grid.__getitem__</c> statements): slices are written as strings
        ///     (<c>np.mgrid["0:5", "0:3"]</c> ≙ <c>np.mgrid[0:5, 0:3]</c>); one string may carry comma-separated
        ///     slices; <see cref="Slice"/> objects work; there are NO directives. A field written as digits is a
        ///     Python int (exact at any size), anything else a Python float. The mesh shares ONE dtype (int64
        ///     unless a float or an imaginary step appears, then float64).
        ///     <para>
        ///     <b>Values.</b> A single slice is <c>arange(start, stop, step)</c>, or for an imaginary step
        ///     <c>arange(0, N) * step + start</c> (a fresh array). In a mesh each axis is <c>i * step + start</c> in
        ///     the mesh dtype — for a float step that is NOT arange's <c>start + i * delta</c>:
        ///     <c>np.mgrid["0.5:1.5:0.1", "0:2"][0]</c> holds 0.8, where <c>arange(0.5, 1.5, 0.1)</c> holds
        ///     0.7999999999999999.
        ///     </para>
        ///     <para>
        ///     <b>Errors.</b> A zero step is NumPy's ZeroDivisionError, <see cref="DivideByZeroException"/> with
        ///     Python's text (<c>division by zero</c> over ints, <c>float division by zero</c> once a float is
        ///     involved). A slice of a multi-slice mesh needs its stop unless its step is imaginary with one point:
        ///     NumPy then leaks <c>AttributeError: 'tuple' object has no attribute 'step'</c>, NumSharp raises a
        ///     <see cref="ValueError"/> naming the slice, at the same point (a negative size of another axis can win
        ///     over it, as in NumPy). A single imaginary slice without its stop is Python's <see cref="TypeError"/>
        ///     (<c>unsupported operand type(s) for -: 'NoneType' and 'int'</c>). A NaN, infinite or past-int64 size
        ///     raises NumPy's text (<c>cannot convert float NaN to integer</c>, <c>cannot convert float infinity to
        ///     integer</c>, <c>Maximum allowed dimension exceeded</c>).
        ///     </para>
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

            /// <summary>
            ///     NumPy's <c>nd_grid.__getitem__</c> with <c>sparse=False</c>: a key of one slice is the bare 1-D array of
            ///     the except branch; a key of several computes every size, the shared dtype, the dense
            ///     <c>indices(size, typ)</c> grid, then <c>nn[k] = nn[k] * step + start</c> layer by layer.
            /// </summary>
            /// <param name="key">The indexer's key: slice strings (one may hold several, comma-separated) and
            ///     <see cref="Slice"/> objects.</param>
            /// <returns>The dense grid, of shape <c>(N, size_0, ..., size_{N-1})</c>; the 1-D array of one slice; an empty
            ///     int64 array for the empty key (NumPy's <c>indices([])</c>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
            /// <exception cref="ArgumentException">An entry that is neither a slice string nor a <see cref="Slice"/>.</exception>
            /// <exception cref="TypeError">See <see cref="GridSingleSlice"/>.</exception>
            /// <exception cref="ValueError">A missing stop (<see cref="GridAxisSize"/>, <see cref="GridFillLine"/>); a NaN
            ///     size; a size past int64 (<c>Maximum allowed dimension exceeded</c>); a negative one (np.indices'
            ///     <c>negative dimensions are not allowed</c>); the allocation guard's.</exception>
            /// <exception cref="DivideByZeroException">A zero real step.</exception>
            /// <exception cref="OverflowException">An infinite size; a Python int NumPy cannot convert (see
            ///     <see cref="GridFillLine"/>).</exception>
            [NDScoped] // reclaims each rescale pass's line/reshape/broadcast temps and the grid[k] layer views
            private static NDArray Build(object[] key)
            {
                // Shared nd_grid parsing (same grammar/errors as ogrid).
                var specs = ParseGridSpecs(key, "mgrid");
                int n = specs.Count;

                // A key of one slice takes NumPy's except branch: a bare 1-D array, NOT a stacked grid.
                if (n == 1)
                    return GridSingleSlice(specs[0]);

                // Every size first, in key order: a zero step, a missing (real-step) stop or a NaN size is raised where
                // NumPy's size loop meets it. The sizes are ceil((stop-start)/step) (imaginary: the point count) and MAY be
                // negative - indices then raises "negative dimensions are not allowed", exactly as NumPy's mgrid does (its
                // ogrid twin instead gets an empty axis from arange).
                var sizes = new BigInteger[n];
                for (int k = 0; k < n; k++)
                    sizes[k] = GridAxisSize(specs[k], k, "mgrid");
                NPTypeCode typ = GridMeshDtype(specs);

                // nn = indices(size, typ): NumPy's empty() converts every dimension before it checks any, so a size past
                // int64 is "Maximum allowed dimension exceeded" even beside a negative one. The empty key gives indices([]),
                // an empty int64 array.
                long[] dims = new long[n];
                for (int k = 0; k < n; k++)
                {
                    if (!AxisConcatenator.SliceSpec.FitsInt64(sizes[k]))
                        throw new ValueError("Maximum allowed dimension exceeded");
                    dims[k] = (long)sizes[k];
                }

                NDArray grid = indices(dims, typ);   // (n, size_0, ..., size_{n-1})

                // nn[k] = nn[k] * step + start, layer by layer. A layer holds its coordinate at every position, so the
                // statement is its axis's line broadcast across the grid: the line is computed once and copied in. A "pure"
                // axis (real step 1, start 0) is skipped: nn[k] * 1 + 0 leaves int64 coordinates as they are, and float64
                // ones too (a coordinate is >= 0, so even a -0.0 start adds to +0.0) - the common np.mgrid[0:a, 0:b] case is
                // the single indices fill.
                var fullShape = Shape.Adopt(dims);

                for (int k = 0; k < n; k++)
                {
                    var spec = specs[k];
                    if (spec.Start == 0d && spec.Step == 1d && !spec.ImaginaryStep)
                        continue;

                    NDArray line = new NDArray(typ, dims[k], false);
                    GridFillLine(line, spec, k, typ, "mgrid");

                    long[] lineShape = new long[n];
                    for (int d = 0; d < n; d++)
                        lineShape[d] = 1L;
                    lineShape[k] = dims[k];

                    copyto(grid[k], broadcast_to(line.reshape(Shape.Adopt(lineShape)), fullShape));
                }

                return grid;
            }
        }

        /// <summary>
        ///     Returns a dense multi-dimensional "meshgrid" when indexed — see <see cref="MGridClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html</remarks>
        public static MGridClass mgrid { get; } = new MGridClass();
    }
}
