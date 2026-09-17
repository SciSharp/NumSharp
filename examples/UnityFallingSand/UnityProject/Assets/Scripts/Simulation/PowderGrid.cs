using System;
using NumSharp;

namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The falling-sand cellular automaton: a 2-D grid of <see cref="Cell"/> ids that advances one time
    /// step of gravity-driven material motion per <see cref="Step"/>, entirely through vectorized NumSharp
    /// array math. This is the "physics" of the sand game — every grain, drop and puff is a cell, and the
    /// whole grid updates at once via the primitives in <see cref="GridOps"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>The update is three rule passes, and their difference is the whole model:</b>
    /// (1) a VERTICAL density swap applied to everything — heavier sinks, lighter rises — which alone
    /// produces falling, sinking, floating and rising; (2) a DIAGONAL slide for SAND only, so granular
    /// material forms ~45° piles instead of leveling; (3) a HORIZONTAL spread for FLUIDS only, so liquids
    /// and gases flatten out. Each pass is a mass-conserving swap, so the grid is only ever permuted.
    /// </para>
    /// <para>
    /// <b>Randomized fluid flow is load-bearing, not decorative.</b> A lone fluid cell on a flat surface
    /// has empty on both sides; a fixed left/right preference makes it oscillate in place forever and the
    /// surface never levels. Giving each fluid cell a RANDOM flow direction each step turns that into a
    /// random walk that finds a drop and settles — which is what actually makes water level flat. The
    /// randomness uses the seeded global <c>np.random</c> stream, so a given <see cref="PowderGrid(int,int,int)"/>
    /// seed replays identically.
    /// </para>
    /// <para>
    /// <b>Cost.</b> Each step allocates several full-grid arrays (the price of a pure-vectorized CA), so
    /// keep the grid to the low hundreds per side for interactive frame rates; the sample defaults to a
    /// modest resolution and a small substep count.
    /// </para>
    /// </remarks>
    public sealed class PowderGrid
    {
        /// <summary>The material grid, an <c>(Height, Width)</c> int32 array; each entry is a <see cref="Cell"/> id.</summary>
        public NDArray Cells { get; private set; }

        /// <summary>Grid height in cells (row 0 is the top).</summary>
        public int Height { get; }

        /// <summary>Grid width in cells (column 0 is the left).</summary>
        public int Width { get; }

        private int _step;   // parity counter: alternates the diagonal-slide bias so piles stay symmetric

        /// <summary>
        /// Creates an all-empty grid and seeds the random stream that drives fluid flow direction.
        /// </summary>
        /// <param name="height">Grid height in cells; must be positive.</param>
        /// <param name="width">Grid width in cells; must be positive.</param>
        /// <param name="seed">Seed for <c>np.random</c> so runs are reproducible. Note this sets the GLOBAL stream.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="height"/> or <paramref name="width"/> is not positive.</exception>
        public PowderGrid(int height, int width, int seed = 1)
        {
            if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            Height = height;
            Width = width;
            Cells = np.zeros(new Shape(height, width), NPTypeCode.Int32);
            np.random.seed(seed);   // deterministic fluid flow (global stream — documented)
        }

        /// <summary>
        /// Advances the whole grid by one time step: vertical density motion, then sand's diagonal slide,
        /// then fluids' randomized horizontal spread. Runs entirely as whole-grid NumSharp operations.
        /// </summary>
        public void Step()
        {
            var cells = Cells;

            // 1. VERTICAL (all non-wall): swap down with a strictly-lighter non-wall cell below.
            {
                var dens = DensityOf(cells);
                var notWall = cells != Cell.Wall;
                var canDown = notWall & GridOps.ShiftMask(notWall, 1, 0) & (dens > GridOps.Shift(dens, 1, 0, Cell.WallDensity));
                cells = GridOps.ApplySwap(cells, canDown, 1, 0, Cell.Wall);
            }

            int firstDc = (_step % 2 == 0) ? -1 : 1;   // alternate which diagonal is tried first (symmetry)

            // 2. DIAGONAL (SAND only): slide down-left/right only when straight down is blocked → piling.
            foreach (int dc in new[] { firstDc, -firstDc })
            {
                var dens = DensityOf(cells);
                var notWall = cells != Cell.Wall;
                var isSand = cells == Cell.Sand;
                var canStraight = notWall & GridOps.ShiftMask(notWall, 1, 0) & (dens > GridOps.Shift(dens, 1, 0, Cell.WallDensity));
                var canDiag = isSand & GridOps.ShiftMask(notWall, 1, dc) & (dens > GridOps.Shift(dens, 1, dc, Cell.WallDensity)) & !canStraight;
                cells = GridOps.ApplySwap(cells, canDiag, 1, dc, Cell.Wall);
            }

            // 3. HORIZONTAL (FLUIDS only): spill into an EMPTY side cell when unable to move vertically.
            //    Each fluid cell picks a RANDOM side so lone cells random-walk to a drop and level out.
            {
                var goRight = np.random.rand(Height, Width) >= 0.5;
                for (int pass = 0; pass < 2; pass++)
                {
                    int dc = (pass == 0) ? -1 : 1;
                    var chose = (pass == 0) ? !goRight : goRight;   // each cell attempts only its rolled side
                    var dens = DensityOf(cells);
                    var notWall = cells != Cell.Wall;
                    var isFluid = FluidMaskOf(cells);
                    var canDescend = GridOps.ShiftMask(notWall, 1, 0) & (dens > GridOps.Shift(dens, 1, 0, Cell.WallDensity));
                    var canAscend = GridOps.ShiftMask(notWall, -1, 0) & (GridOps.Shift(dens, -1, 0, Cell.WallDensity) > dens);
                    var sideEmpty = GridOps.Shift(cells, 0, dc, Cell.Wall) == Cell.Empty;
                    var move = isFluid & chose & sideEmpty & !canDescend & !canAscend;
                    cells = GridOps.ApplySwap(cells, move, 0, dc, Cell.Wall);
                }
            }

            Cells = cells;
            _step++;
        }

        /// <summary>
        /// Builds the per-cell density field from the material grid, generically from
        /// <see cref="Cell.DensityTable"/> so no material is hard-coded. Empty (density 0) is the default.
        /// </summary>
        /// <param name="cells">The material grid.</param>
        /// <returns>An int array of densities, same shape as <paramref name="cells"/>.</returns>
        private static NDArray DensityOf(NDArray cells)
        {
            var dens = np.zeros_like(cells);
            // Start at id 1: id 0 (Empty) has density 0, already the zero default.
            for (int id = 1; id < Cell.Count; id++)
                dens = np.where(cells == id, Cell.Density(id), dens);
            return dens;
        }

        /// <summary>Builds a boolean "is a fluid" mask from the material grid, generically from <see cref="Cell.IsFluid"/>.</summary>
        /// <param name="cells">The material grid.</param>
        /// <returns>A bool array, true where the cell is a spreading fluid (water/oil/smoke).</returns>
        private static NDArray FluidMaskOf(NDArray cells)
        {
            var fl = np.zeros(cells.shape, NPTypeCode.Boolean);
            for (int id = 1; id < Cell.Count; id++)
                if (Cell.IsFluid(id)) fl = fl | (cells == id);
            return fl;
        }

        /// <summary>
        /// Stamps a filled disk of <paramref name="material"/> into the grid — the brush behind the game's
        /// mouse painting. Out-of-range parts of the disk are clipped to the grid; the disk overwrites
        /// whatever was there (so an <see cref="Cell.Empty"/> brush is an eraser).
        /// </summary>
        /// <param name="row">Disk centre row.</param>
        /// <param name="col">Disk centre column.</param>
        /// <param name="radius">Disk radius in cells (0 paints a single cell).</param>
        /// <param name="material">The material id to write.</param>
        public void Paint(int row, int col, int radius, int material)
        {
            if (radius < 0) radius = 0;
            int r0 = Math.Max(0, row - radius), r1 = Math.Min(Height, row + radius + 1);
            int c0 = Math.Max(0, col - radius), c1 = Math.Min(Width, col + radius + 1);
            if (r1 <= r0 || c1 <= c0) return;   // disk entirely off-grid

            // Build a circular mask over the clipped bounding box and write the material where inside it.
            var rows = np.arange(r0, r1).reshape(r1 - r0, 1);   // (h',1)
            var cols = np.arange(c0, c1).reshape(1, c1 - c0);   // (1,w')
            var dr = rows - row;
            var dc = cols - col;
            var dist2 = dr * dr + dc * dc;                      // (h',w') via broadcasting
            var mask = dist2 <= radius * radius;
            var sub = Cells[$"{r0}:{r1}, {c0}:{c1}"];
            Cells[$"{r0}:{r1}, {c0}:{c1}"] = np.where(mask, material, sub);
        }

        /// <summary>Writes a single cell (used by emitters). No-op if out of range.</summary>
        /// <param name="row">Row.</param>
        /// <param name="col">Column.</param>
        /// <param name="material">Material id to write.</param>
        public void SetCell(int row, int col, int material)
        {
            if ((uint)row >= (uint)Height || (uint)col >= (uint)Width) return;
            Cells[row, col] = material;
        }

        /// <summary>Reads a single cell's material id.</summary>
        /// <param name="row">Row.</param>
        /// <param name="col">Column.</param>
        /// <returns>The material id, or <see cref="Cell.Wall"/> for out-of-range (treating the border as solid).</returns>
        public int GetCell(int row, int col)
        {
            if ((uint)row >= (uint)Height || (uint)col >= (uint)Width) return Cell.Wall;
            return (int)Cells[row, col];
        }

        /// <summary>Counts the cells of one material — used for the HUD readout and the mass-conservation gate.</summary>
        /// <param name="material">Material id.</param>
        /// <returns>The number of cells currently holding that material.</returns>
        public int Count(int material) => (int)np.sum((Cells == material).astype(NPTypeCode.Int32));

        /// <summary>Empties the grid.</summary>
        /// <param name="keepWalls">When true, walls are preserved and everything else is cleared; when false, the whole grid is emptied.</param>
        public void Clear(bool keepWalls)
        {
            Cells = keepWalls
                ? np.where(Cells == Cell.Wall, Cell.Wall, Cell.Empty)
                : np.zeros(new Shape(Height, Width), NPTypeCode.Int32);
        }

        /// <summary>Copies the grid to a flat managed array in row-major (C) order — the renderer's per-frame read.</summary>
        /// <returns>A new <c>int[Height*Width]</c> with <c>[r*Width + c]</c> = cell (r,c).</returns>
        public int[] Snapshot() => Cells.ToArray<int>();
    }
}
