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
    /// <b>The update is three MOVEMENT passes, and their difference is the whole model:</b>
    /// (1) a VERTICAL density swap applied to everything — heavier sinks, lighter rises — which alone
    /// produces falling, sinking, floating and rising; (2) a DIAGONAL slide for SAND only, so granular
    /// material forms ~45° piles instead of leveling; (3) a HORIZONTAL spread for FLUIDS only, so liquids
    /// and gases flatten out. Each pass is a mass-conserving swap, so the grid is only ever permuted.
    /// </para>
    /// <para>
    /// <b>A fourth REACTION pass runs after movement, and it is deliberately NOT mass-conserving.</b> It is
    /// the chemistry layer that makes fire and lava fun: oil ignites next to fire/lava, fire burns out to
    /// smoke after a short random lifetime, and lava quenches against water into obsidian while the water
    /// flashes to steam. It transforms materials rather than moving them, so it changes cell counts by
    /// design — exactly like emitters. It early-outs the instant no fire or lava is present, so a world of
    /// only the classic materials is byte-for-byte identical to the pre-reaction engine (the mass-conservation
    /// gate, which uses no fire/lava, is therefore untouched).
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
            React();     // 4. transformations (fire/lava chemistry) — no-op unless fire or lava is present
            _step++;
        }

        /// <summary>
        /// The reaction pass: material TRANSFORMATIONS layered on top of the mass-conserving movement. Unlike
        /// the three movement passes this changes cell counts on purpose (it is a chemistry step, not a swap),
        /// and it runs only when there is something to react — the first thing it does is check for fire or
        /// lava and return immediately if neither exists, so a world of only the classic materials is left
        /// exactly as the movement passes produced it (and no random numbers are drawn, keeping such worlds
        /// bit-for-bit reproducible).
        /// </summary>
        /// <remarks>
        /// <para>
        /// All the reaction conditions are read from the SAME pre-reaction grid and then applied together, so
        /// the outcome is independent of the order the transformations are written. The four rules act on
        /// four DISJOINT source materials (lava, water, oil, fire), so no cell is claimed by two rules:
        /// </para>
        /// <list type="bullet">
        /// <item><description><b>Quench:</b> lava touching water → <see cref="Cell.Wall"/> (obsidian).</description></item>
        /// <item><description><b>Steam:</b> water touching lava → <see cref="Cell.Smoke"/>.</description></item>
        /// <item><description><b>Ignite:</b> oil touching fire or lava → <see cref="Cell.Fire"/> (a random subset each step, so flames flicker rather than convert a whole slick in one frame).</description></item>
        /// <item><description><b>Burn out:</b> fire → <see cref="Cell.Smoke"/> with a per-step probability, giving fire a short finite lifetime.</description></item>
        /// </list>
        /// <para>
        /// The two random subsets (ignite, burn out) are drawn only on the reactive branch, so they never
        /// perturb the fluid-flow random stream of a fire/lava-free world.
        /// </para>
        /// </remarks>
        private void React()
        {
            var cells = Cells;
            var isFire = cells == Cell.Fire;
            var isLava = cells == Cell.Lava;
            // Nothing here reacts without an igniter (fire) or lava, so bail before drawing any randomness.
            if (!AnyTrue(isFire) && !AnyTrue(isLava)) return;

            var isWater = cells == Cell.Water;
            var isOil = cells == Cell.Oil;
            var hotNeighbour = Neighbor4(isFire | isLava);   // cells adjacent to something burning/molten
            var waterNeighbour = Neighbor4(isWater);          // cells adjacent to water
            var lavaNeighbour = Neighbor4(isLava);            // cells adjacent to lava

            var result = cells;
            // Lava that meets water freezes to obsidian; the water it met flashes to steam. Both read the
            // pre-reaction masks, so the pairing is symmetric regardless of which line is written first.
            result = np.where(isLava & waterNeighbour, Cell.Wall, result);
            result = np.where(isWater & lavaNeighbour, Cell.Smoke, result);
            // Oil next to a flame or lava catches — but only a random fraction each step, so a slick lights
            // up as a spreading, flickering front instead of turning to fire all at once.
            var ignite = isOil & hotNeighbour & (np.random.rand(Height, Width) < 0.9);
            result = np.where(ignite, Cell.Fire, result);
            // Fire has a short life: each step a random share of it decays to smoke.
            var burnout = isFire & (np.random.rand(Height, Width) < 0.25);
            result = np.where(burnout, Cell.Smoke, result);

            Cells = result;
        }

        /// <summary>Whether any cell of a boolean mask is true, via a whole-grid reduction (used to early-out the reaction pass).</summary>
        /// <param name="mask">A 2-D bool array.</param>
        /// <returns>True if at least one entry is set.</returns>
        private static bool AnyTrue(NDArray mask) => (int)np.sum(mask.astype(NPTypeCode.Int32)) > 0;

        /// <summary>The 4-neighbour dilation of a mask: true wherever an up/down/left/right neighbour is true. This is how a reaction asks "is this cell touching X?" for the whole grid at once.</summary>
        /// <param name="mask">A 2-D bool array.</param>
        /// <returns>A new bool array, true where any orthogonal neighbour of <paramref name="mask"/> is set.</returns>
        private static NDArray Neighbor4(NDArray mask) =>
            GridOps.ShiftMask(mask, 1, 0) | GridOps.ShiftMask(mask, -1, 0) |
            GridOps.ShiftMask(mask, 0, 1) | GridOps.ShiftMask(mask, 0, -1);

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
        /// Stamps a filled <see cref="BrushShape.Disk"/> of <paramref name="material"/> into the grid — the
        /// disk overload kept for the many callers (verification, scene seeding) that only ever want a round
        /// brush. Equivalent to <see cref="Paint(int,int,int,int,BrushShape)"/> with <see cref="BrushShape.Disk"/>.
        /// </summary>
        /// <param name="row">Disk centre row.</param>
        /// <param name="col">Disk centre column.</param>
        /// <param name="radius">Disk radius in cells (0 paints a single cell).</param>
        /// <param name="material">The material id to write.</param>
        public void Paint(int row, int col, int radius, int material) => Paint(row, col, radius, material, BrushShape.Disk);

        /// <summary>
        /// Stamps the brush of a chosen <paramref name="shape"/> into the grid — the brush behind the game's
        /// mouse painting. Out-of-range parts are clipped to the grid; the stamp overwrites whatever was there
        /// (so an <see cref="Cell.Empty"/> brush is an eraser). A square brush is what you want for drawing
        /// crisp walls; a disk for pouring loose material.
        /// </summary>
        /// <param name="row">Brush centre row.</param>
        /// <param name="col">Brush centre column.</param>
        /// <param name="radius">Brush radius in cells (0 paints a single cell); the square's side is <c>2·radius+1</c>.</param>
        /// <param name="material">The material id to write.</param>
        /// <param name="shape">The footprint to stamp (<see cref="BrushShape.Disk"/> or <see cref="BrushShape.Square"/>).</param>
        public void Paint(int row, int col, int radius, int material, BrushShape shape)
        {
            if (radius < 0) radius = 0;
            int r0 = Math.Max(0, row - radius), r1 = Math.Min(Height, row + radius + 1);
            int c0 = Math.Max(0, col - radius), c1 = Math.Min(Width, col + radius + 1);
            if (r1 <= r0 || c1 <= c0) return;   // brush entirely off-grid

            if (shape == BrushShape.Square)
            {
                // The whole clipped bounding box IS the square footprint — one scalar-broadcast slice write.
                FillRect(r0, c0, r1, c1, material);
                return;
            }

            // Disk: build a circular mask over the clipped bounding box and write the material where inside it.
            var rows = np.arange(r0, r1).reshape(r1 - r0, 1);   // (h',1)
            var cols = np.arange(c0, c1).reshape(1, c1 - c0);   // (1,w')
            var dr = rows - row;
            var dc = cols - col;
            var dist2 = dr * dr + dc * dc;                      // (h',w') via broadcasting
            var mask = dist2 <= radius * radius;
            var sub = Cells[$"{r0}:{r1}, {c0}:{c1}"];
            Cells[$"{r0}:{r1}, {c0}:{c1}"] = np.where(mask, material, sub);
        }

        /// <summary>
        /// Sweeps the brush along the straight segment from <c>(r0,c0)</c> to <c>(r1,c1)</c>, stamping it at
        /// each step. This is what a mouse DRAG uses so a fast stroke paints a continuous line instead of
        /// leaving gaps between the frames the cursor was sampled at, and it is how scenes draw diagonal wall
        /// ledges. The number of stamps is the Chebyshev distance, so adjacent stamps overlap and the line is
        /// solid for any radius ≥ 0.
        /// </summary>
        /// <param name="r0">Start row.</param>
        /// <param name="c0">Start column.</param>
        /// <param name="r1">End row.</param>
        /// <param name="c1">End column.</param>
        /// <param name="radius">Brush radius in cells.</param>
        /// <param name="material">Material id to write.</param>
        /// <param name="shape">Brush footprint stamped at each point along the line.</param>
        public void PaintLine(int r0, int c0, int r1, int c1, int radius, int material, BrushShape shape = BrushShape.Disk)
        {
            int steps = Math.Max(Math.Abs(r1 - r0), Math.Abs(c1 - c0));
            if (steps == 0) { Paint(r0, c0, radius, material, shape); return; }
            // Walk the segment in `steps` equal increments and stamp the brush at each rounded sample point.
            for (int k = 0; k <= steps; k++)
            {
                double t = (double)k / steps;
                int r = (int)Math.Round(r0 + (r1 - r0) * t);
                int c = (int)Math.Round(c0 + (c1 - c0) * t);
                Paint(r, c, radius, material, shape);
            }
        }

        /// <summary>
        /// Fills the half-open rectangle rows <c>[r0,r1)</c> × cols <c>[c0,c1)</c> with one material in a
        /// single vectorized slice-assignment, clamping the rectangle to the grid (an off-grid or empty
        /// rectangle is a no-op). Unlike <see cref="Blit"/> this takes a scalar (not a block) and CLIPS
        /// instead of throwing, which is what makes it convenient for building scenes at any grid size —
        /// walls, ledges and funnels are just rectangles.
        /// </summary>
        /// <param name="r0">Top row (inclusive).</param>
        /// <param name="c0">Left column (inclusive).</param>
        /// <param name="r1">Bottom row (exclusive).</param>
        /// <param name="c1">Right column (exclusive).</param>
        /// <param name="material">Material id to write.</param>
        public void FillRect(int r0, int c0, int r1, int c1, int material)
        {
            r0 = Math.Max(0, r0); c0 = Math.Max(0, c0);
            r1 = Math.Min(Height, r1); c1 = Math.Min(Width, c1);
            if (r1 <= r0 || c1 <= c0) return;   // nothing to fill after clamping
            Cells[$"{r0}:{r1}, {c0}:{c1}"] = np.full(new Shape(r1 - r0, c1 - c0), material, NPTypeCode.Int32);
        }

        /// <summary>
        /// Stamps a rectangular block of material ids into the grid at <c>(r0, c0)</c> in ONE vectorized
        /// assignment. This is the fast way to build a scene or fill a large region — a per-cell
        /// <see cref="SetCell"/> loop over hundreds of thousands of cells costs seconds, while this is a
        /// single slice-assignment.
        /// </summary>
        /// <param name="r0">Top row where the block's row 0 lands.</param>
        /// <param name="c0">Left column where the block's column 0 lands.</param>
        /// <param name="block">An <c>[h, w]</c> array of material ids; must fit within the grid at <c>(r0, c0)</c>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="block"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The block does not fit within the grid at <c>(r0, c0)</c>.</exception>
        public void Blit(int r0, int c0, int[,] block)
        {
            if (block is null) throw new ArgumentNullException(nameof(block));
            int bh = block.GetLength(0), bw = block.GetLength(1);
            if (r0 < 0 || c0 < 0 || r0 + bh > Height || c0 + bw > Width)
                throw new ArgumentOutOfRangeException(nameof(block), "block does not fit within the grid.");
            Cells[$"{r0}:{r0 + bh}, {c0}:{c0 + bw}"] = np.array(block);
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
