using System;

namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The presaved "pretty designs" the game offers behind its template icons — a waterfall, an hourglass,
    /// a volcano, a fountain, a rain tank, and a blank sandbox. Each one wipes the world and rebuilds it from
    /// walls, initial material and emitters, so selecting a template drops you straight into an animated scene
    /// ("water flowing down and such") instead of a blank grid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Everything here is size-RELATIVE.</b> A scene is expressed as fractions of the world's height and
    /// width, never in absolute cells, so the very same template builds a recognizable thumbnail at an icon's
    /// 64×40 and the full show at the game's 1000×500. That is what lets the HUD generate its icons by simply
    /// building each template into a tiny world and rendering it.
    /// </para>
    /// <para>
    /// <b>It is Unity-independent</b> (no <c>UnityEngine</c> types), built only from the public
    /// <see cref="PowderGrid"/>/<see cref="FallingSandWorld"/> surface — <see cref="PowderGrid.FillRect"/>,
    /// <see cref="FallingSandWorld.AddEmitter"/> and the boundary walls — so the verification harness can
    /// load and step every template exactly as the game does.
    /// </para>
    /// <para>
    /// <b>Walls are the load-bearing element.</b> Every scene's shape is walls the water/sand/lava then
    /// interact with — ledges to cascade off, funnels to pour through, a mountain to run down, a basin to pool
    /// in — which is the same paintable <see cref="Cell.Wall"/> the player draws by hand, just placed for them.
    /// </para>
    /// </remarks>
    public static class SandScenes
    {
        /// <summary>Template display names, index-aligned with <see cref="Apply"/>. Order is the HUD icon order.</summary>
        public static readonly string[] Names = { "Waterfall", "Hourglass", "Volcano", "Fountain", "Rain", "Sandbox" };

        /// <summary>The number of templates (valid indices are <c>0 .. Count-1</c>).</summary>
        public static int Count => Names.Length;

        /// <summary>
        /// Rebuilds <paramref name="world"/> as the chosen template: it first wipes everything (loose material,
        /// walls and emitters) and re-adds the boundary basin, then lays out that scene's walls, material and
        /// faucets. After this returns, stepping the world animates the design.
        /// </summary>
        /// <param name="index">Template index; values outside <c>0 .. <see cref="Count"/>-1</c> are clamped into range.</param>
        /// <param name="world">The world to overwrite. Its size is read to scale the scene; it is not otherwise required to be empty.</param>
        /// <exception cref="ArgumentNullException"><paramref name="world"/> is null.</exception>
        public static void Apply(int index, FallingSandWorld world)
        {
            if (world is null) throw new ArgumentNullException(nameof(world));
            if (index < 0) index = 0;
            if (index >= Count) index = Count - 1;

            // Start from a clean, walled basin for every template so they compose the same way at any size.
            world.ClearEmitters();
            world.Clear(keepWalls: false);
            world.AddBoundaryWalls();

            switch (index)
            {
                case 0: BuildWaterfall(world); break;
                case 1: BuildHourglass(world); break;
                case 2: BuildVolcano(world); break;
                case 3: BuildFountain(world); break;
                case 4: BuildRain(world); break;
                default: /* Sandbox: the walled basin is the whole scene */ break;
            }
        }

        // The boundary walls AddBoundaryWalls lays down are 2 cells thick; scenes inset by this margin.
        private const int Margin = 2;

        /// <summary>
        /// A cascade of SHORT wall bars laid out in an offset (brick) pattern, fed by faucets at the top
        /// centre. Water pours onto a bar, spills off its near edges into the gaps, and falls to the offset
        /// bars on the row below — trickling down the whole field like rapids. The bars are deliberately short:
        /// horizontal fluid spread is a slow random walk, so a wide ledge just puddles, whereas a short bar
        /// spills at once. This is the "water flowing down" design.
        /// </summary>
        /// <param name="world">The world to build into (already wiped and walled).</param>
        private static void BuildWaterfall(FallingSandWorld world)
        {
            int h = world.Height, w = world.Width;
            int t = Math.Max(2, h / 60);            // bar thickness
            int period = Math.Max(6, w / 6);        // horizontal spacing between bars
            int barLen = Math.Max(3, period * 2 / 3);   // each bar is shorter than its slot, leaving spill gaps
            const int rows = 5;
            for (int i = 0; i < rows; i++)
            {
                int y = (int)((i + 1) * (double)h / (rows + 1));   // evenly spaced down the basin
                int off = (i % 2 == 0) ? 0 : period / 2;           // offset every other row so gaps don't line up
                for (int c = Margin + off; c < w - Margin; c += period)
                    world.Grid.FillRect(y, c, y + t, Math.Min(w - Margin, c + barLen), Cell.Wall);
            }
            // Faucets clustered at the top centre so the cascade fans out symmetrically down the field.
            int cx = w / 2;
            world.AddEmitter(1, cx, Cell.Water);
            world.AddEmitter(1, Math.Max(Margin, cx - period / 3), Cell.Water);
            world.AddEmitter(1, Math.Min(w - Margin - 1, cx + period / 3), Cell.Water);
        }

        /// <summary>
        /// An hourglass: two wall funnels pinching to a narrow neck at mid-height, with the top chamber filled
        /// with sand. The sand streams through the neck and rebuilds a cone in the lower chamber — the classic
        /// granular-flow demo, and an unmistakable X-shaped icon.
        /// </summary>
        /// <param name="world">The world to build into (already wiped and walled).</param>
        private static void BuildHourglass(FallingSandWorld world)
        {
            int h = world.Height, w = world.Width;
            int cx = w / 2, midR = h / 2;
            int wideHalf = w / 2 - Margin;                 // opening half-width at the chambers' widest
            int neckHalf = Math.Max(2, w / 16);            // opening half-width at the neck
            int topStart = Math.Max(Margin, h / 8);        // leave some open space above the top chamber
            int botEnd = h - Margin;

            // Top funnel: opening shrinks from wide (at topStart) down to the neck (at midR).
            for (int r = topStart; r < midR; r++)
            {
                double f = (double)(r - topStart) / Math.Max(1, midR - topStart);
                int openHalf = (int)(wideHalf - (wideHalf - neckHalf) * f);
                world.Grid.FillRect(r, Margin, r + 1, cx - openHalf, Cell.Wall);
                world.Grid.FillRect(r, cx + openHalf, r + 1, w - Margin, Cell.Wall);
            }
            // Bottom funnel: opening grows back from the neck (at midR) to wide (at the floor).
            for (int r = midR; r < botEnd; r++)
            {
                double f = (double)(r - midR) / Math.Max(1, botEnd - midR);
                int openHalf = (int)(neckHalf + (wideHalf - neckHalf) * f);
                world.Grid.FillRect(r, Margin, r + 1, cx - openHalf, Cell.Wall);
                world.Grid.FillRect(r, cx + openHalf, r + 1, w - Margin, Cell.Wall);
            }
            // Fill the top chamber's opening with sand (leaving a thin gap under the top so it can start moving).
            for (int r = topStart + Math.Max(1, h / 40); r < midR - 1; r++)
            {
                double f = (double)(r - topStart) / Math.Max(1, midR - topStart);
                int openHalf = (int)(wideHalf - (wideHalf - neckHalf) * f);
                world.Grid.FillRect(r, cx - openHalf + 1, r + 1, cx + openHalf, Cell.Sand);
            }
        }

        /// <summary>
        /// A volcano: a wall mountain with a hollow central vent that a lava faucet erupts into, and water
        /// moats banked at its feet. Lava overflows the vent, runs down the mountain's stepped slopes, and
        /// where it reaches the moats it freezes to obsidian (wall) while the water flashes to steam — the
        /// most dramatic scene, and the one that shows off the fire/lava reactions.
        /// </summary>
        /// <param name="world">The world to build into (already wiped and walled).</param>
        private static void BuildVolcano(FallingSandWorld world)
        {
            int h = world.Height, w = world.Width;
            int cx = w / 2, baseR = h - Margin, apexR = Math.Max(Margin + 2, h / 5);
            int baseHalf = (int)((w / 2 - Margin) * 0.82);   // mountain half-width at the base
            int apexHalf = Math.Max(4, w / 12);              // half-width at the summit
            int vent = Math.Max(2, w / 28);                  // radius of the hollow vent

            // Solid mountain: a triangle widening from summit to base.
            for (int r = apexR; r < baseR; r++)
            {
                double f = (double)(r - apexR) / Math.Max(1, baseR - apexR);
                int halfW = (int)(apexHalf + (baseHalf - apexHalf) * f);
                world.Grid.FillRect(r, cx - halfW, r + 1, cx + halfW, Cell.Wall);
            }
            // Carve a vent through the upper mountain and pre-fill it with a glowing lava column (dense lava
            // would otherwise take a long time to trickle down a deep empty tube, leaving it dark). The summit
            // faucet keeps the column topped up so it stays lit.
            int ventBot = apexR + (int)((baseR - apexR) * 0.55);
            world.Grid.FillRect(apexR, cx - vent, ventBot, cx + vent, Cell.Empty);      // hollow the tube
            world.Grid.FillRect(apexR + 1, cx - vent, ventBot, cx + vent, Cell.Lava);   // molten column
            world.AddEmitter(apexR, cx, Cell.Lava);                                     // keep it lit
            // Water moats in the gaps the mountain leaves at each foot.
            int poolTop = baseR - Math.Max(4, h / 8);
            world.Grid.FillRect(poolTop, Margin, baseR, Math.Max(Margin + 3, cx - baseHalf - 2), Cell.Water);
            world.Grid.FillRect(poolTop, Math.Min(w - Margin - 3, cx + baseHalf + 2), baseR, w - Margin, Cell.Water);
        }

        /// <summary>
        /// A fountain: a walled basin in the lower middle already filled with water under a floating layer of
        /// oil, topped up by a faucet dripping from above. It reads as a still pool and makes the oil-floats-on-
        /// water stratification obvious at a glance.
        /// </summary>
        /// <param name="world">The world to build into (already wiped and walled).</param>
        private static void BuildFountain(FallingSandWorld world)
        {
            int h = world.Height, w = world.Width;
            int cx = w / 2;
            int bTop = (int)(h * 0.45), bBot = h - Margin;
            int bl = (int)(w * 0.20), br = (int)(w * 0.80);
            int tw = Math.Max(2, w / 40);                    // basin wall thickness

            world.Grid.FillRect(bTop, bl, bBot, bl + tw, Cell.Wall);        // basin left wall
            world.Grid.FillRect(bTop, br - tw, bBot, br, Cell.Wall);        // basin right wall
            world.Grid.FillRect(bBot - tw, bl, bBot, br, Cell.Wall);        // basin floor (above the outer bottom)

            int inL = bl + tw, inR = br - tw, inBot = bBot - tw, inTop = bTop + 1;
            int waterTop = (int)(inTop + (inBot - inTop) * 0.35);
            world.Grid.FillRect(waterTop, inL, inBot, inR, Cell.Water);                       // water fills most of the basin
            world.Grid.FillRect(waterTop - Math.Max(2, h / 20), inL, waterTop, inR, Cell.Oil); // an oil layer on top
            world.AddEmitter(1, cx, Cell.Water);                                             // a lively drip from above
        }

        /// <summary>
        /// Rain: a row of faucets across the top of the open tank, cycling sand, water and oil, so the basin
        /// fills with a falling mix that settles into clean density layers — the density-stratification
        /// signature of the whole engine, running by itself.
        /// </summary>
        /// <param name="world">The world to build into (already wiped and walled).</param>
        private static void BuildRain(FallingSandWorld world)
        {
            int w = world.Width;
            int n = Math.Max(4, w / 12);                     // number of faucets across the width
            int[] cycle = { Cell.Sand, Cell.Water, Cell.Oil };
            for (int k = 0; k < n; k++)
            {
                int c = Margin + (int)((k + 0.5) * (w - 2 * Margin) / n);
                world.AddEmitter(1, c, cycle[k % cycle.Length]);
            }
        }
    }
}
