using System;
using System.Collections.Generic;

namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The game-facing wrapper around the <see cref="PowderGrid"/> physics: it owns the current brush,
    /// the emitters ("faucets"), the boundary walls, and the per-frame stepping, so the Unity layer only
    /// has to translate mouse input into <see cref="Paint"/> calls and draw the grid. It adds nothing to
    /// the physics — every rule lives in <see cref="PowderGrid"/> — it only bundles the sandbox's
    /// controls in one place.
    /// </summary>
    /// <remarks>
    /// The world is Unity-independent (no <c>UnityEngine</c> types), which is what lets the verification
    /// harness exercise the exact same code the game runs.
    /// </remarks>
    public sealed class FallingSandWorld
    {
        /// <summary>The underlying cellular-automaton grid.</summary>
        public PowderGrid Grid { get; }

        /// <summary>Grid height in cells.</summary>
        public int Height => Grid.Height;

        /// <summary>Grid width in cells.</summary>
        public int Width => Grid.Width;

        /// <summary>The material the brush paints (a <see cref="Cell"/> id). <see cref="Cell.Empty"/> makes the brush an eraser.</summary>
        public int BrushMaterial { get; set; } = Cell.Sand;

        /// <summary>The brush radius in cells. Larger paints (and pours) more per stroke.</summary>
        public int BrushRadius { get; set; } = 3;

        /// <summary>The brush footprint. Disk is natural for pouring; square draws crisp walls (see <see cref="BrushShape"/>).</summary>
        public BrushShape BrushShape { get; set; } = BrushShape.Disk;

        // Persistent emitters: each drips its material into its cell every substep, so the player can set
        // up a faucet and walk away. Emitters ADD mass (by design), unlike the mass-conserving physics.
        private readonly List<(int row, int col, int material)> _emitters = new List<(int, int, int)>();

        /// <summary>Creates the world; optionally lines the left, right and bottom edges with walls so material pools instead of falling off.</summary>
        /// <param name="height">Grid height in cells.</param>
        /// <param name="width">Grid width in cells.</param>
        /// <param name="seed">Random seed for reproducible fluid flow.</param>
        /// <param name="withBoundary">When true, adds left/right/bottom wall borders (top left open to pour into).</param>
        public FallingSandWorld(int height, int width, int seed = 1, bool withBoundary = true)
        {
            Grid = new PowderGrid(height, width, seed);
            if (withBoundary) AddBoundaryWalls();
        }

        /// <summary>Lines the left, right and bottom edges with a 2-cell-thick wall so material stays in the basin.</summary>
        public void AddBoundaryWalls()
        {
            int t = 2;   // thickness so a fast grain can't tunnel through in one step
            for (int r = 0; r < Height; r++)
                for (int k = 0; k < t; k++) { Grid.SetCell(r, k, Cell.Wall); Grid.SetCell(r, Width - 1 - k, Cell.Wall); }
            for (int c = 0; c < Width; c++)
                for (int k = 0; k < t; k++) Grid.SetCell(Height - 1 - k, c, Cell.Wall);
        }

        /// <summary>Advances the simulation by <paramref name="subSteps"/> physics steps, running the emitters before each. More substeps per frame settle material faster.</summary>
        /// <param name="subSteps">Number of <see cref="PowderGrid.Step"/> calls to run (clamped to ≥ 0).</param>
        public void Step(int subSteps)
        {
            for (int i = 0; i < subSteps; i++)
            {
                RunEmitters();
                Grid.Step();
            }
        }

        /// <summary>Paints the current brush (material, radius and shape) at a grid cell (the game maps the mouse to a cell and calls this).</summary>
        /// <param name="row">Grid row (0 = top).</param>
        /// <param name="col">Grid column (0 = left).</param>
        public void Paint(int row, int col) => Grid.Paint(row, col, BrushRadius, BrushMaterial, BrushShape);

        /// <summary>
        /// Paints a continuous stroke of the current brush from one cell to another — what a mouse DRAG uses,
        /// so a quick swipe leaves an unbroken line rather than dots at the two frames the cursor happened to
        /// be sampled at.
        /// </summary>
        /// <param name="fromRow">Stroke start row.</param>
        /// <param name="fromCol">Stroke start column.</param>
        /// <param name="toRow">Stroke end row.</param>
        /// <param name="toCol">Stroke end column.</param>
        public void PaintStroke(int fromRow, int fromCol, int toRow, int toCol) =>
            Grid.PaintLine(fromRow, fromCol, toRow, toCol, BrushRadius, BrushMaterial, BrushShape);

        /// <summary>
        /// Replaces the whole scene with a presaved <see cref="SandScenes"/> template (a waterfall, hourglass,
        /// volcano, …). This wipes the current world — loose material, walls AND emitters — and rebuilds it,
        /// scaled to this world's size, so the same template looks right at an icon's resolution and at the
        /// full game grid.
        /// </summary>
        /// <param name="index">Template index in <c>0 .. <see cref="TemplateCount"/>-1</c>; out-of-range is clamped.</param>
        public void LoadTemplate(int index) => SandScenes.Apply(index, this);

        /// <summary>The display names of the available templates, index-aligned with <see cref="LoadTemplate"/>.</summary>
        public static string[] TemplateNames => SandScenes.Names;

        /// <summary>How many templates <see cref="LoadTemplate"/> accepts.</summary>
        public static int TemplateCount => SandScenes.Count;

        /// <summary>Places a persistent emitter that drips <paramref name="material"/> at the given cell every step.</summary>
        /// <param name="row">Emitter row.</param>
        /// <param name="col">Emitter column.</param>
        /// <param name="material">Material to emit (usually sand or water).</param>
        public void AddEmitter(int row, int col, int material) => _emitters.Add((row, col, material));

        /// <summary>Removes all emitters.</summary>
        public void ClearEmitters() => _emitters.Clear();

        /// <summary>The number of active emitters.</summary>
        public int EmitterCount => _emitters.Count;

        /// <summary>Empties the basin.</summary>
        /// <param name="keepWalls">When true, keeps the boundary walls and clears only loose material.</param>
        public void Clear(bool keepWalls) => Grid.Clear(keepWalls);

        /// <summary>Counts the cells of a material (for the HUD).</summary>
        /// <param name="material">Material id.</param>
        /// <returns>The current cell count of that material.</returns>
        public int Count(int material) => Grid.Count(material);

        /// <summary>Drips each emitter's material into its cell, but only when the cell is currently empty (so it forms a stream rather than a solid plug).</summary>
        private void RunEmitters()
        {
            foreach (var e in _emitters)
                if (Grid.GetCell(e.row, e.col) == Cell.Empty)
                    Grid.SetCell(e.row, e.col, e.material);
        }
    }
}
