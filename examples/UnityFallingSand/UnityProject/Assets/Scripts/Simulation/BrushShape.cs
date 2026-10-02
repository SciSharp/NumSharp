namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The footprint the brush stamps into the grid. This is a paint-time choice only — it changes which
    /// cells a single <see cref="PowderGrid.Paint(int,int,int,int,BrushShape)"/> touches, never the physics
    /// of what happens next.
    /// </summary>
    /// <remarks>
    /// A disk feels natural for pouring loose material (round piles), while a square is what you want for
    /// drawing straight-edged walls (basins, funnels, ledges) without the ragged rounded corners a disk
    /// leaves — which is exactly why paintable walls get their own shape rather than forcing every stroke
    /// through one round brush.
    /// </remarks>
    public enum BrushShape
    {
        /// <summary>A filled circle of the brush radius — the default; good for pouring sand/water and round strokes.</summary>
        Disk = 0,

        /// <summary>A filled axis-aligned square of side <c>2·radius+1</c> — good for drawing crisp walls and rectangular structures.</summary>
        Square = 1,
    }
}
