namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The material vocabulary of the falling-sand world and the per-material physical constants that
    /// drive its behaviour. Every cell of the grid holds one of these ids, and ALL of the emergent
    /// physics — what sinks, what floats, what rises, what piles, what levels — falls out of two tables
    /// here: <see cref="Density(int)"/> and <see cref="IsFluid(int)"/>. There are no per-material special
    /// cases in the engine; change a density and the stratification order changes with it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Density is the master rule.</b> The engine moves a cell down whenever the cell below is
    /// strictly LESS dense (Archimedes: heavier sinks, lighter floats). Ordering the densities is
    /// therefore ordering the layers a mixture settles into. The chosen order
    /// <c>Smoke(-1) &lt; Empty(0) &lt; Oil(1) &lt; Water(2) &lt; Sand(3)</c> makes smoke rise through
    /// everything (it is lighter than air), oil float on water, and sand sink to the bottom — all from
    /// the one comparison.
    /// </para>
    /// <para>
    /// <b>Smoke's density is NEGATIVE on purpose.</b> Treating empty space as a fluid of density 0 means
    /// "smoke rises" needs no separate rule: the empty cell above smoke is DENSER than the smoke, so the
    /// ordinary downward density swap moves the empty down and the smoke up. One rule, two directions.
    /// </para>
    /// <para>
    /// <b>Walls never move.</b> <see cref="Wall"/> is given a huge density so it would "want" to sink, but
    /// the engine excludes walls from every swap, so it is immovable regardless. The large number only
    /// ensures a wall is never mistaken for something a particle can displace.
    /// </para>
    /// </remarks>
    public static class Cell
    {
        /// <summary>Vacuum / air. The default cell; acts as a density-0 fluid so gases can rise into it.</summary>
        public const int Empty = 0;
        /// <summary>Immovable solid boundary. Excluded from all swaps; particles pile and pool against it.</summary>
        public const int Wall = 1;
        /// <summary>Granular solid. Falls, and slides diagonally off obstacles to form ~45° piles; does NOT spread flat.</summary>
        public const int Sand = 2;
        /// <summary>A liquid, denser than oil. Falls, sinks below oil, and spreads to a flat surface.</summary>
        public const int Water = 3;
        /// <summary>A lighter liquid. Falls through air but floats up through water; spreads flat.</summary>
        public const int Oil = 4;
        /// <summary>A gas lighter than air. Rises and spreads under ceilings.</summary>
        public const int Smoke = 5;

        /// <summary>The number of distinct material ids (ids are <c>0 .. Count-1</c>).</summary>
        public const int Count = 6;

        /// <summary>The density assigned to <see cref="Wall"/> — large enough that nothing displaces it (and it is excluded from swaps anyway).</summary>
        public const int WallDensity = 1000;

        // Density per id, index-aligned with the constants above. Negative = lighter than air (rises).
        private static readonly int[] _density = { 0, WallDensity, 3, 2, 1, -1 };

        // Whether a material spreads horizontally to level out (liquids and gases) rather than piling (granular/solid).
        private static readonly bool[] _isFluid = { false, false, false, true, true, true };

        /// <summary>Human-readable material names, index-aligned with the ids (for HUD/logging).</summary>
        public static readonly string[] Name = { "Empty", "Wall", "Sand", "Water", "Oil", "Smoke" };

        /// <summary>
        /// Display colours as RGB in 0..1, index-aligned with the ids. The view maps these to pixels; they
        /// carry no physics. Empty is drawn as the near-black background.
        /// </summary>
        public static readonly (double r, double g, double b)[] Color =
        {
            (0.05, 0.05, 0.08),   // Empty  — background
            (0.40, 0.40, 0.45),   // Wall   — grey
            (0.85, 0.72, 0.38),   // Sand   — tan
            (0.20, 0.45, 0.95),   // Water  — blue
            (0.55, 0.35, 0.12),   // Oil    — dark amber
            (0.75, 0.75, 0.80),   // Smoke  — light grey
        };

        /// <summary>The material's density (see the class remarks — this table IS the physics).</summary>
        /// <param name="id">A material id in <c>0 .. Count-1</c>.</param>
        /// <returns>The density; higher sinks, negative rises, <see cref="WallDensity"/> for walls.</returns>
        public static int Density(int id) => _density[id];

        /// <summary>Whether the material levels horizontally (a liquid or gas) as opposed to piling (sand) or staying put (wall).</summary>
        /// <param name="id">A material id in <c>0 .. Count-1</c>.</param>
        /// <returns>True for water/oil/smoke; false for empty/wall/sand.</returns>
        public static bool IsFluid(int id) => _isFluid[id];

        /// <summary>The full density table (index = id), exposed so the engine can build a whole-grid density field without hard-coding materials.</summary>
        public static int[] DensityTable => _density;
    }
}
