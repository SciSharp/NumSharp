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
    /// <para>
    /// <b>Fire and Lava add reactions on top of the density rule.</b> They still MOVE purely by density
    /// (<see cref="Fire"/> is lighter than air so it rises like smoke; <see cref="Lava"/> is heavier than
    /// sand so it sinks to the bottom), so the movement engine needs no new cases — they drop straight out
    /// of these tables. What makes them "fun" is a separate transformation pass (<see cref="PowderGrid.Step"/>
    /// → the reaction step): fire ignites oil and burns out to smoke; lava quenches against water into
    /// obsidian (a <see cref="Wall"/>) and flashes the water to steam (<see cref="Smoke"/>). Those
    /// transformations deliberately do NOT conserve mass — like emitters, they are a chemistry layer sitting
    /// on top of the mass-conserving physics, and they only ever fire when fire or lava is actually present,
    /// so a world without them is byte-for-byte the old behaviour.
    /// </para>
    /// </remarks>
    public static class Cell
    {
        /// <summary>Vacuum / air. The default cell; acts as a density-0 fluid so gases can rise into it.</summary>
        public const int Empty = 0;
        /// <summary>Immovable solid boundary. Excluded from all swaps; particles pile and pool against it. Also what lava becomes when it quenches (obsidian).</summary>
        public const int Wall = 1;
        /// <summary>Granular solid. Falls, and slides diagonally off obstacles to form ~45° piles; does NOT spread flat.</summary>
        public const int Sand = 2;
        /// <summary>A liquid, denser than oil. Falls, sinks below oil, and spreads to a flat surface. Flashes to steam (smoke) where it touches lava.</summary>
        public const int Water = 3;
        /// <summary>A lighter liquid. Falls through air but floats up through water; spreads flat. Flammable — ignites where it touches fire or lava.</summary>
        public const int Oil = 4;
        /// <summary>A gas lighter than air. Rises and spreads under ceilings. Also the product of burning fuel and of quenched water (steam).</summary>
        public const int Smoke = 5;
        /// <summary>A gas lighter than smoke (rises fastest). Ignites adjacent oil, then burns out to smoke after a short, random lifetime.</summary>
        public const int Fire = 6;
        /// <summary>A liquid heavier than sand (sinks to the bottom). Ignites adjacent oil and, against water, freezes into obsidian (a wall) while the water steams off.</summary>
        public const int Lava = 7;

        /// <summary>The number of distinct material ids (ids are <c>0 .. Count-1</c>).</summary>
        public const int Count = 8;

        /// <summary>The density assigned to <see cref="Wall"/> — large enough that nothing displaces it (and it is excluded from swaps anyway).</summary>
        public const int WallDensity = 1000;

        // Density per id, index-aligned with the constants above. Negative = lighter than air (rises).
        // Fire(-2) is lighter than smoke(-1) so it rises fastest; lava(4) is heavier than sand(3) so it sinks below everything.
        private static readonly int[] _density = { 0, WallDensity, 3, 2, 1, -1, -2, 4 };

        // Whether a material spreads horizontally to level out (liquids and gases) rather than piling (granular/solid).
        // Fire and lava are fluids (they spread and seek a surface); their reactions are handled separately.
        private static readonly bool[] _isFluid = { false, false, false, true, true, true, true, true };

        /// <summary>Human-readable material names, index-aligned with the ids (for HUD/logging).</summary>
        public static readonly string[] Name = { "Empty", "Wall", "Sand", "Water", "Oil", "Smoke", "Fire", "Lava" };

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
            (1.00, 0.70, 0.15),   // Fire   — bright orange
            (0.90, 0.25, 0.05),   // Lava   — molten red
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
