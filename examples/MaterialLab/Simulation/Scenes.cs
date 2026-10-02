using System;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>
    /// Ready-made experiments — each wipes the world and builds a set piece from obstacles, filled shapes and
    /// faucets, in normalized units (the tank is 1 unit tall, <see cref="MpmGrid.Width"/> wide), so every scene
    /// works at any grid resolution.
    /// </summary>
    public static class Scenes
    {
        /// <summary>A scene: display name, one-line description, and its builder.</summary>
        /// <param name="Name">Display name.</param>
        /// <param name="Description">What to watch for.</param>
        /// <param name="Build">Builds the scene into a freshly reset world.</param>
        public readonly record struct Scene(string Name, string Description, Action<MpmWorld> Build);

        /// <summary>All scenes, in F-key order (F1 = index 0).</summary>
        public static readonly Scene[] All =
        {
            new Scene("Dam Break", "A water column collapses around a stone pillar — surge, splash, slosh.", DamBreak),
            new Scene("Waterfall", "Water cascades down staggered ledges into a pool.", Waterfall),
            new Scene("Sandcastle vs Wave", "A wave hits a sand castle: granular collapse and erosion.", Sandcastle),
            new Scene("Jelly & Rubber", "Wobbly jelly and bouncing rubber drop onto clay and ramps.", JellyRubber),
            new Scene("Snowballs", "Snowballs smash into a wall and a snow bank — packing and fracture.", Snowballs),
            new Scene("Oil, Water & Honey", "Buoyancy: oil rises through water, honey sinks and coils.", OilWater),
            new Scene("Lava Flow", "Lava pours down rocky steps into a lake and freezes into rock.", LavaFlow),
            new Scene("Hourglass", "Sand drains through a funnel into a growing cone.", Hourglass),
            new Scene("Material Zoo", "One blob of every material dropped side by side.", Zoo),
            new Scene("Sandbox", "An empty tank — paint your own experiment.", w => { }),
        };

        /// <summary>Resets the world and builds scene <paramref name="index"/>.</summary>
        /// <param name="world">The world.</param>
        /// <param name="index">Scene index (clamped).</param>
        /// <returns>The scene that was built.</returns>
        public static Scene Load(MpmWorld world, int index)
        {
            index = Math.Clamp(index, 0, All.Length - 1);
            world.Reset();
            All[index].Build(world);
            return All[index];
        }

        /// <summary>Height where fills may start: one cell above the floor's surface (filled points need φ ≥ 1).</summary>
        /// <param name="w">World.</param>
        /// <returns>The height in units.</returns>
        private static float Floor(MpmWorld w) => (MpmGrid.Border + 0.5f) * w.Grid.Dx;
        /// <summary>The matching inner limit at the right wall.</summary>
        /// <param name="w">World.</param>
        /// <returns>The x in units.</returns>
        private static float Right(MpmWorld w) => w.Grid.Width - (MpmGrid.Border + 0.5f) * w.Grid.Dx;

        /// <summary>A tall water column at the left collapses; a pillar and a ramp break up the surge.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void DamBreak(MpmWorld w)
        {
            float f = Floor(w);
            w.Grid.PaintRect(1.02f, 0f, 1.08f, 0.22f);                    // stone pillar
            w.Grid.PaintSegment(1.35f, f, 1.62f, 0.14f, 0.012f);           // a ramp to throw the surge up
            w.FillRect(MaterialId.Water, f, f, 0.52f, 0.74f);
        }

        /// <summary>A faucet at the top left runs down three sloped ledges into a walled pool.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void Waterfall(MpmWorld w)
        {
            float f = Floor(w), right = Right(w);
            w.Grid.PaintSegment(0.02f, 0.78f, 0.62f, 0.72f, 0.014f);       // top ledge, gently sloped
            w.Grid.PaintSegment(0.45f, 0.52f, 1.05f, 0.47f, 0.014f);       // middle ledge
            w.Grid.PaintSegment(0.95f, 0.28f, 1.45f, 0.24f, 0.014f);       // lower ledge
            w.Grid.PaintRect(1.52f, 0f, 1.56f, 0.16f);                      // pool wall
            w.FillRect(MaterialId.Water, 1.56f, f, right, 0.12f);
            w.Emitters.Add(new Emitter { Material = MaterialId.Water, X = 0.08f, Y = 0.86f, Radius = 0.035f, VX = 3.5f, VY = -1.0f });
        }

        /// <summary>A keep, towers and battlements of sand, and a wave already moving toward them.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void Sandcastle(MpmWorld w)
        {
            float f = Floor(w);
            w.FillRect(MaterialId.Sand, 1.05f, f, 1.62f, 0.18f);           // keep
            w.FillRect(MaterialId.Sand, 1.10f, 0.18f, 1.22f, 0.36f);       // left tower
            w.FillRect(MaterialId.Sand, 1.45f, 0.18f, 1.57f, 0.36f);       // right tower
            w.FillRect(MaterialId.Sand, 1.26f, 0.18f, 1.41f, 0.27f);       // gate block
            for (int i = 0; i < 3; i++)
            {
                w.FillRect(MaterialId.Sand, 1.10f + i * 0.045f, 0.36f, 1.13f + i * 0.045f, 0.40f);   // battlements
                w.FillRect(MaterialId.Sand, 1.45f + i * 0.045f, 0.36f, 1.48f + i * 0.045f, 0.40f);
            }
            w.FillRect(MaterialId.Water, f, f, 0.40f, 0.66f, vx: 1.5f);   // the wave, already moving
        }

        /// <summary>Jelly blocks and thrown rubber balls land on ramps and clay pads (elastic vs plastic side by side).</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void JellyRubber(MpmWorld w)
        {
            float f = Floor(w);
            w.Grid.PaintSegment(0.12f, 0.46f, 0.62f, 0.30f, 0.014f);
            w.Grid.PaintSegment(1.72f, 0.52f, 1.22f, 0.36f, 0.014f);
            w.FillRect(MaterialId.Clay, 0.78f, f, 1.02f, 0.10f);
            w.FillRect(MaterialId.Clay, 1.05f, f, 1.14f, 0.16f);
            w.FillRect(MaterialId.Jelly, 0.20f, 0.62f, 0.36f, 0.78f);
            w.FillRect(MaterialId.Jelly, 1.45f, 0.66f, 1.58f, 0.80f);
            w.FillRect(MaterialId.Jelly, 0.86f, 0.55f, 0.98f, 0.67f);
            w.FillDisk(MaterialId.Rubber, 0.90f, 0.86f, 0.045f, vx: 0.8f);
            w.FillDisk(MaterialId.Rubber, 0.62f, 0.90f, 0.035f, vx: 2.0f);
            w.FillDisk(MaterialId.Rubber, 1.30f, 0.88f, 0.04f, vx: -1.5f);
        }

        /// <summary>Fast snowballs hit a wall, a snow ground and a snow bank — compaction and fracture.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void Snowballs(MpmWorld w)
        {
            float f = Floor(w), right = Right(w);
            w.Grid.PaintRect(1.40f, 0f, 1.46f, 0.55f);                      // a wall to hit
            w.FillRect(MaterialId.Snow, f, f, 1.40f, 0.07f);               // snow ground
            w.FillRect(MaterialId.Snow, 1.46f, f, right, 0.22f);           // snow bank behind the wall
            w.FillDisk(MaterialId.Snow, 0.25f, 0.55f, 0.065f, vx: 14f, vy: 3f);
            w.FillDisk(MaterialId.Snow, 0.45f, 0.30f, 0.055f, vx: 16f, vy: 5f);
            w.FillDisk(MaterialId.Snow, 0.12f, 0.80f, 0.05f, vx: 12f, vy: 0f);
            w.FillDisk(MaterialId.Snow, 1.60f, 0.62f, 0.06f, vx: -6f, vy: 2f);
        }

        /// <summary>Oil blobs trapped under a water tank rise as plumes; a honey slab plunges in and sinks.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void OilWater(MpmWorld w)
        {
            float f = Floor(w), right = Right(w);
            // Enclosed material first: fills never over-pack a cell, so the water then flows AROUND the oil.
            w.FillDisk(MaterialId.Oil, 0.45f, 0.17f, 0.12f);               // oil trapped at the bottom rises
            w.FillDisk(MaterialId.Oil, 1.30f, 0.14f, 0.10f);
            w.FillRect(MaterialId.Water, f, f, right, 0.46f);
            w.FillRect(MaterialId.Honey, 0.95f, 0.70f, 1.05f, 0.92f);      // a slab of honey falls in and sinks
        }

        /// <summary>A lava faucet runs down three rocky steps into a rimmed lake, crusting in air and freezing in water.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void LavaFlow(MpmWorld w)
        {
            float f = Floor(w), right = Right(w);
            w.Grid.PaintSegment(0.02f, 0.70f, 0.40f, 0.62f, 0.02f);
            w.Grid.PaintSegment(0.30f, 0.52f, 0.80f, 0.40f, 0.02f);
            w.Grid.PaintSegment(0.70f, 0.30f, 1.08f, 0.22f, 0.02f);
            w.Grid.PaintRect(1.08f, 0f, 1.12f, 0.20f);                      // the lake's rim, just under the last step
            w.FillRect(MaterialId.Water, 1.12f, f, right, 0.17f);          // the lake
            // About a third of the lake's volume in lava: enough to build a rock shelf out into it, not to fill it.
            w.Emitters.Add(new Emitter { Material = MaterialId.Lava, X = 0.07f, Y = 0.80f, Radius = 0.03f, VX = 1.5f, VY = -0.5f, Budget = 4200 });
        }

        /// <summary>A funnel of sand drains through a neck several grains wide onto the floor.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void Hourglass(MpmWorld w)
        {
            float cx = 0.5f * w.Grid.Width;
            // The neck must be several grains wide or the sand arches and jams — exactly as in a real hourglass.
            w.Grid.PaintSegment(cx - 0.42f, 0.95f, cx - 0.046f, 0.50f, 0.014f);
            w.Grid.PaintSegment(cx + 0.42f, 0.95f, cx + 0.046f, 0.50f, 0.014f);
            // Fill the bowl: a triangle, row by row, matching the funnel walls.
            for (float y = 0.56f; y < 0.92f; y += 0.01f)
            {
                float half = (y - 0.50f) / 0.45f * 0.40f - 0.03f;
                if (half > 0.01f) w.FillRect(MaterialId.Sand, cx - half, y, cx + half, y + 0.01f);
            }
        }

        /// <summary>A shallow water bed with one block of every placeable material dropped above it.</summary>
        /// <param name="w">The freshly reset world to build into.</param>
        private static void Zoo(MpmWorld w)
        {
            float f = Floor(w), right = Right(w);
            w.FillRect(MaterialId.Water, f, f, right, 0.10f);
            MaterialId[] order = { MaterialId.Water, MaterialId.Oil, MaterialId.Honey, MaterialId.Lava, MaterialId.Sand,
                                   MaterialId.Snow, MaterialId.Jelly, MaterialId.Rubber, MaterialId.Clay };
            float span = right - f, step = span / order.Length;
            for (int i = 0; i < order.Length; i++)
            {
                float cx = f + step * (i + 0.5f);
                w.FillRect(order[i], cx - step * 0.32f, 0.45f, cx + step * 0.32f, 0.45f + step * 0.64f);
            }
        }
    }
}
