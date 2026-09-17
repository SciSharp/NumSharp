using System;
using NumSharp;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand.Verification
{
    /// <summary>
    /// Console gate for the falling-sand physics. It compiles the same simulation source Unity runs and
    /// asserts the cellular automaton's invariants — most importantly MASS CONSERVATION, which a naive
    /// vectorized sand update silently violates by duplicating or dropping cells. A non-zero exit code
    /// means a physics regression.
    /// </summary>
    public static class Program
    {
        private static int _failures;

        /// <summary>Runs every check; returns the failure count as the process exit code.</summary>
        /// <param name="args">Unused.</param>
        /// <returns>0 if all checks pass, else the number of failures.</returns>
        public static int Main(string[] args)
        {
            Console.WriteLine("NumSharp Falling-Sand — physics verification\n");

            CheckMassConservation();
            CheckSandSettles();
            CheckStratification();
            CheckWaterLevels();
            CheckPaint();
            CheckEmitter();
            CheckDeterminism();
            CheckBoundaryContains();

            Console.WriteLine();
            if (_failures == 0) Console.WriteLine("ALL CHECKS PASSED.");
            else Console.WriteLine($"{_failures} CHECK(S) FAILED.");
            return _failures;
        }

        private static void Check(bool ok, string msg)
        {
            Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {msg}");
            if (!ok) _failures++;
        }

        /// <summary>Mean row index of a material from a flat C-order snapshot (smaller = higher on screen).</summary>
        private static double MeanRow(int[] snap, int h, int w, int mat)
        {
            long sum = 0; int cnt = 0;
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++)
                    if (snap[r * w + c] == mat) { sum += r; cnt++; }
            return cnt == 0 ? double.NaN : (double)sum / cnt;
        }

        /// <summary>
        /// The defining correctness property: every material's cell count must be invariant under stepping
        /// (with no emitters), because each rule is a pure swap. We fill a busy random grid and run it hard.
        /// </summary>
        private static void CheckMassConservation()
        {
            Console.WriteLine("Mass conservation:");
            int h = 48, w = 48;
            var world = new FallingSandWorld(h, w, seed: 7, withBoundary: false);
            var rng = new Random(7);
            int[] palette = { Cell.Empty, Cell.Empty, Cell.Sand, Cell.Water, Cell.Oil, Cell.Smoke, Cell.Wall };
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++)
                    world.Grid.SetCell(r, c, palette[rng.Next(palette.Length)]);

            int[] mats = { Cell.Empty, Cell.Wall, Cell.Sand, Cell.Water, Cell.Oil, Cell.Smoke };
            var before = new int[mats.Length];
            for (int i = 0; i < mats.Length; i++) before[i] = world.Count(mats[i]);

            world.Step(300);

            bool ok = true; var sb = new System.Text.StringBuilder();
            for (int i = 0; i < mats.Length; i++)
            {
                int after = world.Count(mats[i]);
                ok &= after == before[i];
                sb.Append($"{Cell.Name[mats[i]]}:{before[i]}→{after} ");
            }
            Check(ok, "per-material counts invariant over 300 steps  " + sb.ToString());
        }

        /// <summary>Sand dropped in a vacuum must end up on the floor (the bottom edge acts as a wall).</summary>
        private static void CheckSandSettles()
        {
            Console.WriteLine("Behaviors:");
            int h = 48, w = 48;
            var world = new FallingSandWorld(h, w, seed: 1, withBoundary: false);
            for (int r = 2; r < 8; r++) for (int c = 20; c < 28; c++) world.Grid.SetCell(r, c, Cell.Sand);
            int total = world.Count(Cell.Sand);
            world.Step(400);
            var snap = world.Grid.Snapshot();
            int atFloor = 0;
            for (int r = h - 10; r < h; r++) for (int c = 0; c < w; c++) if (snap[r * w + c] == Cell.Sand) atFloor++;
            Check(atFloor == total, $"all {total} sand settled to the bottom 10 rows ({atFloor})");
        }

        /// <summary>A jumble of all four movable materials must stratify by density: smoke over oil over water over sand.</summary>
        private static void CheckStratification()
        {
            int h = 44, w = 44;
            var world = new FallingSandWorld(h, w, seed: 3, withBoundary: true);
            var rng = new Random(3);
            int[] mix = { Cell.Water, Cell.Water, Cell.Oil, Cell.Sand, Cell.Smoke };
            for (int r = 4; r < h - 3; r++) for (int c = 3; c < w - 3; c++) world.Grid.SetCell(r, c, mix[rng.Next(mix.Length)]);
            world.Step(1400);
            var snap = world.Grid.Snapshot();
            double sm = MeanRow(snap, h, w, Cell.Smoke), oi = MeanRow(snap, h, w, Cell.Oil),
                   wa = MeanRow(snap, h, w, Cell.Water), sa = MeanRow(snap, h, w, Cell.Sand);
            Console.WriteLine($"    mean rows  smoke={sm:F1} oil={oi:F1} water={wa:F1} sand={sa:F1}  (smaller = higher)");
            Check(sm < oi && oi < wa && wa < sa, "density stratification: smoke < oil < water < sand");
        }

        /// <summary>Water poured as a tall column must LEVEL — pool into a near-flat slab, not stay a column or pile.</summary>
        private static void CheckWaterLevels()
        {
            int h = 40, w = 40;
            var world = new FallingSandWorld(h, w, seed: 5, withBoundary: true);
            for (int r = 3; r < 15; r++) for (int c = 3; c < 7; c++) world.Grid.SetCell(r, c, Cell.Water);   // tall column
            int total = world.Count(Cell.Water);
            world.Step(4000);
            var snap = world.Grid.Snapshot();
            int rowsWithWater = 0;
            for (int r = 0; r < h; r++)
            {
                bool any = false;
                for (int c = 0; c < w; c++) if (snap[r * w + c] == Cell.Water) { any = true; break; }
                if (any) rowsWithWater++;
            }
            int interiorWidth = w - 4;   // minus the 2-thick walls each side
            int flatDepth = (total + interiorWidth - 1) / interiorWidth;
            Check(world.Count(Cell.Water) == total, $"water mass conserved ({total})");
            Check(rowsWithWater <= flatDepth + 2, $"water leveled: occupies {rowsWithWater} rows (flat depth {flatDepth}, started 12 tall)");
        }

        /// <summary>The brush must stamp a clipped disk of roughly the right area, centred where asked.</summary>
        private static void CheckPaint()
        {
            int h = 40, w = 40;
            var world = new FallingSandWorld(h, w, seed: 1, withBoundary: false);
            world.BrushMaterial = Cell.Wall; world.BrushRadius = 5;
            world.Paint(20, 20);
            int painted = world.Count(Cell.Wall);
            double expected = Math.PI * 25;   // ~78.5 cells for r=5
            Check(Math.Abs(painted - expected) < 20, $"disk brush painted {painted} cells (~{expected:F0} expected)");
            Check(world.Grid.GetCell(20, 20) == Cell.Wall, "brush centre is painted");
            Check(world.Grid.GetCell(20, 20 + 6) == Cell.Empty, "just outside the radius is untouched");
        }

        /// <summary>An emitter must ADD material over time and produce a falling stream (mass grows, material reaches lower rows).</summary>
        private static void CheckEmitter()
        {
            int h = 48, w = 48;
            var world = new FallingSandWorld(h, w, seed: 2, withBoundary: true);
            world.AddEmitter(4, 24, Cell.Sand);
            int before = world.Count(Cell.Sand);
            world.Step(300);
            int after = world.Count(Cell.Sand);
            var snap = world.Grid.Snapshot();
            bool reachedBottom = false;
            for (int c = 0; c < w && !reachedBottom; c++) if (snap[(h - 3) * w + c] == Cell.Sand) reachedBottom = true;
            Check(after > before, $"emitter added sand ({before} → {after})");
            Check(reachedBottom, "emitted sand fell and reached the floor");
        }

        /// <summary>Same seed + same actions must give a bit-identical grid — the randomness is reproducible.</summary>
        private static void CheckDeterminism()
        {
            Console.WriteLine("Reproducibility:");
            int[] Run()
            {
                var world = new FallingSandWorld(30, 30, seed: 42, withBoundary: true);
                world.BrushMaterial = Cell.Water; world.BrushRadius = 4;
                world.Paint(6, 15);
                world.Paint(6, 8);
                world.Step(120);
                return world.Grid.Snapshot();
            }
            var a = Run();
            var b = Run();
            bool identical = a.Length == b.Length;
            for (int i = 0; identical && i < a.Length; i++) identical &= a[i] == b[i];
            Check(identical, "two runs with the same seed produce identical grids");
        }

        /// <summary>With boundary walls and no emitters, no loose material escapes: the non-wall count is invariant.</summary>
        private static void CheckBoundaryContains()
        {
            int h = 40, w = 40;
            var world = new FallingSandWorld(h, w, seed: 9, withBoundary: true);
            world.BrushMaterial = Cell.Sand; world.BrushRadius = 6;
            world.Paint(10, 20);
            world.BrushMaterial = Cell.Water; world.BrushRadius = 6;
            world.Paint(6, 12);
            int loose = world.Count(Cell.Sand) + world.Count(Cell.Water);
            world.Step(600);
            int looseAfter = world.Count(Cell.Sand) + world.Count(Cell.Water);
            Check(loose == looseAfter, $"boundary contains material: loose count {loose} → {looseAfter}");
        }
    }
}
