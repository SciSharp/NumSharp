using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NumSharp.Examples.MaterialLab.Simulation;

namespace NumSharp.Examples.MaterialLab.Verification
{
    /// <summary>
    /// Console gate for the Material Lab physics. It compiles the exact simulation source the app runs and checks
    /// each physical claim the lab makes with a MEASURED invariant — conservation in the transfers, volume kept by
    /// a resting liquid, buoyancy ordering, the sand pile's slope, elastic recovery vs. plastic denting, walls
    /// that hold, fills that never over-pack — plus determinism and "every scene runs clean". A non-zero exit
    /// code is the number of failed checks.
    /// </summary>
    /// <remarks>
    /// Checks run on a 72-row grid (the app's "low" quality) so the whole gate takes well under a minute; every
    /// law under test is resolution-independent, which is also why the thresholds are stated in cells or as
    /// ratios. Each line prints the measured value next to the verdict, so a drift is visible before it fails.
    /// </remarks>
    public static unsafe class Program
    {
        /// <summary>Grid rows used by every check (Δx = 1/72).</summary>
        private const int Rows = 72;

        /// <summary>Failed checks (the exit code) and checks run.</summary>
        private static int _failures, _checks;

        /// <summary>Runs every check and reports.</summary>
        /// <param name="args">Unused.</param>
        /// <returns>0 when all checks pass, else the number of failures.</returns>
        public static int Main(string[] args)
        {
            // The report uses Δ, φ, °, → and friends; without UTF-8 a Windows console prints them as '?'.
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("NumSharp Material Lab — physics verification");
            Console.WriteLine($"grid rows {Rows}, materials {Materials.Count}\n");
            var total = Stopwatch.StartNew();

            Section("Particle ↔ grid transfers");
            CheckFusedMatchesReference();
            CheckP2GConservation();

            Section("Liquids");
            CheckWaterAtRestKeepsVolume();
            CheckBuoyancyOrdering();
            CheckLavaFreezesToRock();
            CheckTiltedGravity();

            Section("Granular materials");
            CheckSandReposeAngle();
            CheckGranularCollapseScaling();
            CheckSnowCompacts();
            CheckSandSinksInWater();

            Section("Solids");
            CheckJellyRecoversClayDents();
            CheckRubberBounces();

            Section("Obstacles and painting");
            CheckWallsHoldLiquid();
            CheckCarveAndReset();
            CheckFillNeverOverpacks();

            Section("Robustness");
            CheckEverySceneRunsClean();
            CheckDeterminism();

            Console.WriteLine();
            Console.WriteLine($"{_checks - _failures}/{_checks} checks passed in {total.Elapsed.TotalSeconds:F1} s.");
            Console.WriteLine(_failures == 0 ? "ALL CHECKS PASSED." : $"{_failures} CHECK(S) FAILED.");
            return _failures;
        }

        // ------------------------------------------------------------------ reporting and helpers

        /// <summary>Prints a section header.</summary>
        /// <param name="title">Section name.</param>
        private static void Section(string title) => Console.WriteLine($"\n{title}");

        /// <summary>Records one verdict and prints it with the measurement that decided it.</summary>
        /// <param name="ok">Whether the claim held.</param>
        /// <param name="claim">What was claimed.</param>
        /// <param name="measured">The measured values (printed either way, so drift is visible before failure).</param>
        private static void Check(bool ok, string claim, string measured)
        {
            _checks++;
            if (!ok) _failures++;
            Console.WriteLine($"  [{(ok ? "PASS" : "FAIL")}] {claim} — {measured}");
        }

        /// <summary>A fresh 16:9 world on the verification grid. The caller owns (and must dispose) it.</summary>
        /// <param name="seed">Particle-jitter seed.</param>
        /// <returns>The world.</returns>
        private static MpmWorld NewWorld(int seed = 1) => new MpmWorld(Rows, 16f / 9f, seed);

        /// <summary>The y of the tank floor surface (the inner edge of the bottom wall).</summary>
        /// <param name="w">World.</param>
        /// <returns>Floor height in units.</returns>
        private static float Floor(MpmWorld w) => (MpmGrid.Border + 0.5f) * w.Grid.Dx;

        /// <summary>The x of the right tank wall's inner edge.</summary>
        /// <param name="w">World.</param>
        /// <returns>Right wall position in units.</returns>
        private static float Right(MpmWorld w) => w.Grid.Width - (MpmGrid.Border + 0.5f) * w.Grid.Dx;

        /// <summary>Advances the world by whole frames.</summary>
        /// <param name="w">World.</param>
        /// <param name="frames">Frames to run.</param>
        private static void Run(MpmWorld w, int frames)
        {
            for (int f = 0; f < frames; f++) w.StepFrame();
        }

        /// <summary>Copies one particle field of a group into a managed array (for statistics).</summary>
        /// <param name="g">Group.</param>
        /// <param name="f">Field.</param>
        /// <returns>The field values of the live particles.</returns>
        private static float[] Field(ParticleGroup g, PField f)
        {
            var a = new float[g.Count];
            if (g.Count > 0) new ReadOnlySpan<float>(g.Ptr(f), g.Count).CopyTo(a);
            return a;
        }

        /// <summary>Mean of a group's field (NaN when the group is empty).</summary>
        /// <param name="g">Group.</param>
        /// <param name="f">Field.</param>
        /// <returns>The mean.</returns>
        private static double Mean(ParticleGroup g, PField f) => g.Count == 0 ? double.NaN : Field(g, f).Average(v => (double)v);

        /// <summary>The q-quantile (0..1) of a group's field.</summary>
        /// <param name="g">Group.</param>
        /// <param name="f">Field.</param>
        /// <param name="q">Quantile.</param>
        /// <returns>The value at that quantile.</returns>
        private static double Quantile(ParticleGroup g, PField f, double q)
        {
            var a = Field(g, f);
            Array.Sort(a);
            return a.Length == 0 ? double.NaN : a[Math.Clamp((int)(q * (a.Length - 1)), 0, a.Length - 1)];
        }

        /// <summary>Largest particle speed in a group.</summary>
        /// <param name="g">Group.</param>
        /// <returns>max |v|.</returns>
        private static double MaxSpeed(ParticleGroup g)
        {
            var vx = Field(g, PField.VX); var vy = Field(g, PField.VY);
            double m = 0;
            for (int i = 0; i < vx.Length; i++) m = Math.Max(m, Math.Sqrt(vx[i] * vx[i] + vy[i] * vy[i]));
            return m;
        }

        /// <summary>Axis-aligned bounds of a group.</summary>
        /// <param name="g">Group.</param>
        /// <returns>(minX, minY, maxX, maxY).</returns>
        private static (double X0, double Y0, double X1, double Y1) Bounds(ParticleGroup g)
        {
            var x = Field(g, PField.X); var y = Field(g, PField.Y);
            return (x.Min(), y.Min(), x.Max(), y.Max());
        }

        // ------------------------------------------------------------------ transfers

        /// <summary>
        /// The real-time fused SIMD transfers and the pure-NumSharp reference transfers (<c>np.bincount</c> scatter,
        /// <c>np.take</c> gather) are two implementations of one algorithm: after 3 frames (~60 substeps) of every
        /// material interacting, they must agree to float rounding. Positions compare in cells; the APIC velocity
        /// gradient C (whose entries reach thousands in an impact) compares relative to its largest magnitude.
        /// </summary>
        private static void CheckFusedMatchesReference()
        {
            using var a = NewWorld(3);
            using var b = NewWorld(3);
            foreach (var w in new[] { a, b }) BuildMixedScene(w);
            b.Transfers = TransferMode.Reference;
            Run(a, 3); Run(b, 3);
            double posCells = 0, vel = 0, jDiff = 0, cDiff = 0, cMax = 1e-9;
            bool countsMatch = true;
            foreach (MaterialId id in Enum.GetValues<MaterialId>())
            {
                var ga = a.Group(id); var gb = b.Group(id);
                if (ga.Count != gb.Count) { countsMatch = false; continue; }
                for (int i = 0; i < ga.Count; i++)
                {
                    posCells = Math.Max(posCells, Math.Max(Math.Abs(ga.Ptr(PField.X)[i] - gb.Ptr(PField.X)[i]), Math.Abs(ga.Ptr(PField.Y)[i] - gb.Ptr(PField.Y)[i])) / a.Grid.Dx);
                    vel = Math.Max(vel, Math.Max(Math.Abs(ga.Ptr(PField.VX)[i] - gb.Ptr(PField.VX)[i]), Math.Abs(ga.Ptr(PField.VY)[i] - gb.Ptr(PField.VY)[i])));
                    jDiff = Math.Max(jDiff, Math.Abs(ga.Ptr(PField.Jp)[i] - gb.Ptr(PField.Jp)[i]));
                    foreach (var c in new[] { PField.C00, PField.C01, PField.C10, PField.C11 })
                    {
                        cDiff = Math.Max(cDiff, Math.Abs(ga.Ptr(c)[i] - gb.Ptr(c)[i]));
                        cMax = Math.Max(cMax, Math.Abs(ga.Ptr(c)[i]));
                    }
                }
            }
            Check(countsMatch && posCells < 1e-3 && vel < 1e-3 && jDiff < 1e-3 && cDiff / cMax < 1e-4,
                "fused SIMD transfers match the pure-NumSharp reference",
                $"max Δposition {posCells:E1} cells, Δv {vel:E1}, ΔJ {jDiff:E1}, ΔC {cDiff / cMax:E1} of max|C| over {a.ParticleCount} particles");
        }

        /// <summary>
        /// Particle-to-grid is a pure redistribution: whatever affine momentum the particles carry, the grid must
        /// receive exactly their total mass, their total volume and their total linear momentum (the quadratic
        /// B-spline satisfies Σ w·(xᵢ − xₚ) = 0, so APIC's affine term adds none). Checked for BOTH transfer
        /// implementations on a mid-flight mixed scene, against sums taken in double precision.
        /// </summary>
        private static void CheckP2GConservation()
        {
            using var w = NewWorld(5);
            BuildMixedScene(w);
            Run(w, 20);   // mid-flight: nonzero velocities and stresses everywhere
            double pm = 0, pmx = 0, pmy = 0, pAbs = 0, pv = 0;
            float vol = 0.25f * w.Grid.Dx * w.Grid.Dx;
            foreach (var g in w.Groups)
            {
                float m = w.ParticleMass(g.Material.Id);
                for (int i = 0; i < g.Count; i++)
                {
                    pm += m; pv += vol;
                    pmx += m * g.Ptr(PField.VX)[i]; pmy += m * g.Ptr(PField.VY)[i];
                    pAbs += m * (Math.Abs(g.Ptr(PField.VX)[i]) + Math.Abs(g.Ptr(PField.VY)[i]));
                }
            }
            foreach (var mode in new[] { TransferMode.Fused, TransferMode.Reference })
            {
                w.Grid.ClearNodes();
                if (mode == TransferMode.Fused)
                {
                    foreach (var g in w.Groups)
                        Transfers.P2GFused(w.Grid, g, w.ParticleMass(g.Material.Id), vol, g.Material.Id == MaterialId.Water);
                }
                else
                {
                    var groups = w.Groups.ToArray();
                    Transfers.P2GReference(w.Grid, groups, groups.Select(g => w.ParticleMass(g.Material.Id)).ToArray(), vol,
                        groups.Select(g => g.Material.Id == MaterialId.Water).ToArray());
                }
                double gm = 0, gmx = 0, gmy = 0, gv = 0;
                float* nodes = w.Grid.Nodes.Unsafe.Pointer<float>();
                for (int k = 0; k < w.Grid.G; k++)
                {
                    float* nd = nodes + (long)k * MpmGrid.Lanes;
                    gm += nd[MpmGrid.LaneMass]; gmx += nd[MpmGrid.LaneMomX]; gmy += nd[MpmGrid.LaneMomY]; gv += nd[MpmGrid.LaneVolume];
                }
                double em = Math.Abs(gm - pm) / pm, ev = Math.Abs(gv - pv) / pv;
                double ep = Math.Max(Math.Abs(gmx - pmx), Math.Abs(gmy - pmy)) / pAbs;
                Check(em < 1e-5 && ev < 1e-5 && ep < 1e-5,
                    $"P2G ({mode}) conserves mass, volume and momentum",
                    $"relative error mass {em:E1}, volume {ev:E1}, momentum {ep:E1} (of Σm|v|)");
            }
        }

        /// <summary>One block of every placeable material, spread across the tank, some of it moving.</summary>
        /// <param name="w">World to fill.</param>
        private static void BuildMixedScene(MpmWorld w)
        {
            float f = Floor(w);
            w.FillRect(MaterialId.Water, f, f, 0.45f, 0.55f);
            w.FillRect(MaterialId.Oil, 0.50f, f, 0.65f, 0.20f);
            w.FillRect(MaterialId.Honey, 0.55f, 0.55f, 0.65f, 0.70f);
            w.FillRect(MaterialId.Lava, 0.70f, 0.10f, 0.82f, 0.25f);
            w.FillRect(MaterialId.Sand, 1.05f, f, 1.35f, 0.28f);
            w.FillDisk(MaterialId.Snow, 1.55f, 0.60f, 0.07f, -3f, 0f);
            w.FillRect(MaterialId.Jelly, 0.85f, 0.40f, 0.98f, 0.52f);
            w.FillDisk(MaterialId.Rubber, 0.95f, 0.80f, 0.05f, 1f, 0f);
            w.FillRect(MaterialId.Clay, 1.20f, 0.60f, 1.30f, 0.70f);
            w.FillRect(MaterialId.Rock, 1.40f, f, 1.50f, 0.12f);
        }

        // ------------------------------------------------------------------ liquids

        /// <summary>
        /// A pool left alone must stay a pool: calm (no self-excited jitter), flat, and — the property an
        /// integrated volume ratio loses — the same VOLUME, compressed by exactly what its equation of state
        /// predicts. Every particle carries V₀ = Δx²/4, so the pool's area is N·V₀; at rest the liquid fills the
        /// tank from ½ cell off each wall surface (the push-out distance the wall-volume lane assumes). A
        /// weakly-compressible liquid of bulk modulus K then squeezes by ρg·d/K at depth d, which lowers the mean
        /// particle height by the fraction ρgh/(2K) of the half-depth; the measured drop must match that
        /// prediction to within 1.5 percentage points, and the pool must end flat and still.
        /// </summary>
        private static void CheckWaterAtRestKeepsVolume()
        {
            using var w = NewWorld();
            float dx = w.Grid.Dx;
            // Tank wall surfaces sit at Border − ½ cells from the domain edge; the liquid rests ½ cell further in.
            float bottom = MpmGrid.Border * dx, left = MpmGrid.Border * dx, right = (w.Grid.Nx - 1 - MpmGrid.Border) * dx;
            w.FillRect(MaterialId.Water, Floor(w), Floor(w), Right(w), 0.40f);
            Run(w, 500);
            var water = w.Group(MaterialId.Water);
            var m = Materials.Get(MaterialId.Water);
            double depth = water.Count * 0.25 * dx * dx / (right - left);
            double expectedMean = bottom + depth / 2;
            double meanY = Mean(water, PField.Y);
            double shortfall = (expectedMean - meanY) / (depth / 2);
            double predicted = m.Density * w.Gravity * depth / (2 * m.BulkModulus);
            // Flatness: the highest particle in each of 24 columns.
            var x = Field(water, PField.X); var y = Field(water, PField.Y);
            var top = new double[24];
            for (int i = 0; i < x.Length; i++)
            {
                int bin = Math.Clamp((int)((x[i] - left) / (right - left) * top.Length), 0, top.Length - 1);
                top[bin] = Math.Max(top[bin], y[i]);
            }
            double topMean = top.Average(), topSpread = Math.Sqrt(top.Average(t => (t - topMean) * (t - topMean))) / dx;
            double vmax = MaxSpeed(water);
            Check(Math.Abs(shortfall - predicted) < 0.015 && topSpread < 1.0 && vmax < 0.6,
                "water at rest keeps its volume, lies flat and stays calm",
                $"depth {depth:F3}: compressed {shortfall * 100:0.0} % (equation of state predicts {predicted * 100:0.0} %), surface spread {topSpread:F2} cells, max speed {vmax:F2}");
        }

        /// <summary>
        /// Buoyancy orders liquids by density with no rule written for it: oil (0.72) released at the bottom of a
        /// water tank must end ON the water, honey (1.4) dropped from above must end UNDER it. Checked on the
        /// lab's own "Oil, Water &amp; Honey" scene after ~2 simulated seconds.
        /// </summary>
        private static void CheckBuoyancyOrdering()
        {
            using var w = NewWorld();
            Scenes.Load(w, 5);
            Run(w, 700);
            var oil = w.Group(MaterialId.Oil); var water = w.Group(MaterialId.Water); var honey = w.Group(MaterialId.Honey);
            double waterTop = Quantile(water, PField.Y, 0.98);
            // "On top": above the water surface minus two cells (the interface layer is ~2 cells thick).
            var oy = Field(oil, PField.Y);
            double onTop = oy.Count(v => v > waterTop - 2 * w.Grid.Dx) / (double)oy.Length;
            var hy = Field(honey, PField.Y);
            double waterBottom = Quantile(water, PField.Y, 0.05);
            double below = hy.Count(v => v < waterBottom + 3 * w.Grid.Dx) / (double)hy.Length;
            double mo = Mean(oil, PField.Y), mw = Mean(water, PField.Y), mh = Mean(honey, PField.Y);
            Check(mo > mw && mw > mh && onTop > 0.9 && below > 0.8,
                "oil floats to the top and honey sinks to the bottom",
                $"mean height oil {mo:F3} > water {mw:F3} > honey {mh:F3}; {onTop:P0} of oil at the surface, {below:P0} of honey on the bottom");
        }

        /// <summary>
        /// Lava cools in air and far faster in water; below the solidification temperature a particle becomes rock
        /// (same position and velocity, now an elastic solid). The "Lava Flow" scene pours lava into a lake.
        /// </summary>
        private static void CheckLavaFreezesToRock()
        {
            using var w = NewWorld();
            Scenes.Load(w, 6);
            Run(w, 700);
            int rock = w.Group(MaterialId.Rock).Count, lava = w.Group(MaterialId.Lava).Count;
            double meanT = lava > 0 ? Mean(w.Group(MaterialId.Lava), PField.Temp) : 0;
            Check(rock > 200 && meanT < 1.0,
                "lava cools and freezes into rock",
                $"{rock} rock particles formed, {lava} still molten at mean temperature {meanT:F2}");
        }

        /// <summary>Tilting gravity must move a pool downhill: its center of mass shifts toward the low side.</summary>
        private static void CheckTiltedGravity()
        {
            using var w = NewWorld();
            float f = Floor(w), right = Right(w);
            w.FillRect(MaterialId.Water, f, f, right, 0.30f);
            Run(w, 60);
            double before = Mean(w.Group(MaterialId.Water), PField.X);
            w.GravityAngle = 0.5f;   // tilted to the right
            Run(w, 200);
            double after = Mean(w.Group(MaterialId.Water), PField.X);
            Check(after - before > 0.05,
                "tilted gravity pours the pool downhill",
                $"center of mass x {before:F3} → {after:F3}");
        }

        // ------------------------------------------------------------------ granular

        /// <summary>
        /// Sand poured slowly onto one spot builds a cone whose flanks stand at the angle of repose, set by the
        /// Drucker–Prager friction angle (35°) — neither a puddle (a liquid) nor a tower (a solid). The slope is
        /// the heap's height over the half-width of its base; quantiles, not extremes, so one bouncing grain
        /// cannot move the answer.
        /// </summary>
        private static void CheckSandReposeAngle()
        {
            using var w = NewWorld();
            float f = Floor(w), cx = 0.5f * w.Grid.Width;
            w.Emitters.Add(new Emitter { Material = MaterialId.Sand, X = cx, Y = 0.40f, Radius = 0.02f, Budget = 2200 });
            Run(w, 700);
            var sand = w.Group(MaterialId.Sand);
            double height = Quantile(sand, PField.Y, 0.995) - f;
            double halfBase = (Quantile(sand, PField.X, 0.995) - Quantile(sand, PField.X, 0.005)) / 2;
            double angle = Math.Atan2(height, halfBase) * 180 / Math.PI;
            double vmax = MaxSpeed(sand);
            Check(angle > 22 && angle < 42 && vmax < 0.5,
                "poured sand builds a cone at its angle of repose",
                $"{sand.Count} grains, heap {height:F3} high on a {2 * halfBase:F3} base: flanks at {angle:F1}° (friction angle 35°), max speed {vmax:F2}");
        }

        /// <summary>
        /// The collapse of a tall granular column is a classic laboratory experiment with published scaling laws
        /// (Lube, Huppert, Sparks &amp; Freundt 2005, "Collapses of two-dimensional granular columns", Phys. Rev. E
        /// 72, 041301): for aspect ratio a = H₀/L₀ &gt; 2.8 the run-out obeys (L∞ − L₀)/L₀ ≈ 1.9·a^(2/3) and the final
        /// height H∞/L₀ ≈ a^(2/5). A symmetric column of half-width L₀ is two such experiments back to back. The
        /// sand here — Drucker–Prager plasticity on a 12-cell-wide column, with no rule written for collapse —
        /// must land within 40% of both laws.
        /// </summary>
        private static void CheckGranularCollapseScaling()
        {
            using var w = NewWorld();
            float f = Floor(w), cx = 0.5f * w.Grid.Width, l0 = 0.08f, h0 = 0.45f;
            w.FillRect(MaterialId.Sand, cx - l0, f, cx + l0, f + h0);
            Run(w, 600);
            var sand = w.Group(MaterialId.Sand);
            double a = h0 / l0;
            double runOut = (Quantile(sand, PField.X, 0.995) - Quantile(sand, PField.X, 0.005)) / 2;
            double height = Quantile(sand, PField.Y, 0.995) - f;
            double runOutLaw = l0 * (1 + 1.9 * Math.Pow(a, 2.0 / 3.0)), heightLaw = l0 * Math.Pow(a, 0.4);
            double rr = runOut / runOutLaw, hr = height / heightLaw;
            Check(rr > 0.6 && rr < 1.4 && hr > 0.6 && hr < 1.4,
                "a collapsing sand column follows the laboratory scaling laws",
                $"a = {a:F2}: run-out {runOut:F3} vs {runOutLaw:F3} (×{rr:F2}), final height {height:F3} vs {heightLaw:F3} (×{hr:F2})");
        }

        /// <summary>
        /// Snow is plastic in compression: a snowball smashed into the floor records its compaction in Jp (the
        /// plastic volume ratio, 1 = fresh), and compacted snow hardens by exp(ξ(1 − Jp)).
        /// </summary>
        private static void CheckSnowCompacts()
        {
            using var w = NewWorld();
            w.FillDisk(MaterialId.Snow, 0.9f, 0.25f, 0.07f, 0f, -12f);
            Run(w, 120);
            var jp = Field(w.Group(MaterialId.Snow), PField.Jp);
            double minJp = jp.Min(), compacted = jp.Count(v => v < 0.97) / (double)jp.Length;
            var m = Materials.Get(MaterialId.Snow);
            double hardening = Math.Min(Math.Exp(m.SnowHardening * (1 - minJp)), m.SnowHardeningMax);
            Check(minJp < 0.9 && compacted > 0.1,
                "a snowball compacts (and hardens) where it hits",
                $"min Jp {minJp:F3}, {compacted:P0} of the snow compacted, peak hardening ×{hardening:F2}");
        }

        /// <summary>Sand poured into water settles through it to the bottom (denser than water, drift-flux settling).</summary>
        private static void CheckSandSinksInWater()
        {
            using var w = NewWorld();
            float f = Floor(w), right = Right(w);
            w.FillRect(MaterialId.Sand, 0.8f, 0.55f, 1.0f, 0.70f);
            w.FillRect(MaterialId.Water, f, f, right, 0.35f);
            Run(w, 500);
            double ms = Mean(w.Group(MaterialId.Sand), PField.Y), mw = Mean(w.Group(MaterialId.Water), PField.Y);
            double sandTop = Quantile(w.Group(MaterialId.Sand), PField.Y, 0.95), waterTop = Quantile(w.Group(MaterialId.Water), PField.Y, 0.98);
            Check(ms < mw && sandTop < waterTop,
                "sand sinks through water to the bottom",
                $"mean height sand {ms:F3} < water {mw:F3}; sand top {sandTop:F3} below the water surface {waterTop:F3}");
        }

        // ------------------------------------------------------------------ solids

        /// <summary>
        /// The same square dropped from the same height: jelly (elastic) must spring back to a square and keep its
        /// area (det F ≈ 1), clay (von Mises plastic) must keep the dent — flatter than the jelly afterwards.
        /// </summary>
        private static void CheckJellyRecoversClayDents()
        {
            double Drop(MaterialId id, out double meanDetF)
            {
                using var w = NewWorld();
                float cx = 0.9f;
                w.FillRect(id, cx - 0.08f, 0.45f, cx + 0.08f, 0.61f, 0f, -6f);
                Run(w, 600);
                var g = w.Group(id);
                var b = Bounds(g);
                var f00 = Field(g, PField.F00); var f01 = Field(g, PField.F01); var f10 = Field(g, PField.F10); var f11 = Field(g, PField.F11);
                meanDetF = Enumerable.Range(0, g.Count).Average(i => (double)f00[i] * f11[i] - (double)f01[i] * f10[i]);
                return (b.Y1 - b.Y0) / (b.X1 - b.X0);
            }
            double jelly = Drop(MaterialId.Jelly, out double jellyDet);
            double clay = Drop(MaterialId.Clay, out _);
            Check(Math.Abs(jelly - 1) < 0.15 && Math.Abs(jellyDet - 1) < 0.05 && clay < jelly - 0.1,
                "jelly springs back to its shape, clay keeps the dent",
                $"height/width after the drop: jelly {jelly:F2} (mean det F {jellyDet:F3}), clay {clay:F2} (started 1.00)");
        }

        /// <summary>A rubber ball dropped onto the floor bounces: after the impact it rises again (positive mean vertical velocity).</summary>
        private static void CheckRubberBounces()
        {
            using var w = NewWorld();
            w.FillDisk(MaterialId.Rubber, 0.9f, 0.55f, 0.05f);
            var rubber = w.Group(MaterialId.Rubber);
            double lowest = double.MaxValue, bestUp = 0;
            bool hit = false;
            for (int frame = 0; frame < 400; frame++)
            {
                w.StepFrame();
                double vy = Mean(rubber, PField.VY), y = Mean(rubber, PField.Y);
                lowest = Math.Min(lowest, y);
                if (y < 0.12) hit = true;
                if (hit) bestUp = Math.Max(bestUp, vy);
            }
            Check(hit && bestUp > 1.0,
                "a rubber ball bounces off the floor",
                $"lowest center {lowest:F3}, fastest rebound {bestUp:F2} units/s");
        }

        // ------------------------------------------------------------------ obstacles and painting

        /// <summary>
        /// A painted wall is a real boundary: a shelf spanning the tank holds a pool of water on top of it, and no
        /// particle leaks below the shelf's lower face.
        /// </summary>
        private static void CheckWallsHoldLiquid()
        {
            using var w = NewWorld();
            float shelf = 0.45f, thick = 0.03f;
            w.Grid.PaintSegment(0f, shelf, w.Grid.Width, shelf, thick);
            w.FillRect(MaterialId.Water, 0.3f, shelf + thick, 1.4f, 0.75f);
            int n = w.Group(MaterialId.Water).Count;
            Run(w, 400);
            var y = Field(w.Group(MaterialId.Water), PField.Y);
            int leaked = y.Count(v => v < shelf - thick / 2);
            Check(leaked == 0 && y.Length == n,
                "a painted shelf holds water — nothing leaks through",
                $"{leaked} of {n} particles below the shelf, lowest particle {y.Min():F3} (shelf top {shelf + thick / 2:F3})");
        }

        /// <summary>
        /// Painting and carving edit one signed distance field: a painted disk is solid (φ &lt; 0 at its center),
        /// carving the same disk makes it free space again, and resetting restores the bare tank.
        /// </summary>
        private static void CheckCarveAndReset()
        {
            using var w = NewWorld();
            float cx = 0.9f, cy = 0.5f;
            int i = (int)MathF.Round(cx * w.Grid.InvDx), j = (int)MathF.Round(cy * w.Grid.InvDx);
            float Phi() => w.Grid.Phi.Unsafe.Pointer<float>()[j * w.Grid.Nx + i];
            float free = Phi();
            w.Grid.PaintDisk(cx, cy, 0.08f, solid: true);
            float painted = Phi();
            w.Grid.PaintDisk(cx, cy, 0.08f, solid: false);
            float carved = Phi();
            w.Grid.PaintDisk(cx, cy, 0.05f, solid: true);
            w.Grid.ResetObstacles();
            float reset = Phi();
            Check(free > 0 && painted < -3 && carved > 0 && Math.Abs(reset - free) < 1e-4,
                "painting, carving and resetting the obstacle field",
                $"φ at the disk center (cells): free {free:F1} → painted {painted:F1} → carved {carved:F1} → reset {reset:F1}");
        }

        /// <summary>
        /// Every fill tops cells up to the rest packing and never beyond (liquids read their volume from the local
        /// occupancy, so an over-packed cell would store pressure and erupt): filling the same rectangle twice adds
        /// nothing, a disk inside it adds nothing, and a rectangle half over it adds only the uncovered half.
        /// </summary>
        private static void CheckFillNeverOverpacks()
        {
            using var w = NewWorld();
            int first = w.FillRect(MaterialId.Water, 0.4f, 0.2f, 0.8f, 0.5f);
            int again = w.FillRect(MaterialId.Water, 0.4f, 0.2f, 0.8f, 0.5f);
            int disk = w.FillDisk(MaterialId.Oil, 0.6f, 0.35f, 0.08f);
            int half = w.FillRect(MaterialId.Sand, 0.6f, 0.2f, 1.0f, 0.5f);
            double ratio = half / (first / 2.0);
            Check(first > 0 && again == 0 && disk == 0 && Math.Abs(ratio - 1) < 0.1,
                "fills top cells up to rest packing and never over-pack",
                $"first fill {first}, same rect again {again}, disk inside {disk}, half-overlapping rect {half} (= {ratio:F2} × the free half)");
        }

        // ------------------------------------------------------------------ robustness

        /// <summary>
        /// Every ready-made scene runs 200 frames with every particle finite and inside the tank — the stability
        /// claim of the adaptive time step (CFL on the fastest wave present, plus a half-cell-per-substep bound).
        /// </summary>
        private static void CheckEverySceneRunsClean()
        {
            for (int s = 0; s < Scenes.All.Length; s++)
            {
                using var w = NewWorld();
                var scene = Scenes.Load(w, s);
                var sw = Stopwatch.StartNew();
                Run(w, 200);
                double ms = sw.Elapsed.TotalMilliseconds / 200;
                int bad = 0, outside = 0;
                float width = w.Grid.Width;
                foreach (var g in w.Groups)
                {
                    for (int i = 0; i < g.Count; i++)
                    {
                        float x = g.Ptr(PField.X)[i], y = g.Ptr(PField.Y)[i];
                        for (int k = 0; k < ParticleGroup.FieldCount; k++)
                            if (!float.IsFinite(g.Ptr((PField)k)[i])) { bad++; break; }
                        if (x <= 0 || y <= 0 || x >= width || y >= 1) outside++;
                    }
                }
                bool expectParticles = scene.Name != "Sandbox";
                Check(bad == 0 && outside == 0 && (!expectParticles || w.ParticleCount > 0),
                    $"scene \"{scene.Name}\" runs clean",
                    $"{w.ParticleCount} particles, {bad} non-finite, {outside} outside the tank, {ms:F1} ms/frame at {Rows} rows");
            }
        }

        /// <summary>Two worlds with the same seed and the same scene evolve bit-for-bit identically (single-threaded, no hidden state).</summary>
        private static void CheckDeterminism()
        {
            using var a = NewWorld(7);
            using var b = NewWorld(7);
            Scenes.Load(a, 0); Scenes.Load(b, 0);
            Run(a, 90); Run(b, 90);
            var ga = a.Group(MaterialId.Water); var gb = b.Group(MaterialId.Water);
            bool same = ga.Count == gb.Count;
            for (int i = 0; same && i < ga.Count; i++)
                same = ga.Ptr(PField.X)[i] == gb.Ptr(PField.X)[i] && ga.Ptr(PField.Y)[i] == gb.Ptr(PField.Y)[i]
                    && ga.Ptr(PField.VX)[i] == gb.Ptr(PField.VX)[i] && ga.Ptr(PField.Jp)[i] == gb.Ptr(PField.Jp)[i];
            Check(same, "same seed, same scene → bit-identical evolution", $"{ga.Count} particles compared after 90 frames");
        }
    }
}
