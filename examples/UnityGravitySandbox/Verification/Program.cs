using System;
using NumSharp;
using NumSharp.Examples.GravitySandbox.Physics;

namespace NumSharp.Examples.GravitySandbox.Verification
{
    /// <summary>
    /// Console entry point that exercises the physics engine and asserts its scientific invariants. It is
    /// the sample's correctness gate: run <c>dotnet run</c> here and a non-zero exit code means a physics
    /// regression. Each check prints its measured quantity so a human can see HOW well it held, not just
    /// that it passed.
    /// </summary>
    public static class Program
    {
        private static int _failures;

        /// <summary>Program entry point. Runs every check and returns the failure count as the exit code.</summary>
        /// <param name="args">Unused.</param>
        /// <returns>0 if every check passed; otherwise the number of failed checks.</returns>
        public static int Main(string[] args)
        {
            Console.WriteLine("NumSharp Gravity Sandbox — physics verification\n");

            CheckScenariosLoadAndRun();
            CheckEnergyConservation();
            CheckMomentumConservation();
            CheckFigureEightPeriodicity();
            CheckSpawnAndMerge();
            CheckRelativisticPrecession();
            CheckIntegratorContrast();

            Console.WriteLine();
            if (_failures == 0) Console.WriteLine("ALL CHECKS PASSED.");
            else Console.WriteLine($"{_failures} CHECK(S) FAILED.");
            return _failures;
        }

        /// <summary>Records and prints a pass/fail line; increments the global failure count on failure.</summary>
        /// <param name="condition">The invariant that must hold.</param>
        /// <param name="message">A human description including the measured value.</param>
        private static void Check(bool condition, string message)
        {
            Console.WriteLine($"  [{(condition ? "PASS" : "FAIL")}] {message}");
            if (!condition) _failures++;
        }

        /// <summary>Sanity: every library scenario builds, loads into a world, and advances without throwing.</summary>
        private static void CheckScenariosLoadAndRun()
        {
            Console.WriteLine("Scenarios load and step:");
            foreach (var s in ScenarioLibrary.All())
            {
                bool ok = true;
                string err = "";
                try
                {
                    var world = new NBodyWorld(s);
                    for (int i = 0; i < 50; i++) world.Step();
                    var d = world.Diagnostics();
                    // A healthy run keeps every quantity finite; a NaN means the step blew up.
                    ok = double.IsFinite(d.Total) && double.IsFinite(d.Kinetic) && double.IsFinite(d.Potential);
                }
                catch (Exception ex) { ok = false; err = " — " + ex.GetType().Name + ": " + ex.Message; }
                Check(ok, $"{s.Name} ({s.InitialState.Count} bodies){err}");
            }
        }

        /// <summary>
        /// Leapfrog must conserve total energy to a tiny bounded band. We run the two-body circular orbit
        /// and the figure-eight for many steps and assert the maximum relative energy drift stays small —
        /// the defining property of a symplectic integrator.
        /// </summary>
        private static void CheckEnergyConservation()
        {
            Console.WriteLine("Energy conservation (leapfrog):");

            double drift1 = MaxEnergyDrift(ScenarioLibrary.TwoBodyCircular(), 40000);
            Check(drift1 < 1e-6, $"two-body max |ΔE/E₀| = {drift1:E3} (< 1e-6)");

            double drift2 = MaxEnergyDrift(ScenarioLibrary.FigureEight(), 25000);
            Check(drift2 < 1e-4, $"figure-eight max |ΔE/E₀| = {drift2:E3} (< 1e-4)");
        }

        /// <summary>Runs a scenario and returns the maximum relative deviation of total energy from its start value.</summary>
        /// <param name="s">The scenario to run.</param>
        /// <param name="steps">How many steps to integrate.</param>
        /// <returns>max |E(t) − E₀| / |E₀| over the run.</returns>
        private static double MaxEnergyDrift(Scenario s, int steps)
        {
            var world = new NBodyWorld(s);
            double e0 = world.Diagnostics().Total;
            double max = 0;
            for (int i = 0; i < steps; i++)
            {
                world.Step();
                double rel = Math.Abs((world.Diagnostics().Total - e0) / e0);
                if (rel > max) max = rel;
            }
            return max;
        }

        /// <summary>
        /// Linear and angular momentum have no source in a closed gravitating system, so they must hold to
        /// round-off regardless of the integrator. This is a stronger check than energy: any asymmetry in
        /// the pairwise force computation would break it immediately.
        /// </summary>
        private static void CheckMomentumConservation()
        {
            Console.WriteLine("Momentum conservation (figure-eight):");
            var world = new NBodyWorld(ScenarioLibrary.FigureEight());
            var d0 = world.Diagnostics();
            for (int i = 0; i < 20000; i++) world.Step();
            var d1 = world.Diagnostics();

            double dp = (d1.LinearMomentum - d0.LinearMomentum).Length;
            double dl = (d1.AngularMomentum - d0.AngularMomentum).Length;
            Check(dp < 1e-10, $"|Δp| = {dp:E3} (< 1e-10)");
            Check(dl < 1e-10, $"|ΔL| = {dl:E3} (< 1e-10)");
        }

        /// <summary>
        /// The figure-eight is periodic with T ≈ 6.3259, so after one period every body should return close
        /// to its starting point. This proves the solver reproduces the actual choreography, not merely
        /// "some bounded motion".
        /// </summary>
        private static void CheckFigureEightPeriodicity()
        {
            Console.WriteLine("Figure-eight periodicity:");
            var s = ScenarioLibrary.FigureEight();
            var start0 = s.InitialState.PositionOf(0);
            var world = new NBodyWorld(s);

            const double period = 6.32591398;
            int steps = (int)Math.Round(period / world.TimeStep);
            for (int i = 0; i < steps; i++) world.Step();

            double back = (world.State.PositionOf(0) - start0).Length;
            Check(back < 5e-3, $"body 0 returns to within {back:E3} of its start after one period (< 5e-3)");
        }

        /// <summary>
        /// Spawning must append a body without disturbing the others, and a merge must conserve total mass
        /// exactly and linear momentum to round-off (a perfectly inelastic collision). We drive two bodies
        /// head-on with collisions on and check the books balance across the merge.
        /// </summary>
        private static void CheckSpawnAndMerge()
        {
            Console.WriteLine("Spawn & merge (accretion):");

            // Two equal masses moving toward each other; touch radius 0.6 so they merge as they close.
            var state = NBodyState.FromArrays(
                new double[,] { { -1, 0, 0 }, { 1, 0, 0 } },
                new double[,] { { 0.5, 0, 0 }, { -0.3, 0, 0 } },
                new double[] { 1, 1 });
            var scenario = new Scenario
            {
                Name = "merge-test", InitialState = state,
                Descriptors = new[]
                {
                    new BodyDescriptor("A", 1, 0, 0, 0.3),
                    new BodyDescriptor("B", 0, 0, 1, 0.3),
                },
                G = 1, Softening = 0.05, TimeStep = 1e-3, EnableCollisions = true,
            };
            var world = new NBodyWorld(scenario);

            var p0 = world.Diagnostics().LinearMomentum;   // 1·0.5 + 1·(−0.3) = 0.2 along x
            double m0 = world.Diagnostics().TotalMass;

            // Spawn a third distant body; it must not change the count-2 pair's imminent merge.
            world.AddBody(new Vector3d(0, 20, 0), new Vector3d(0, 0, 0), 0.5,
                          new BodyDescriptor("C", 0, 1, 0, 0.2));
            Check(world.State.Count == 3, $"after spawn Count = {world.State.Count} (expected 3)");
            double m1 = world.Diagnostics().TotalMass;
            Check(Math.Abs(m1 - (m0 + 0.5)) < 1e-12, $"spawn added mass: total = {m1:F4} (expected {m0 + 0.5:F4})");

            var pAfterSpawn = world.Diagnostics().LinearMomentum;   // C is at rest, so p unchanged

            int mergedAt = -1;
            for (int i = 0; i < 4000 && mergedAt < 0; i++)
            {
                world.Step();
                if (world.State.Count == 2) mergedAt = i;   // 3 → 2 means A and B merged (C is far away)
            }
            Check(mergedAt >= 0, mergedAt >= 0 ? $"A and B merged at step {mergedAt}" : "A and B never merged");

            var p1 = world.Diagnostics().LinearMomentum;
            double dp = (p1 - pAfterSpawn).Length;
            Check(dp < 1e-9, $"momentum conserved across merge: |Δp| = {dp:E3} (< 1e-9)");
            double m2 = world.Diagnostics().TotalMass;
            Check(Math.Abs(m2 - m1) < 1e-12, $"mass conserved across merge: total = {m2:F4} (expected {m1:F4})");
        }

        /// <summary>
        /// The relativistic force model must make an eccentric orbit precess at the leading-order
        /// Schwarzschild rate <c>Δϖ = 6πGM/(c²a(1−e²))</c> per orbit. We measure the angle of successive
        /// perihelion passages and compare their advance to the analytic value — the quantitative test that
        /// the 1PN term is not just "some extra force" but the RIGHT one.
        /// </summary>
        private static void CheckRelativisticPrecession()
        {
            Console.WriteLine("Relativistic perihelion precession:");
            double G = 1, M = 1, a = 1, e = 0.5, c = 10.0;
            double rp = a * (1 - e);
            double vp = Math.Sqrt(G * M * (2 / rp - 1 / a));

            var state = NBodyState.FromArrays(
                new double[,] { { 0, 0, 0 }, { rp, 0, 0 } },
                new double[,] { { 0, 0, 0 }, { 0, vp, 0 } },
                new double[] { M, 1e-6 });                 // near-test-particle so the analytic formula applies
            var solver = new GravitySolver(G, 0.0, ForceModel.RelativisticPrecession, c);
            var integ = new Integrator(IntegrationMethod.Leapfrog);

            double dt = 2e-4;
            double prevR = double.MaxValue, prevPrevR = double.MaxValue;
            double lastPeriAngle = double.NaN;
            double sumAdvance = 0; int nAdvances = 0;

            // Integrate several orbits; a perihelion is a local minimum of the star→planet distance.
            for (int i = 0; i < 260000; i++)
            {
                integ.Step(state, solver, dt);
                var rel = state.PositionOf(1) - state.PositionOf(0);
                double r = rel.Length;

                // Three-point local-minimum test on the previous sample.
                if (prevR < prevPrevR && prevR < r)
                {
                    double ang = Math.Atan2(rel.Y, rel.X);   // note: uses current sample's angle as a proxy for the peri direction
                    if (!double.IsNaN(lastPeriAngle))
                    {
                        double d = ang - lastPeriAngle;
                        while (d > Math.PI) d -= 2 * Math.PI;   // wrap into (−π, π]
                        while (d < -Math.PI) d += 2 * Math.PI;
                        sumAdvance += d; nAdvances++;
                    }
                    lastPeriAngle = ang;
                }
                prevPrevR = prevR; prevR = r;
            }

            double analytic = 6 * Math.PI * G * M / (c * c * a * (1 - e * e));   // ≈ 0.2513 rad
            double measured = nAdvances > 0 ? sumAdvance / nAdvances : double.NaN;
            Check(nAdvances >= 3, $"detected {nAdvances} perihelion passages (need ≥ 3)");
            // 15% tolerance: the finite dt and the "current-sample angle" proxy add a little slack, but the
            // measured advance must clearly be the analytic value and clearly non-zero.
            bool close = nAdvances >= 3 && Math.Abs(measured - analytic) / analytic < 0.15;
            Check(close, $"precession/orbit measured {measured:F4} rad vs analytic {analytic:F4} rad (within 15%)");
        }

        /// <summary>
        /// Educational contrast that captures the ACTUAL difference between the schemes: symplectic
        /// leapfrog's energy error is BOUNDED (it oscillates with the orbital phase and never runs away),
        /// while non-symplectic RK4's energy error GROWS secularly (it accumulates a little at every close
        /// perihelion passage). Note this is NOT "leapfrog is more accurate per step" — RK4's local
        /// accuracy is higher — it is "leapfrog is stable forever, RK4 is not", which is why leapfrog is
        /// the default for long orbital runs. We measure both on an eccentric orbit and compare the energy
        /// drift in the first quarter of the run against the last quarter.
        /// </summary>
        private static void CheckIntegratorContrast()
        {
            Console.WriteLine("Integrator contrast (bounded vs secular energy error, eccentric orbit):");

            const double e = 0.6, dt = 5e-3;
            const int steps = 150000;   // ≈ 119 orbits, long enough for RK4's secular drift to show
            var (lpFirst, lpLast) = DriftProfile(IntegrationMethod.Leapfrog, e, dt, steps);
            var (rkFirst, rkLast) = DriftProfile(IntegrationMethod.Rk4, e, dt, steps);
            Console.WriteLine($"      leapfrog: first-quarter max |ΔE/E₀| = {lpFirst:E2}, last-quarter = {lpLast:E2}");
            Console.WriteLine($"      rk4:      first-quarter max |ΔE/E₀| = {rkFirst:E2}, last-quarter = {rkLast:E2}");

            // Leapfrog is symplectic: the last quarter's error is about the same size as the first — bounded.
            double lpRatio = lpLast / Math.Max(lpFirst, 1e-300);
            Check(lpRatio < 5.0, $"leapfrog energy error stays bounded (last/first = {lpRatio:F2}, expected ≲ 5)");

            // RK4 is not symplectic: the last quarter's error clearly exceeds the first — secular growth.
            Check(rkLast > rkFirst, $"RK4 energy error grows over time (last {rkLast:E2} > first {rkFirst:E2})");
        }

        /// <summary>
        /// Integrates an eccentric two-body orbit and reports the maximum relative energy drift over the
        /// first quarter and the last quarter of the run — the shape that distinguishes a bounded
        /// (symplectic) error from a growing (secular) one.
        /// </summary>
        /// <param name="method">The integration scheme.</param>
        /// <param name="e">Orbital eccentricity of the test orbit.</param>
        /// <param name="dt">Time step.</param>
        /// <param name="steps">Total steps to integrate.</param>
        /// <returns>The max |ΔE/E₀| over the first quarter and over the last quarter of the run.</returns>
        private static (double first, double last) DriftProfile(IntegrationMethod method, double e, double dt, int steps)
        {
            double G = 1, M = 1, a = 1;
            double rp = a * (1 - e);
            double vp = Math.Sqrt(G * M * (2.0 / rp - 1.0 / a));   // vis-viva speed at perihelion
            var state = NBodyState.FromArrays(
                new double[,] { { 0, 0, 0 }, { rp, 0, 0 } },
                new double[,] { { 0, 0, 0 }, { 0, vp, 0 } },
                new double[] { M, 1e-3 });
            var solver = new GravitySolver(G, 1e-4, ForceModel.Newtonian);
            var integ = new Integrator(method);

            double e0 = ConservedQuantities.KineticEnergy(state) + ConservedQuantities.PotentialEnergy(state, G, 1e-4);
            double firstMax = 0, lastMax = 0;
            for (int i = 0; i < steps; i++)
            {
                integ.Step(state, solver, dt);
                double energy = ConservedQuantities.KineticEnergy(state) + ConservedQuantities.PotentialEnergy(state, G, 1e-4);
                double rel = Math.Abs((energy - e0) / e0);
                if (i < steps / 4 && rel > firstMax) firstMax = rel;
                else if (i >= 3 * steps / 4 && rel > lastMax) lastMax = rel;
            }
            return (firstMax, lastMax);
        }
    }
}
