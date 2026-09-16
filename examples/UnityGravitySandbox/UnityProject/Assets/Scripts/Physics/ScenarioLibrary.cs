using System;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// Factory of ready-to-run <see cref="Scenario"/>s — the sandbox's "levels". Each one is authored in a
    /// clearly-stated unit system and documents the scientific phenomenon it demonstrates, from a textbook
    /// two-body orbit to a genuine reproduction of Mercury's relativistic perihelion advance. These are
    /// the source of truth for the initial conditions the verification harness also checks.
    /// </summary>
    public static class ScenarioLibrary
    {
        /// <summary>
        /// The default ordered set the game exposes on the number keys. The order is roughly increasing
        /// conceptual complexity: circular orbit, choreography, solar system, binary, accretion, relativity.
        /// </summary>
        /// <returns>A fresh array of scenarios (each call builds new instances, so callers may mutate freely).</returns>
        public static Scenario[] All() => new[]
        {
            TwoBodyCircular(),
            FigureEight(),
            SolarSystem(),
            BinaryStarWithPlanet(),
            AccretingCluster(),
            MercuryPrecession(),
        };

        /// <summary>
        /// A light body on a near-circular orbit about a much heavier one. The classic first check: the
        /// orbit closes and total energy holds. Units are dimensionless with <c>G = 1</c>, central mass 1,
        /// orbital radius 1, so the circular speed is exactly <c>√(GM/r) = 1</c>.
        /// </summary>
        /// <returns>The two-body circular scenario.</returns>
        public static Scenario TwoBodyCircular()
        {
            double G = 1, M = 1, m = 1e-3, r = 1;
            double v = Math.Sqrt(G * M / r);   // circular-orbit speed
            var state = NBodyState.FromArrays(
                new double[,] { { 0, 0, 0 }, { r, 0, 0 } },
                new double[,] { { 0, 0, 0 }, { 0, v, 0 } },
                new double[] { M, m });
            return new Scenario
            {
                Name = "Two-Body Circular Orbit",
                Description = "A satellite on a closed circular orbit — energy and angular momentum are conserved to round-off.",
                InitialState = state,
                Descriptors = new[]
                {
                    new BodyDescriptor("Primary", 1.0, 0.85, 0.30, 0.12, isStar: true),
                    new BodyDescriptor("Satellite", 0.45, 0.75, 1.0, 0.04),
                },
                G = G, Softening = 1e-3, TimeStep = 1e-3, TimeScale = 1.5,
                CameraDistance = 3.5, TrailLength = 260,
            };
        }

        /// <summary>
        /// The Chenciner–Montgomery figure-eight: three EQUAL masses that chase each other forever around a
        /// single figure-eight curve. A celebrated exact periodic solution of the three-body problem and a
        /// stringent correctness test — a wrong force law or a drifting integrator destroys the pattern
        /// within a period. Units are dimensionless (<c>G = 1</c>, all masses 1); the period is ≈ 6.3259.
        /// </summary>
        /// <returns>The figure-eight choreography scenario.</returns>
        public static Scenario FigureEight()
        {
            // Standard initial conditions (Chenciner & Montgomery, 2000). Bodies 1 and 2 are mirror images;
            // body 3 starts at the centre moving to balance total momentum to zero.
            double x = 0.97000436, y = 0.24308753;
            double vx = 0.93240737, vy = 0.86473146;
            var state = NBodyState.FromArrays(
                new double[,] { { x, -y, 0 }, { -x, y, 0 }, { 0, 0, 0 } },
                new double[,] { { vx / 2, vy / 2, 0 }, { vx / 2, vy / 2, 0 }, { -vx, -vy, 0 } },
                new double[] { 1, 1, 1 });
            return new Scenario
            {
                Name = "Three-Body Figure-Eight",
                Description = "Three equal masses tracing one figure-eight (Chenciner–Montgomery choreography).",
                InitialState = state,
                Descriptors = new[]
                {
                    new BodyDescriptor("Alpha", 1.0, 0.42, 0.42, 0.06),
                    new BodyDescriptor("Beta",  0.42, 1.0, 0.53, 0.06),
                    new BodyDescriptor("Gamma", 0.47, 0.62, 1.0, 0.06),
                },
                G = 1, Softening = 1e-4, TimeStep = 5e-4, TimeScale = 1.0,
                CameraDistance = 3.2, TrailLength = 400,
            };
        }

        /// <summary>
        /// The inner + outer solar system on circular orbits, in the astronomer's unit system: distances in
        /// AU, masses in solar masses, time in years, so <c>G = 4π²</c> (which makes Earth's circular speed
        /// exactly <c>2π</c> AU/yr and its period exactly one year). Relative planet masses are real, so
        /// Jupiter measurably perturbs the system over long runs.
        /// </summary>
        /// <returns>The solar-system scenario.</returns>
        public static Scenario SolarSystem()
        {
            double G = 4 * Math.PI * Math.PI;   // AU³ · M_sun⁻¹ · yr⁻²
            // (name, semi-major axis AU, mass M_sun, colour rgb, display radius AU)
            var planets = new (string name, double a, double mass, double r, double g, double b, double rad)[]
            {
                ("Mercury", 0.387, 1.66e-7, 0.70, 0.70, 0.70, 0.015),
                ("Venus",   0.723, 2.45e-6, 0.95, 0.78, 0.45, 0.022),
                ("Earth",   1.000, 3.00e-6, 0.35, 0.55, 1.00, 0.024),
                ("Mars",    1.524, 3.20e-7, 0.90, 0.42, 0.28, 0.018),
                ("Jupiter", 5.203, 9.54e-4, 0.85, 0.68, 0.45, 0.09),
                ("Saturn",  9.537, 2.86e-4, 0.90, 0.82, 0.55, 0.08),
            };

            int n = planets.Length + 1;   // + the Sun
            var pos = new double[n, 3];
            var vel = new double[n, 3];
            var mass = new double[n];
            var descs = new BodyDescriptor[n];

            // Body 0 is the Sun at the origin, momentarily at rest; RecenterMomentum below gives it the tiny
            // recoil that keeps the whole system's centre of mass fixed for a steady camera.
            mass[0] = 1.0;
            descs[0] = new BodyDescriptor("Sun", 1.0, 0.86, 0.30, 0.16, isStar: true);

            for (int i = 0; i < planets.Length; i++)
            {
                var p = planets[i];
                int k = i + 1;
                double speed = Math.Sqrt(G * mass[0] / p.a);   // circular-orbit speed at radius a
                pos[k, 0] = p.a;                                // start on the +x axis
                vel[k, 1] = speed;                             // moving +y → counter-clockwise orbit
                mass[k] = p.mass;
                descs[k] = new BodyDescriptor(p.name, p.r, p.g, p.b, p.rad);
            }

            var state = new NBodyState(np.array(pos), np.array(vel), np.array(mass));
            RecenterMomentum(state);   // zero total momentum so the barycentre does not drift off-screen
            return new Scenario
            {
                Name = "Solar System",
                Description = "Sun + six planets on circular orbits (AU / M_sun / year units, G = 4π²).",
                InitialState = state,
                Descriptors = descs,
                G = G, Softening = 1e-3, TimeStep = 2e-3, TimeScale = 2.0,
                CameraDistance = 24.0, TrailLength = 300,
            };
        }

        /// <summary>
        /// A close binary star with a distant circumbinary ("Tatooine") planet. The planet orbits the pair's
        /// combined mass while the stars whirl about their common centre; it demonstrates a hierarchical
        /// system and stable motion around a time-varying two-source potential. Dimensionless units,
        /// <c>G = 1</c>.
        /// </summary>
        /// <returns>The binary-star scenario.</returns>
        public static Scenario BinaryStarWithPlanet()
        {
            double G = 1, Mstar = 1, d = 2;                 // two equal stars separated by d
            double vStar = Math.Sqrt(G * Mstar / (2 * d));  // circular mutual-orbit speed for equal masses
            double rP = 5.0, mP = 1e-3;
            double vP = Math.Sqrt(G * (2 * Mstar) / rP);   // planet ~ circular about the combined stellar mass

            var state = NBodyState.FromArrays(
                new double[,] { { -1, 0, 0 }, { 1, 0, 0 }, { rP, 0, 0 } },
                new double[,] { { 0, -vStar, 0 }, { 0, vStar, 0 }, { 0, vP, 0 } },
                new double[] { Mstar, Mstar, mP });
            RecenterMomentum(state);
            return new Scenario
            {
                Name = "Binary Star + Planet",
                Description = "A close binary with a distant circumbinary planet orbiting the pair.",
                InitialState = state,
                Descriptors = new[]
                {
                    new BodyDescriptor("Star A", 1.0, 0.80, 0.35, 0.16, isStar: true),
                    new BodyDescriptor("Star B", 0.75, 0.85, 1.0, 0.14, isStar: true),
                    new BodyDescriptor("Planet", 0.55, 0.90, 0.65, 0.05),
                },
                G = G, Softening = 5e-3, TimeStep = 1e-3, TimeScale = 3.0,
                CameraDistance = 14.0, TrailLength = 320,
            };
        }

        /// <summary>
        /// A cloud of small bodies in a slowly rotating disk with collisions ENABLED — over time gravity and
        /// perfectly-inelastic merging pull them into a few larger bodies, a toy model of planetesimal
        /// accretion. Uses a FIXED random seed so the run is reproducible. Dimensionless units, <c>G = 1</c>.
        /// </summary>
        /// <param name="count">Number of initial bodies (kept modest for the O(N²) solver).</param>
        /// <param name="seed">Seed for the deterministic layout, so the same demo replays identically.</param>
        /// <returns>The accreting-cluster scenario.</returns>
        public static Scenario AccretingCluster(int count = 60, int seed = 12345)
        {
            var rng = new Random(seed);   // managed RNG → deterministic, and independent of np.random parity
            double G = 1, R = 6, mBody = 0.02;
            double spin = 0.25;           // gentle solid-body rotation so the cloud shears into a disk

            var pos = new double[count, 3];
            var vel = new double[count, 3];
            var mass = new double[count];
            var descs = new BodyDescriptor[count];

            for (int i = 0; i < count; i++)
            {
                // Uniform-in-area sample of a disk: radius ∝ √u so bodies aren't bunched at the centre.
                double ang = rng.NextDouble() * 2 * Math.PI;
                double rad = R * Math.Sqrt(rng.NextDouble());
                double px = rad * Math.Cos(ang), py = rad * Math.Sin(ang);
                pos[i, 0] = px; pos[i, 1] = py; pos[i, 2] = 0;

                // Tangential velocity for rotation, plus a little isotropic scatter to seed clumping.
                vel[i, 0] = -spin * py + (rng.NextDouble() - 0.5) * 0.05;
                vel[i, 1] = spin * px + (rng.NextDouble() - 0.5) * 0.05;

                mass[i] = mBody;
                // Radius from mass at constant density; here all equal, but merges will grow survivors.
                double disp = 0.06;
                double shade = 0.5 + 0.5 * rng.NextDouble();
                descs[i] = new BodyDescriptor($"P{i}", shade, shade * 0.7, shade * 0.5, disp);
            }

            var state = new NBodyState(np.array(pos), np.array(vel), np.array(mass));
            RecenterMomentum(state);
            return new Scenario
            {
                Name = "Accreting Cluster",
                Description = "A rotating cloud that gravitationally clumps and merges — toy planetesimal accretion.",
                InitialState = state,
                Descriptors = descs,
                G = G, Softening = 0.08, TimeStep = 2e-3, TimeScale = 1.5,
                EnableCollisions = true,
                CameraDistance = 18.0, TrailLength = 0,   // trails off: too many bodies to be legible
            };
        }

        /// <summary>
        /// A single planet on an eccentric orbit under the <see cref="ForceModel.RelativisticPrecession"/>
        /// force law, with the speed of light dialed DOWN so the perihelion advance is visible in seconds
        /// rather than the 43″/century of the real Mercury. Watch the ellipse's long axis rotate — the
        /// signature of general relativity that Newtonian gravity cannot produce. Dimensionless units,
        /// <c>G = 1</c>, <c>M = 1</c>, semi-major axis 1, eccentricity 0.5.
        /// </summary>
        /// <returns>The relativistic-precession scenario.</returns>
        public static Scenario MercuryPrecession()
        {
            double G = 1, M = 1, a = 1, e = 0.5;
            double rp = a * (1 - e);                       // perihelion distance
            double vp = Math.Sqrt(G * M * (2 / rp - 1 / a)); // vis-viva speed at perihelion (perpendicular)
            double c = 10.0;   // small light-speed → large, visible precession (~14°/orbit; see PHYSICS.md)

            var state = NBodyState.FromArrays(
                new double[,] { { 0, 0, 0 }, { rp, 0, 0 } },
                new double[,] { { 0, 0, 0 }, { 0, vp, 0 } },
                new double[] { M, 1e-4 });
            return new Scenario
            {
                Name = "Relativistic Precession",
                Description = "An eccentric orbit whose perihelion advances — the 1PN correction, Mercury's 43″/century writ large.",
                InitialState = state,
                Descriptors = new[]
                {
                    new BodyDescriptor("Star", 1.0, 0.80, 0.30, 0.10, isStar: true),
                    new BodyDescriptor("Planet", 0.75, 0.55, 1.0, 0.035),
                },
                G = G, Softening = 1e-4, TimeStep = 1e-4, TimeScale = 0.5,
                Model = ForceModel.RelativisticPrecession, SpeedOfLight = c,
                CameraDistance = 3.2, TrailLength = 900,   // long trail so the rotating ellipse is unmistakable
            };
        }

        /// <summary>
        /// Shifts a state into its centre-of-mass frame: subtracts the barycentre position and the mean
        /// velocity from every body, so the system's total momentum is zero and its centre of mass sits at
        /// the origin. Purely a framing convenience — it changes the inertial frame, not the dynamics — but
        /// it keeps the action centred for the camera instead of drifting off-screen.
        /// </summary>
        /// <param name="state">The state to recenter, mutated in place.</param>
        public static void RecenterMomentum(NBodyState state)
        {
            var com = ConservedQuantities.CenterOfMass(state, out double totalMass);
            var p = ConservedQuantities.LinearMomentum(state);
            double inv = totalMass != 0 ? 1.0 / totalMass : 0.0;
            var comVel = new Vector3d(p.X * inv, p.Y * inv, p.Z * inv);

            // Broadcast-subtract the (3,) offsets from every (N,3) row.
            state.Positions = state.Positions - np.array(new double[] { com.X, com.Y, com.Z });
            state.Velocities = state.Velocities - np.array(new double[] { comVel.X, comVel.Y, comVel.Z });
        }
    }
}
