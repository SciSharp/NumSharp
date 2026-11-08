using System;
using System.Collections.Generic;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// The complete, self-contained simulation the game drives: it owns the dynamical
    /// <see cref="NBodyState"/>, the <see cref="GravitySolver"/> (force law) and the
    /// <see cref="Integrator"/> (time stepping), keeps the presentation <see cref="Descriptors"/> aligned
    /// with the bodies, tracks simulated time, and handles the structural edits a sandbox needs —
    /// spawning bodies and merging them on collision. It is the ONE type the Unity layer needs to know;
    /// everything below it is pure numerics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Topology versioning.</b> The body set can change size (spawn, merge) between frames. Rather than
    /// raising events, the world bumps <see cref="Version"/> on every structural change; the view compares
    /// the version it last drew against the current one and rebuilds its GameObjects when they differ.
    /// This keeps the view a pure function of world state and avoids event-ordering bugs.
    /// </para>
    /// <para>
    /// <b>Not thread-safe.</b> A world is meant to be stepped and read from one thread (Unity's main
    /// thread in the sample). The physics arrays are reassigned during a step, so reading them
    /// concurrently is undefined.
    /// </para>
    /// </remarks>
    public sealed class NBodyWorld
    {
        private readonly List<BodyDescriptor> _descriptors = new List<BodyDescriptor>();
        private Scenario _scenario;

        /// <summary>The current dynamical state (positions/velocities/masses). Reassigned each step.</summary>
        public NBodyState State { get; private set; }

        /// <summary>The force model in force. Its fields (G, softening, model) may be tuned live.</summary>
        public GravitySolver Solver { get; private set; }

        /// <summary>The time-stepping scheme. Its <see cref="Integrator.Method"/> may be switched live.</summary>
        public Integrator Integrator { get; private set; }

        /// <summary>Presentation data for each body, aligned with the rows of <see cref="State"/>.</summary>
        public IReadOnlyList<BodyDescriptor> Descriptors => _descriptors;

        /// <summary>Total elapsed simulated time in the scenario's time unit (not wall-clock).</summary>
        public double Time { get; private set; }

        /// <summary>The number of integrator steps taken since the scenario was loaded.</summary>
        public long StepCount { get; private set; }

        /// <summary>
        /// A counter bumped on every structural change (load, reset, spawn, merge). The view watches it to
        /// know when to rebuild; a caller can also use it to detect that indices may have shifted.
        /// </summary>
        public int Version { get; private set; }

        /// <summary>The fixed physics step size, taken from the loaded scenario.</summary>
        public double TimeStep { get; private set; }

        /// <summary>Simulated seconds elapsed per real second at 1× speed (from the scenario).</summary>
        public double TimeScale { get; set; }

        /// <summary>Whether the collision/merge pass runs after each step (from the scenario, toggleable).</summary>
        public bool EnableCollisions { get; set; }

        /// <summary>The scenario currently loaded (used by <see cref="Reset"/> to restore initial conditions).</summary>
        public Scenario CurrentScenario => _scenario;

        /// <summary>Creates a world and immediately loads a scenario into it.</summary>
        /// <param name="scenario">The scenario to run. Its initial state is cloned, so the scenario object is not mutated.</param>
        /// <exception cref="ArgumentNullException"><paramref name="scenario"/> or its initial state is null.</exception>
        public NBodyWorld(Scenario scenario)
        {
            LoadScenario(scenario);
        }

        /// <summary>
        /// Replaces the running simulation with a fresh copy of <paramref name="scenario"/>, rebuilding the
        /// solver and integrator from its configuration and resetting time. Cloning the initial state is
        /// what makes <see cref="Reset"/> able to return here exactly.
        /// </summary>
        /// <param name="scenario">The scenario to load.</param>
        /// <exception cref="ArgumentNullException"><paramref name="scenario"/> or its <see cref="Scenario.InitialState"/> is null.</exception>
        /// <exception cref="ArgumentException">The scenario's descriptor count does not match its body count.</exception>
        public void LoadScenario(Scenario scenario)
        {
            _scenario = scenario ?? throw new ArgumentNullException(nameof(scenario));
            if (scenario.InitialState is null) throw new ArgumentNullException(nameof(scenario), "Scenario.InitialState is null.");
            if (scenario.Descriptors.Length != scenario.InitialState.Count)
                throw new ArgumentException($"Scenario '{scenario.Name}' has {scenario.Descriptors.Length} descriptors for {scenario.InitialState.Count} bodies.");

            Solver = new GravitySolver(scenario.G, scenario.Softening, scenario.Model, scenario.SpeedOfLight);
            Integrator = new Integrator(scenario.Method);
            TimeStep = scenario.TimeStep;
            TimeScale = scenario.TimeScale;
            EnableCollisions = scenario.EnableCollisions;
            Reset();
        }

        /// <summary>
        /// Restores the current scenario's initial conditions — a fresh clone of its state and descriptors
        /// — and zeroes elapsed time and step count. Also resets the integrator cache, because the cached
        /// acceleration belonged to the old (now discarded) configuration.
        /// </summary>
        /// <exception cref="InvalidOperationException">No scenario has been loaded.</exception>
        public void Reset()
        {
            if (_scenario is null) throw new InvalidOperationException("No scenario loaded.");
            State = _scenario.InitialState.Clone();          // deep copy so future steps don't touch the recipe
            _descriptors.Clear();
            _descriptors.AddRange(_scenario.Descriptors);    // structs: AddRange copies by value
            Time = 0.0;
            StepCount = 0;
            Integrator.Reset();
            Version++;
        }

        /// <summary>
        /// Advances the simulation by exactly one physics step of <see cref="TimeStep"/>, then runs the
        /// collision pass if enabled. Prefer <see cref="Advance"/> from a variable-rate render loop; use
        /// this directly only when you want deterministic single stepping.
        /// </summary>
        public void Step()
        {
            Integrator.Step(State, Solver, TimeStep);
            Time += TimeStep;
            StepCount++;
            if (EnableCollisions)
                CollisionPass();
        }

        /// <summary>
        /// Advances the simulation to cover <paramref name="simulatedSeconds"/> of simulated time using
        /// fixed <see cref="TimeStep"/> substeps, capped at <paramref name="maxSubsteps"/> to protect the
        /// frame rate. Fixed substeps (rather than one variable-sized step per frame) keep the integration
        /// stable and frame-rate independent; the cap prevents the "spiral of death" where a slow frame
        /// asks for more steps than the next frame can afford.
        /// </summary>
        /// <param name="simulatedSeconds">How much simulated time to advance (already scaled by the caller if desired).</param>
        /// <param name="maxSubsteps">The hard ceiling on substeps this call may take. Excess simulated time is dropped.</param>
        /// <returns>The number of substeps actually taken.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxSubsteps"/> is negative.</exception>
        public int Advance(double simulatedSeconds, int maxSubsteps = 64)
        {
            if (maxSubsteps < 0) throw new ArgumentOutOfRangeException(nameof(maxSubsteps));
            if (!(simulatedSeconds > 0) || TimeStep <= 0) return 0;   // nothing to do (also guards NaN)

            int want = (int)(simulatedSeconds / TimeStep);
            int steps = Math.Min(want, maxSubsteps);
            for (int i = 0; i < steps; i++)
                Step();
            return steps;
        }

        /// <summary>Computes the system's conserved-quantity snapshot right now (see <see cref="ConservedQuantities"/>).</summary>
        /// <returns>A diagnostics snapshot for the current state.</returns>
        public SystemDiagnostics Diagnostics() => ConservedQuantities.Compute(State, Solver);

        /// <summary>
        /// Spawns a new body, appending it as the last row of every physics array and its descriptor.
        /// Appending (rather than inserting) keeps every existing index stable; the <see cref="Version"/>
        /// bump still tells the view a body was added.
        /// </summary>
        /// <param name="position">Initial position in scenario units.</param>
        /// <param name="velocity">Initial velocity in scenario units.</param>
        /// <param name="mass">Mass in scenario units; must be positive and finite.</param>
        /// <param name="descriptor">Presentation data for the new body.</param>
        /// <returns>The index of the newly added body (always <c>Count-1</c>).</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mass"/> is not a positive, finite number.</exception>
        public int AddBody(Vector3d position, Vector3d velocity, double mass, BodyDescriptor descriptor)
        {
            if (!(mass > 0) || !double.IsFinite(mass)) throw new ArgumentOutOfRangeException(nameof(mass), "Mass must be positive and finite.");

            // Append one row to each (N,·) array. np.concatenate over axis 0 keeps existing rows intact,
            // so every current body index is preserved; only a new last index appears.
            var posRow = np.array(new double[,] { { position.X, position.Y, position.Z } });
            var velRow = np.array(new double[,] { { velocity.X, velocity.Y, velocity.Z } });
            var massRow = np.array(new double[] { mass });

            State.Positions = np.concatenate((State.Positions, posRow), axis: 0);
            State.Velocities = np.concatenate((State.Velocities, velRow), axis: 0);
            State.Masses = np.concatenate((State.Masses, massRow), axis: 0);
            _descriptors.Add(descriptor);

            // The added body changes the acceleration field, so any cached acceleration is stale.
            Integrator.Reset();
            Version++;
            return State.Count - 1;
        }

        /// <summary>
        /// Detects touching bodies and merges each touching group into a single body, conserving total
        /// mass and linear momentum (a perfectly inelastic collision — the physics of accretion). Two
        /// bodies "touch" when their centres are within the sum of their <see cref="BodyDescriptor.DisplayRadius"/>.
        /// Runs only when <see cref="EnableCollisions"/> is set.
        /// </summary>
        /// <returns>True if at least one merge happened (and indices therefore changed); otherwise false.</returns>
        private bool CollisionPass()
        {
            int n = State.Count;
            if (n < 2) return false;

            // Pairwise squared separations in one vectorized pass, then pulled into a flat managed buffer
            // once (a single bulk copy) so the O(N²) touch test below is plain array indexing rather than
            // N² NumSharp scalar reads.
            var ri = State.Positions.reshape(n, 1, 3);
            var rj = State.Positions.reshape(1, n, 3);
            var disp = rj - ri;
            var r2mat = np.sum(disp * disp, axis: 2);        // (N,N)
            double[] r2 = r2mat.ToArray<double>();           // row-major length N*N

            // Union-find over bodies: unite every touching pair, so a chain of contacts collapses into one
            // merged body in a single pass (avoids order-dependent pairwise merging).
            int[] parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;

            bool any = false;
            for (int i = 0; i < n; i++)
            {
                double radI = _descriptors[i].DisplayRadius;
                for (int j = i + 1; j < n; j++)
                {
                    double touch = radI + _descriptors[j].DisplayRadius;
                    if (r2[i * n + j] <= touch * touch)
                    {
                        Union(parent, i, j);
                        any = true;
                    }
                }
            }
            if (!any) return false;

            MergeGroups(parent, n);
            Integrator.Reset();   // topology changed underneath the cached acceleration
            Version++;
            return true;
        }

        /// <summary>Union-find "find" with path compression.</summary>
        /// <param name="parent">The parent array.</param>
        /// <param name="i">The element whose set representative is wanted.</param>
        /// <returns>The representative (root) of <paramref name="i"/>'s set.</returns>
        private static int Find(int[] parent, int i)
        {
            while (parent[i] != i)
            {
                parent[i] = parent[parent[i]];   // halve the path so repeated finds stay near-constant time
                i = parent[i];
            }
            return i;
        }

        /// <summary>Union-find "union" of the sets containing <paramref name="a"/> and <paramref name="b"/>.</summary>
        /// <param name="parent">The parent array.</param>
        /// <param name="a">First element.</param>
        /// <param name="b">Second element.</param>
        private static void Union(int[] parent, int a, int b)
        {
            int ra = Find(parent, a), rb = Find(parent, b);
            if (ra != rb) parent[ra] = rb;
        }

        /// <summary>
        /// Collapses each union-find group into one body and rebuilds the state. The merged body sits at
        /// the group's centre of mass, carries the summed momentum (so velocity is the mass-weighted mean),
        /// takes the appearance of its most massive member, and grows by volume
        /// (<c>r = (Σ rₖ³)^{1/3}</c>, i.e. constant density) so a merge looks like accretion, not a jump.
        /// </summary>
        /// <param name="parent">The resolved union-find forest.</param>
        /// <param name="n">The pre-merge body count.</param>
        private void MergeGroups(int[] parent, int n)
        {
            // Bucket original indices by their group root, preserving order so results are deterministic.
            var groups = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int root = Find(parent, i);
                if (!groups.TryGetValue(root, out var list)) { list = new List<int>(); groups[root] = list; }
                list.Add(i);
            }

            int m = groups.Count;
            var pos = new double[m, 3];
            var vel = new double[m, 3];
            var mass = new double[m];
            var descs = new BodyDescriptor[m];

            int outIdx = 0;
            foreach (var kv in groups)
            {
                var members = kv.Value;
                if (members.Count == 1)
                {
                    // Untouched body: copy through unchanged.
                    int i = members[0];
                    var p = State.PositionOf(i); var v = State.VelocityOf(i);
                    pos[outIdx, 0] = p.X; pos[outIdx, 1] = p.Y; pos[outIdx, 2] = p.Z;
                    vel[outIdx, 0] = v.X; vel[outIdx, 1] = v.Y; vel[outIdx, 2] = v.Z;
                    mass[outIdx] = State.MassOf(i);
                    descs[outIdx] = _descriptors[i];
                }
                else
                {
                    // Merge: accumulate mass, momentum (m·v) and first moment (m·r); the merged body is the
                    // centre of mass moving with the group's total momentum — conserving both exactly.
                    double mSum = 0, px = 0, py = 0, pz = 0, cx = 0, cy = 0, cz = 0, r3 = 0;
                    int heaviest = members[0]; double heaviestMass = -1;
                    bool anyStar = false;
                    foreach (int i in members)
                    {
                        double mi = State.MassOf(i);
                        var p = State.PositionOf(i); var v = State.VelocityOf(i);
                        mSum += mi;
                        px += mi * v.X; py += mi * v.Y; pz += mi * v.Z;
                        cx += mi * p.X; cy += mi * p.Y; cz += mi * p.Z;
                        double rd = _descriptors[i].DisplayRadius; r3 += rd * rd * rd;
                        anyStar |= _descriptors[i].IsStar;
                        if (mi > heaviestMass) { heaviestMass = mi; heaviest = i; }
                    }
                    double inv = 1.0 / mSum;
                    pos[outIdx, 0] = cx * inv; pos[outIdx, 1] = cy * inv; pos[outIdx, 2] = cz * inv;
                    vel[outIdx, 0] = px * inv; vel[outIdx, 1] = py * inv; vel[outIdx, 2] = pz * inv;
                    mass[outIdx] = mSum;

                    var d = _descriptors[heaviest];      // inherit colour/name/star from the dominant member
                    d.DisplayRadius = Math.Cbrt(r3);     // grow by volume, keeping density roughly constant
                    d.IsStar = anyStar;
                    descs[outIdx] = d;
                }
                outIdx++;
            }

            // Swap in the rebuilt, smaller system. The ctor re-validates the shape contract.
            State = new NBodyState(np.array(pos), np.array(vel), np.array(mass));
            _descriptors.Clear();
            _descriptors.AddRange(descs);
        }
    }
}
