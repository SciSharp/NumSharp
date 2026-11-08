using System;
using System.Collections.Generic;
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends.Iteration;
using static NumSharp.Backends.Iteration.NDExpr;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>A faucet: continuously spawns one material at a point with an initial velocity.</summary>
    public sealed class Emitter
    {
        /// <summary>Material emitted.</summary>
        public MaterialId Material { get; init; }
        /// <summary>Nozzle center x (units).</summary>
        public float X { get; init; }
        /// <summary>Nozzle center y (units).</summary>
        public float Y { get; init; }
        /// <summary>Nozzle radius (units).</summary>
        public float Radius { get; init; } = 0.02f;
        /// <summary>Exit velocity x.</summary>
        public float VX { get; init; }
        /// <summary>Exit velocity y.</summary>
        public float VY { get; init; }
        /// <summary>Stop after this many particles (0 = never).</summary>
        public int Budget { get; init; }
        /// <summary>Particles emitted so far.</summary>
        public int Emitted { get; internal set; }
    }

    /// <summary>Timing and state of the most recent frame (for the HUD and the benchmark).</summary>
    public struct FrameStats
    {
        /// <summary>Substeps run.</summary>
        public int Substeps;
        /// <summary>Substep Δt (s).</summary>
        public float Dt;
        /// <summary>Milliseconds in the NumSharp constitutive kernels.</summary>
        public double ConstitutiveMs;
        /// <summary>Milliseconds in particle→grid.</summary>
        public double P2GMs;
        /// <summary>Milliseconds in the NumSharp grid update.</summary>
        public double GridMs;
        /// <summary>Milliseconds in grid→particle.</summary>
        public double G2PMs;
        /// <summary>Milliseconds in everything else (emitters, cooling, sorting).</summary>
        public double OtherMs;
        /// <summary>Total simulation milliseconds this frame.</summary>
        public double TotalMs;
        /// <summary>Particles alive.</summary>
        public int Particles;
        /// <summary>Fastest particle (units/s).</summary>
        public float MaxSpeed;
    }

    /// <summary>
    /// The whole simulated world: the MLS-MPM solver, its particles (one <see cref="ParticleGroup"/> per
    /// material), the grid and obstacles, faucets, and the interaction entry points the front-end calls.
    /// Unity-/UI-independent — the verification harness drives exactly this class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Time.</b> Each rendered frame advances a fixed slice of simulated time (<see cref="FrameTime"/>),
    /// split into as many substeps as stability requires: Δt = min(CFL·Δx / fastest elastic wave of the
    /// materials present, ½·Δx / fastest particle). Water-only scenes therefore run fewer, longer substeps
    /// than scenes with stiff rubber; and if the machine cannot keep up, the frame rate drops but the
    /// physics stays exactly as stable.
    /// </para>
    /// <para>
    /// <b>Substep.</b> clear grid → per material: NumSharp constitutive kernels (stress, plasticity) then
    /// particle→grid scatter → NumSharp grid update (velocity, gravity, stir, collisions, friction) → per
    /// material: grid→particle gather + advection.
    /// </para>
    /// </remarks>
    public sealed unsafe class MpmWorld : IDisposable
    {
        /// <summary>One particle group per material, index = <see cref="MaterialId"/>.</summary>
        private readonly ParticleGroup[] _groups;
        /// <summary>Each group's compiled constitutive stages.</summary>
        private readonly ConstitutiveKernels[] _kernels;
        /// <summary>Particle mass per material (ρ·V₀).</summary>
        private readonly float[] _mass;
        /// <summary>Whether a material counts as coolant for lava (water only).</summary>
        private readonly bool[] _coolant;
        /// <summary>V₀ = Δx²/4, the volume of every particle.</summary>
        private readonly float _volume;   // V₀ of every particle (2×2 per cell)
        /// <summary>Jitter for filled particles (seeded: scenes are reproducible).</summary>
        private readonly Random _rng;
        /// <summary>Fastest particle of the previous frame (the half-cell time-step bound).</summary>
        private float _lastMaxSpeed;
        /// <summary>Frames stepped (drives the periodic spatial sort).</summary>
        private int _frame;
        /// <summary>Per-quarter-cell particle counts for <see cref="Fill"/> (a reused buffer).</summary>
        private int[] _slotCount;         // per-quarter-cell particle counts for Fill (reused buffer)

        /// <summary>The grid and obstacle field.</summary>
        public MpmGrid Grid { get; }
        /// <summary>Runtime scalars shared by the NumSharp kernels (Δt, stress scale, viscosities).</summary>
        public SolverParams Params { get; } = new SolverParams();
        /// <summary>The NumSharp grid-update kernels (gravity, stir, collisions).</summary>
        public GridUpdate GridStep { get; }
        /// <summary>Which transfer implementation runs (switchable live).</summary>
        public TransferMode Transfers { get; set; } = TransferMode.Fused;
        /// <summary>Simulated seconds advanced per rendered frame.</summary>
        public float FrameTime { get; set; } = 2.8e-3f;
        /// <summary>Wave-speed CFL number (≤ 1 for explicit MLS-MPM).</summary>
        public float Cfl { get; set; } = 0.85f;
        /// <summary>Upper bound on substeps per frame (beyond it the simulation slows down rather than destabilizing).</summary>
        public int MaxSubsteps { get; set; } = 36;
        /// <summary>Gravity magnitude (units/s²; the domain is 1 unit tall).</summary>
        public float Gravity { get; set; } = 110f;
        /// <summary>Gravity direction in radians from straight down (positive = tilted to the right).</summary>
        public float GravityAngle { get; set; }
        /// <summary>Faucets currently running.</summary>
        public List<Emitter> Emitters { get; } = new List<Emitter>();
        /// <summary>Statistics of the most recent <see cref="StepFrame"/>.</summary>
        public FrameStats Stats { get; private set; }
        /// <summary>
        /// Lava cooling per simulated second in air (full cooling ≈ 1/CoolRate s). Slow on purpose: air is a poor
        /// coolant, so a flow stays molten long enough to run down a slope (~2.5 s here) while crusting — its
        /// viscosity climbs as it cools — and it is water (<see cref="LavaQuenchRate"/>) that freezes it fast.
        /// </summary>
        public float LavaCoolRate { get; set; } = 0.35f;
        /// <summary>Extra lava cooling per simulated second when fully surrounded by water.</summary>
        public float LavaQuenchRate { get; set; } = 60f;
        /// <summary>Temperature below which lava solidifies into rock.</summary>
        public float LavaSolidifyAt { get; set; } = 0.12f;

        /// <summary>Creates an empty tank.</summary>
        /// <param name="ny">Grid rows (the domain is 1 unit tall, Δx = 1/ny).</param>
        /// <param name="aspect">Width/height of the domain (16/9 matches a widescreen window).</param>
        /// <param name="seed">Random seed (particle jitter, color seeds).</param>
        public MpmWorld(int ny, float aspect, int seed = 1)
        {
            int nx = Math.Max(16, (int)MathF.Round(ny * aspect));
            Grid = new MpmGrid(nx, ny);
            GridStep = new GridUpdate(Grid, Params.Dt);
            _rng = new Random(seed);
            int count = Materials.Count;
            _groups = new ParticleGroup[count];
            _kernels = new ConstitutiveKernels[count];
            _mass = new float[count];
            _coolant = new bool[count];
            float pVol = 0.25f * Grid.Dx * Grid.Dx;   // 2×2 particles per cell
            _volume = pVol;
            for (int i = 0; i < count; i++)
            {
                var m = Materials.All[i];
                _groups[i] = new ParticleGroup(m, seed);
                _mass[i] = pVol * m.Density;
                _coolant[i] = m.Id == MaterialId.Water;
                _kernels[i] = new ConstitutiveKernels(_groups[i], Params, _mass[i]);
            }
        }

        /// <summary>The particle group of a material.</summary>
        /// <param name="id">Material.</param>
        /// <returns>Its group.</returns>
        public ParticleGroup Group(MaterialId id) => _groups[(int)id];

        /// <summary>All groups, index = <see cref="MaterialId"/>.</summary>
        public IReadOnlyList<ParticleGroup> Groups => _groups;

        /// <summary>Mass of one particle of a material.</summary>
        /// <param name="id">Material.</param>
        /// <returns>V₀·ρ.</returns>
        public float ParticleMass(MaterialId id) => _mass[(int)id];

        /// <summary>Total live particles.</summary>
        public int ParticleCount
        {
            get { int n = 0; foreach (var g in _groups) n += g.Count; return n; }
        }

        /// <summary>
        /// The substep Δt the materials currently present allow: CFL·Δx over the fastest elastic wave, and at most
        /// half a cell per substep for the fastest particle of the previous frame.
        /// </summary>
        /// <returns>The stable Δt in seconds.</returns>
        public float StableDt()
        {
            float c = 20f;
            foreach (var g in _groups)
                if (g.Count > 0) c = MathF.Max(c, g.Material.WaveSpeed);
            float dt = Cfl * Grid.Dx / c;
            if (_lastMaxSpeed > 1e-3f) dt = MathF.Min(dt, 0.5f * Grid.Dx / _lastMaxSpeed);
            return dt;
        }

        /// <summary>
        /// Advances one rendered frame: runs the faucets, the substeps, lava cooling/solidification and the
        /// periodic spatial sort, and records <see cref="Stats"/>.
        /// </summary>
        public void StepFrame()
        {
            long t0 = Stopwatch.GetTimestamp();
            var st = new FrameStats();
            RunEmitters();
            float dtStable = StableDt();
            int substeps = Math.Clamp((int)MathF.Ceiling(FrameTime / dtStable), 1, MaxSubsteps);
            float dt = MathF.Min(FrameTime / substeps, dtStable);
            Params.Update(dt, Grid.Dx);
            GridStep.GravityX.SetValue(Gravity * MathF.Sin(GravityAngle));
            GridStep.GravityY.SetValue(-Gravity * MathF.Cos(GravityAngle));
            long tOther = Stopwatch.GetTimestamp() - t0;

            float gx = Gravity * MathF.Sin(GravityAngle), gy = -Gravity * MathF.Cos(GravityAngle);
            var ap = new AdvectParams(dt, 0.45f * Grid.Dx / dt, false, LavaCoolRate, LavaQuenchRate, gx, gy,
                InvCellArea: 1f / (Grid.Dx * Grid.Dx));
            float maxSpeed = 0f;
            long tc = 0, tp = 0, tg = 0, tq = 0;
            for (int s = 0; s < substeps; s++)
            {
                Grid.ClearNodes();
                if (Transfers == TransferMode.Fused)
                {
                    for (int i = 0; i < _groups.Length; i++)
                    {
                        if (_groups[i].Count == 0) continue;
                        long c0 = Stopwatch.GetTimestamp();
                        _kernels[i].Apply();
                        long c1 = Stopwatch.GetTimestamp();
                        Transfers_P2G(i);
                        long c2 = Stopwatch.GetTimestamp();
                        tc += c1 - c0; tp += c2 - c1;
                    }
                }
                else
                {
                    long c0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < _groups.Length; i++) _kernels[i].Apply();
                    long c1 = Stopwatch.GetTimestamp();
                    Simulation.Transfers.P2GReference(Grid, _groups, _mass, _volume, _coolant);
                    long c2 = Stopwatch.GetTimestamp();
                    tc += c1 - c0; tp += c2 - c1;
                }
                long b = Stopwatch.GetTimestamp();
                GridStep.Apply();
                long c = Stopwatch.GetTimestamp();
                tg += c - b;
                maxSpeed = 0f;
                for (int i = 0; i < _groups.Length; i++)
                {
                    var g = _groups[i];
                    if (g.Count == 0) continue;
                    var p = ap with
                    {
                        IsLava = g.Material.Id == MaterialId.Lava,
                        Mobility = g.Material.DriftMobility,
                        Density = g.Material.Density,
                        // Liquids re-derive their volume ratio from the gathered occupancy every substep.
                        IsFluid = g.Material.Model == ConstitutiveModel.Fluid,
                    };
                    float sp = Transfers == TransferMode.Fused
                        ? Simulation.Transfers.G2PFused(Grid, g, p)
                        : Simulation.Transfers.G2PReference(Grid, g, p);
                    if (sp > maxSpeed) maxSpeed = sp;
                }
                tq += Stopwatch.GetTimestamp() - c;
            }
            _lastMaxSpeed = maxSpeed;

            long o0 = Stopwatch.GetTimestamp();
            SolidifyLava();
            if (++_frame % 24 == 0) SortSpatially();
            tOther += Stopwatch.GetTimestamp() - o0;

            double toMs = 1000.0 / Stopwatch.Frequency;
            st.Substeps = substeps;
            st.Dt = dt;
            st.ConstitutiveMs = tc * toMs;
            st.P2GMs = tp * toMs;
            st.GridMs = tg * toMs;
            st.G2PMs = tq * toMs;
            st.OtherMs = tOther * toMs;
            st.TotalMs = (Stopwatch.GetTimestamp() - t0) * toMs;
            st.Particles = ParticleCount;
            st.MaxSpeed = maxSpeed;
            Stats = st;
        }

        /// <summary>Fused P2G for group <paramref name="i"/>.</summary>
        /// <param name="i">Group index.</param>
        private void Transfers_P2G(int i) => Simulation.Transfers.P2GFused(Grid, _groups[i], _mass[i], _volume, _coolant[i]);

        // ------------------------------------------------------------------ spawning / erasing

        /// <summary>
        /// Fills an axis-aligned rectangle with material at the rest density (2×2 particles per cell, jittered),
        /// skipping points inside obstacles and topping up — never over-packing — cells that already hold
        /// material (see <see cref="Fill"/>). Scene builders therefore place the ENCLOSED material first (the oil
        /// blob, then the water around it): the later fill flows around what is already there.
        /// </summary>
        /// <param name="id">Material.</param>
        /// <param name="x0">Left.</param>
        /// <param name="y0">Bottom.</param>
        /// <param name="x1">Right.</param>
        /// <param name="y1">Top.</param>
        /// <param name="vx">Initial velocity x.</param>
        /// <param name="vy">Initial velocity y.</param>
        /// <returns>Particles added.</returns>
        public int FillRect(MaterialId id, float x0, float y0, float x1, float y1, float vx = 0, float vy = 0)
            => Fill(id, x0, y0, x1, y1, (x, y) => true, vx, vy);

        /// <summary>Fills a disk with material at rest density, around anything already there (see <see cref="FillRect"/>).</summary>
        /// <param name="id">Material.</param>
        /// <param name="cx">Center x.</param>
        /// <param name="cy">Center y.</param>
        /// <param name="r">Radius.</param>
        /// <param name="vx">Initial velocity x.</param>
        /// <param name="vy">Initial velocity y.</param>
        /// <returns>Particles added.</returns>
        public int FillDisk(MaterialId id, float cx, float cy, float r, float vx = 0, float vy = 0)
            => Fill(id, cx - r, cy - r, cx + r, cy + r, (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r, vx, vy);

        /// <summary>
        /// The painting brush (and every faucet): adds material in a disk, only where there is room — pouring
        /// into a full region adds nothing, and a half-empty cell is topped up to the rest packing.
        /// </summary>
        /// <param name="id">Material.</param>
        /// <param name="cx">Center x.</param>
        /// <param name="cy">Center y.</param>
        /// <param name="r">Radius.</param>
        /// <param name="vx">Initial velocity x (e.g. the mouse's, to throw material).</param>
        /// <param name="vy">Initial velocity y.</param>
        /// <returns>Particles added.</returns>
        public int Paint(MaterialId id, float cx, float cy, float r, float vx = 0, float vy = 0)
            => Fill(id, cx - r, cy - r, cx + r, cy + r, (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r, vx, vy);

        /// <summary>
        /// Shared filler: one jittered point per free quarter-cell slot of a bounding box, filtered by a shape test
        /// and by obstacles (φ &lt; 1 cell).
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>Why every fill is capped.</b> Liquids read their volume ratio from the local occupancy (J = 1/φ), so
        /// placing material on top of material is not "denser water" but a compressed spring: a region packed twice
        /// over reads J = ½ and releases K·½·½ of pressure in the first substep — the whole tank erupts. The rest
        /// packing is one particle per quarter-cell slot (2×2 per cell) for EVERY material — all particles carry the
        /// same volume V₀ = Δx²/4 — so a slot that already holds a particle, of any material, is full.
        /// </para>
        /// <para>
        /// The cap is per SLOT, not per cell: a per-cell count lets a second fill pack a half-covered edge cell's
        /// two free places into the same covered half (double density there). The lattice sits at slot centers and
        /// the jitter (±0.3 slot) never leaves the slot, so a fill can neither collide with itself nor with what is
        /// there. Occupancy comes from the particles' current positions, not from the last grid transfer, so
        /// back-to-back fills in a scene builder (and a faucet refilling the nozzle its last burst has only partly
        /// left) see each other.
        /// </para>
        /// </remarks>
        /// <param name="id">Material.</param>
        /// <param name="x0">Bounding box left.</param>
        /// <param name="y0">Bounding box bottom.</param>
        /// <param name="x1">Bounding box right.</param>
        /// <param name="y1">Bounding box top.</param>
        /// <param name="inside">Shape test for a candidate point.</param>
        /// <param name="vx">Initial velocity x.</param>
        /// <param name="vy">Initial velocity y.</param>
        /// <returns>Particles added.</returns>
        private int Fill(MaterialId id, float x0, float y0, float x1, float y1, Func<float, float, bool> inside, float vx, float vy)
        {
            float h = 0.5f * Grid.Dx, invH = 2f * Grid.InvDx;
            var xs = new List<float>();
            var ys = new List<float>();
            float* phi = Grid.Phi.Unsafe.Pointer<float>();
            int nx = Grid.Nx, ny = Grid.Ny, sw = 2 * nx, sh = 2 * ny;
            int[] slots = CountParticlesPerSlot();
            // Slots whose CENTER lies in [x0, x1) × [y0, y1).
            int si0 = Math.Max(0, (int)MathF.Ceiling(x0 * invH - 0.5f)), si1 = Math.Min(sw - 1, (int)MathF.Ceiling(x1 * invH - 0.5f) - 1);
            int sj0 = Math.Max(0, (int)MathF.Ceiling(y0 * invH - 0.5f)), sj1 = Math.Min(sh - 1, (int)MathF.Ceiling(y1 * invH - 0.5f) - 1);
            for (int sj = sj0; sj <= sj1; sj++)
                for (int si = si0; si <= si1; si++)
                {
                    ref int taken = ref slots[sj * sw + si];
                    if (taken > 0) continue;                 // the slot already holds material
                    float jx = (si + 0.5f + (float)(_rng.NextDouble() - 0.5) * 0.6f) * h;
                    float jy = (sj + 0.5f + (float)(_rng.NextDouble() - 0.5) * 0.6f) * h;
                    if (!inside(jx, jy)) continue;
                    int i = (int)(jx * Grid.InvDx + 0.5f), j = (int)(jy * Grid.InvDx + 0.5f);
                    if ((uint)i >= (uint)nx || (uint)j >= (uint)ny) continue;
                    if (phi[j * nx + i] < 1.0f) continue;     // inside or touching an obstacle / the tank walls
                    taken = 1;
                    xs.Add(jx); ys.Add(jy);
                }
            return _groups[(int)id].Append(xs.ToArray(), ys.ToArray(), vx, vy);
        }

        /// <summary>
        /// Counts the particles (all materials) in every quarter-cell slot from their current positions, into a
        /// reused buffer — O(particles), a few tens of microseconds, cheap enough for a brush stroke every frame.
        /// </summary>
        /// <returns>Row-major per-slot counts (<c>2Nx·2Ny</c>); valid until the next call.</returns>
        private int[] CountParticlesPerSlot()
        {
            int sw = 2 * Grid.Nx, sh = 2 * Grid.Ny;
            if (_slotCount == null || _slotCount.Length != sw * sh) _slotCount = new int[sw * sh];
            Array.Clear(_slotCount);
            float inv = 2f * Grid.InvDx;
            foreach (var g in _groups)
            {
                if (g.Count == 0) continue;   // an empty group may not have its buffers yet
                float* X = g.Ptr(PField.X), Y = g.Ptr(PField.Y);
                for (int p = 0; p < g.Count; p++)
                {
                    int si = (int)(X[p] * inv), sj = (int)(Y[p] * inv);
                    if ((uint)si < (uint)sw && (uint)sj < (uint)sh) _slotCount[sj * sw + si]++;
                }
            }
            return _slotCount;
        }

        /// <summary>Removes every particle (any material) inside a disk.</summary>
        /// <param name="cx">Center x.</param>
        /// <param name="cy">Center y.</param>
        /// <param name="r">Radius.</param>
        /// <returns>Particles removed.</returns>
        public int Erase(float cx, float cy, float r)
        {
            int removed = 0;
            float r2 = r * r;
            foreach (var g in _groups)
                removed += g.RemoveWhere((x, y, _) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2);
            return removed;
        }

        /// <summary>Removes every particle and faucet (obstacles stay).</summary>
        public void ClearParticles()
        {
            foreach (var g in _groups) g.Clear();
            Emitters.Clear();
            _lastMaxSpeed = 0;
        }

        /// <summary>Empties the world completely: particles, faucets and painted obstacles.</summary>
        public void Reset()
        {
            ClearParticles();
            Grid.ResetObstacles();
            GravityAngle = 0;
        }

        /// <summary>Runs every faucet for one frame: a jittered disk at the nozzle, only where there is room.</summary>
        private void RunEmitters()
        {
            foreach (var e in Emitters)
            {
                if (e.Budget > 0 && e.Emitted >= e.Budget) continue;
                e.Emitted += Paint(e.Material, e.X, e.Y, e.Radius, e.VX, e.VY);
            }
        }

        // ------------------------------------------------------------------ lava → rock, sorting

        /// <summary>Converts lava that has cooled below <see cref="LavaSolidifyAt"/> into rock (same position, velocity and color seed).</summary>
        private void SolidifyLava()
        {
            var lava = _groups[(int)MaterialId.Lava];
            if (lava.Count == 0) return;
            var rock = _groups[(int)MaterialId.Rock];
            float* temp = lava.Ptr(PField.Temp);
            float threshold = LavaSolidifyAt;
            lava.RemoveWhere((x, y, i) =>
            {
                if (temp[i] >= threshold) return false;
                rock.AppendFrom(lava, i);
                return true;
            });
        }

        /// <summary>
        /// Reorders each group's particles by grid cell (NumSharp <c>argsort</c> of the cell key + <c>take</c>
        /// into scratch), so the transfer kernels walk the grid nearly sequentially and stay in cache as
        /// materials mix. Pure permutation — the physics is unchanged.
        /// </summary>
        public void SortSpatially()
        {
            foreach (var g in _groups)
            {
                if (g.Count < 256) continue;
                g.PrepareViews();
                using var scope = NDScope.Open();
                double inv = Grid.InvDx;
                var key = np.evaluate(Cast(Floor(Arr(g[PField.Y]) * inv), NPTypeCode.Int32) * Grid.Nx
                                    + Cast(Floor(Arr(g[PField.X]) * inv), NPTypeCode.Int32));
                var order = np.argsort(key);
                var tmp = np.empty(new Shape(g.Count), NPTypeCode.Single);
                for (int f = 0; f < ParticleGroup.FieldCount; f++)
                {
                    var view = g[(PField)f];
                    np.take(view, order, @out: tmp);
                    np.copyto(view, tmp);
                }
            }
        }

        /// <summary>Releases every array.</summary>
        public void Dispose()
        {
            foreach (var g in _groups) g.Dispose();
            GridStep.Dispose();
            Params.Dispose();
            Grid.Dispose();
        }
    }
}
