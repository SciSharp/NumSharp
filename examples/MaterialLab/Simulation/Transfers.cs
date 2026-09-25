using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;
using NumSharp.Backends.Iteration;
using static NumSharp.Backends.Iteration.NDExpr;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>Which implementation moves data between particles and the grid.</summary>
    public enum TransferMode
    {
        /// <summary>Single-pass SIMD scatter/gather kernels over the NumSharp buffers (the real-time path).</summary>
        Fused,
        /// <summary>Pure NumSharp array math: <c>np.bincount</c> scatter-add + <c>np.take</c> gather + fused reductions (the reference path).</summary>
        Reference,
    }

    /// <summary>Per-group knobs of the particle update (G2P tail).</summary>
    /// <param name="Dt">Substep length.</param>
    /// <param name="MaxSpeed">Velocity clamp (a CFL safety net against user flings).</param>
    /// <param name="IsLava">Whether this group cools (temperature update).</param>
    /// <param name="CoolRate">Lava cooling per second in air.</param>
    /// <param name="QuenchRate">Extra lava cooling per second per unit of surrounding water mass fraction.</param>
    /// <param name="GravityX">Gravity x (units/s²) — the drift direction.</param>
    /// <param name="GravityY">Gravity y (units/s²).</param>
    /// <param name="Mobility">Drift-flux mobility τ of this material (0 = no relative drift).</param>
    /// <param name="Density">This material's density ρₚ.</param>
    /// <param name="IsFluid">Fluids take their volume ratio J = 1/φ from the gathered occupancy φ (see <see cref="Transfers"/> remarks).</param>
    /// <param name="InvCellArea">1/Δx² — converts the gathered volume into the occupancy φ.</param>
    public readonly record struct AdvectParams(float Dt, float MaxSpeed, bool IsLava, float CoolRate, float QuenchRate,
        float GravityX = 0f, float GravityY = 0f, float Mobility = 0f, float Density = 1f, bool IsFluid = false, float InvCellArea = 1f);

    /// <summary>
    /// The two MLS-MPM transfers — particle-to-grid (P2G, a scatter-add of mass, APIC momentum and stress into
    /// the 3×3 neighbouring nodes) and grid-to-particle (G2P, a gather of velocity and its gradient) — in two
    /// interchangeable implementations over the SAME buffers. See <see cref="TransferMode"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why the fused kernels exist.</b> As whole-array math, P2G must materialize nine contributions per
    /// particle (9N-element index, weight and momentum arrays) before <c>np.bincount</c> can add them, and G2P
    /// materializes nine gathered velocities per particle before reducing them; at 28K particles that is ~5×
    /// the cost of doing the same arithmetic in one pass with the node in registers. Both forms are kept: the
    /// reference is the specification (plain NumPy-style MLS-MPM, readable line by line), the fused kernels are
    /// the speed, and the verification harness asserts they agree to float rounding.
    /// </para>
    /// <para>
    /// <b>Weights.</b> Quadratic B-splines: for a particle at x/Δx, base = ⌊x/Δx − ½⌋, f = x/Δx − base ∈ [½, 3/2),
    /// w₀ = ½(3/2−f)², w₁ = ¾ − (f−1)², w₂ = ½(f−½)². A node receives w·(m·v + A·(xᵢ − xₚ)); since
    /// Σ w·(xᵢ − xₚ) = 0 for this kernel, linear momentum is conserved exactly.
    /// </para>
    /// <para>
    /// <b>Drift-flux mixtures.</b> MPM gives every material at a node ONE shared velocity, so once two liquids mix
    /// they could never separate again. The particle update therefore adds a relative drift
    /// v_d = τ·(ρₚ − ρ_mix)/ρₚ·g, where ρ_mix = Σw·m / Σw·V is the local mixture density gathered from the grid:
    /// oil droplets rise through water, sand settles out of it, and a pure region (ρₚ = ρ_mix) does not drift at
    /// all. Because Σ m·v_d = Σ V·(ρₚ − ρ_mix)·τg = 0 per node, the drift moves material without creating net
    /// momentum. It is applied to positions only.
    /// </para>
    /// <para>
    /// <b>Liquid volume from occupancy, not from history.</b> Tracking a liquid's volume ratio by integrating the
    /// velocity divergence (J ← J·(1 + Δt·tr C)) drifts, and clamping it (to stop runaway tension) silently erases
    /// the particle's memory of being stretched — recompression then reads as over-compression and the liquid
    /// slowly GAINS volume. Instead each liquid particle takes J = 1/φ, where φ = Σw·(V + V_wall) / Δx² is the
    /// local occupancy gathered from the volume and wall-volume lanes (1 = rest packing): no memory, no drift,
    /// every material's volume counts (sand displaces water — that is its buoyancy), an obstacle reads as full
    /// (see <see cref="MpmGrid.LaneWall"/>) and a free surface naturally reads φ &lt; 1.
    /// </para>
    /// </remarks>
    public static unsafe class Transfers
    {
        // ------------------------------------------------------------------ fused (SIMD) path

        /// <summary>
        /// Fused P2G for one group: accumulates <c>[m, m·vₓ + (A·d)ₓ, m·v_y + (A·d)_y, V₀, coolant, 0, 0, 0]</c> into
        /// the 9 neighbouring nodes with one Vector256 fused multiply-add per node. The grid must have been cleared.
        /// </summary>
        /// <param name="grid">The grid (interleaved 8-lane nodes).</param>
        /// <param name="g">The particle group (its A00..A11 scratch must hold the affine matrix).</param>
        /// <param name="mass">Particle mass for this material.</param>
        /// <param name="volume">Particle volume V₀ (same for every material: 2×2 particles per cell).</param>
        /// <param name="coolant">Whether this material counts as coolant (water) for the lava quench field.</param>
        /// <remarks>Compiled fully optimized from the first call — see the matching remark on <see cref="G2PFused"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void P2GFused(MpmGrid grid, ParticleGroup g, float mass, float volume, bool coolant)
        {
            if (g.Count == 0) return;
            float* nodes = grid.Nodes.Unsafe.Pointer<float>();
            float* X = g.Ptr(PField.X), Y = g.Ptr(PField.Y), VX = g.Ptr(PField.VX), VY = g.Ptr(PField.VY);
            float* a00 = g.Ptr(SField.A00), a01 = g.Ptr(SField.A01), a10 = g.Ptr(SField.A10), a11 = g.Ptr(SField.A11);
            float invDx = grid.InvDx, dx = grid.Dx, m = mass, cool = coolant ? mass : 0f;
            int nx = grid.Nx, n = g.Count;
            const int L = MpmGrid.Lanes;
            for (int p = 0; p < n; p++)
            {
                float xs = X[p] * invDx, ys = Y[p] * invDx;
                int bx = (int)(xs - 0.5f), by = (int)(ys - 0.5f);
                float fx = xs - bx, fy = ys - by;
                float wx0 = 0.5f * (1.5f - fx) * (1.5f - fx), wx1 = 0.75f - (fx - 1f) * (fx - 1f), wx2 = 0.5f * (fx - 0.5f) * (fx - 0.5f);
                float wy0 = 0.5f * (1.5f - fy) * (1.5f - fy), wy1 = 0.75f - (fy - 1f) * (fy - 1f), wy2 = 0.5f * (fy - 0.5f) * (fy - 0.5f);
                float A0 = a00[p] * dx, A1 = a01[p] * dx, A2 = a10[p] * dx, A3 = a11[p] * dx;
                // Node (i,j) receives base + i·ax + j·ay: the affine term evaluated at d = (i − fx, j − fy)·Δx.
                var bas = Vector256.Create(m, m * VX[p] - A0 * fx - A1 * fy, m * VY[p] - A2 * fx - A3 * fy, volume, cool, 0f, 0f, 0f);
                var ax = Vector256.Create(0f, A0, A2, 0f, 0f, 0f, 0f, 0f);
                var ay = Vector256.Create(0f, A1, A3, 0f, 0f, 0f, 0f, 0f);
                float* row = nodes + ((long)by * nx + bx) * L;
                ScatterRow(row, bas, ax, wy0, wx0, wx1, wx2);
                ScatterRow(row + nx * L, bas + ay, ax, wy1, wx0, wx1, wx2);
                ScatterRow(row + nx * 2 * L, bas + ay + ay, ax, wy2, wx0, wx1, wx2);
            }
        }

        /// <summary>Adds one row of three node contributions (weights wy·wx₀..₂), each as one fused multiply-add.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ScatterRow(float* r, Vector256<float> v, Vector256<float> ax, float wy, float w0, float w1, float w2)
        {
            const int L = MpmGrid.Lanes;
            Vector256.Store(MulAdd(v, Vector256.Create(wy * w0), Vector256.Load(r)), r);
            v += ax; Vector256.Store(MulAdd(v, Vector256.Create(wy * w1), Vector256.Load(r + L)), r + L);
            v += ax; Vector256.Store(MulAdd(v, Vector256.Create(wy * w2), Vector256.Load(r + 2 * L)), r + 2 * L);
        }

        /// <summary>
        /// a·b + c as one fused multiply-add (a single rounding) where the CPU has FMA3, else as a multiply and an
        /// add. The two paths differ only in the last bit of rounding — well inside the fused-vs-reference tolerance.
        /// </summary>
        /// <param name="a">Multiplicand.</param>
        /// <param name="b">Multiplier.</param>
        /// <param name="c">Addend.</param>
        /// <returns>a·b + c per lane.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MulAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
            => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c) : a * b + c;

        /// <summary>
        /// Fused G2P for one group: gathers the new velocity v = Σ w·vᵢ, the APIC matrix
        /// C = 4/Δx² · Σ w·vᵢ·(xᵢ − xₚ)ᵀ, the local occupancy, mixture density and water fraction, then advects the
        /// particle and resolves collisions (see <see cref="FinishParticle"/>).
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>Arithmetic shape.</b> The 3×3 stencil is reduced row by row. For row j with nodes n₀, n₁, n₂ and
        /// t = wx₁·n₁: s_j = wx₀·n₀ + wx₂·n₂ + t = Σᵢ wxᵢ·nᵢ, and m_j = 2wx₂·n₂ + t = Σᵢ i·wxᵢ·nᵢ (the row's first
        /// moment). Then Σw·n = Σⱼ wyⱼ·s_j, Σw·(i − fx)·n = Σⱼ wyⱼ·m_j − fx·Σw·n and
        /// Σw·(j − fy)·n = wy₁·s₁ + 2wy₂·s₂ − fy·Σw·n — nine node loads and about twenty fused multiply-adds per
        /// particle, all eight node lanes at once. The direct form (accumulate w·(xᵢ − xₚ)·n node by node) needs
        /// roughly three times the vector operations and more live registers than AVX2 has, so the JIT spills to the
        /// stack; it measured 13% slower end to end (18.9 → 16.3 ns per particle on a settled 30K-particle pool).
        /// </para>
        /// <para>
        /// <b>Full optimization from the first call.</b> The whole cost of this method is its loop, so under tiered
        /// compilation the first calls run through on-stack replacement code, and an app would stutter through its
        /// first second; <see cref="MethodImplOptions.AggressiveOptimization"/> compiles it fully optimized once.
        /// </para>
        /// </remarks>
        /// <param name="grid">The grid after the grid update (lanes 1–2 = velocity).</param>
        /// <param name="g">The group.</param>
        /// <param name="ap">Advection parameters.</param>
        /// <returns>The largest particle speed after the update (drives the next step's CFL check).</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static float G2PFused(MpmGrid grid, ParticleGroup g, in AdvectParams ap)
        {
            if (g.Count == 0) return 0f;
            var par = ap;   // a local copy: the pointer stores below cannot alias it, so its fields stay in registers
            float* nodes = grid.Nodes.Unsafe.Pointer<float>();
            float* X = g.Ptr(PField.X), Y = g.Ptr(PField.Y), VX = g.Ptr(PField.VX), VY = g.Ptr(PField.VY);
            float* c00 = g.Ptr(PField.C00), c01 = g.Ptr(PField.C01), c10 = g.Ptr(PField.C10), c11 = g.Ptr(PField.C11);
            float* T = g.Ptr(PField.Temp), J = g.Ptr(PField.Jp);
            float invDx = grid.InvDx, k4 = 4f * grid.InvDx;
            int nx = grid.Nx, n = g.Count;
            const int L = MpmGrid.Lanes;
            var ctx = new GridSampler(grid);
            float maxSpeedSq = 0f;
            bool fluid = par.IsFluid, lava = par.IsLava, drift = par.Mobility > 0f;
            float invArea = par.InvCellArea;
            // The three gathered vectors (value, x-moment, y-moment) go through a small stack buffer: scalar reads
            // of a stored vector are cheaper than a lane-extract sequence per component.
            float* sum = stackalloc float[3 * L];
            for (int p = 0; p < n; p++)
            {
                float xs = X[p] * invDx, ys = Y[p] * invDx;
                int bx = (int)(xs - 0.5f), by = (int)(ys - 0.5f);
                float fx = xs - bx, fy = ys - by;
                float ex0 = 1.5f - fx, ex1 = fx - 1f, ex2 = fx - 0.5f;
                float ey0 = 1.5f - fy, ey1 = fy - 1f, ey2 = fy - 0.5f;
                float wx2s = 0.5f * ex2 * ex2, wy2s = 0.5f * ey2 * ey2;
                var wx0 = Vector256.Create(0.5f * ex0 * ex0);
                var wx1 = Vector256.Create(0.75f - ex1 * ex1);
                var wx2 = Vector256.Create(wx2s);
                var wx2d = Vector256.Create(wx2s + wx2s);
                var wy0 = Vector256.Create(0.5f * ey0 * ey0);
                var wy1 = Vector256.Create(0.75f - ey1 * ey1);
                var wy2 = Vector256.Create(wy2s);
                var wy2d = Vector256.Create(wy2s + wy2s);
                float* r0 = nodes + ((long)by * nx + bx) * L;
                float* r1 = r0 + nx * L, r2 = r1 + nx * L;
                // Row sums s_j = Σᵢ wxᵢ·nᵢ and first moments m_j = Σᵢ i·wxᵢ·nᵢ (see remarks).
                var n02 = Vector256.Load(r0 + 2 * L); var t0 = Vector256.Load(r0 + L) * wx1;
                var s0 = MulAdd(Vector256.Load(r0), wx0, MulAdd(n02, wx2, t0)); var m0 = MulAdd(n02, wx2d, t0);
                var n12 = Vector256.Load(r1 + 2 * L); var t1 = Vector256.Load(r1 + L) * wx1;
                var s1 = MulAdd(Vector256.Load(r1), wx0, MulAdd(n12, wx2, t1)); var m1 = MulAdd(n12, wx2d, t1);
                var n22 = Vector256.Load(r2 + 2 * L); var t2 = Vector256.Load(r2 + L) * wx1;
                var s2 = MulAdd(Vector256.Load(r2), wx0, MulAdd(n22, wx2, t2)); var m2 = MulAdd(n22, wx2d, t2);
                var sv = MulAdd(s0, wy0, MulAdd(s1, wy1, s2 * wy2));
                Vector256.Store(sv, sum);
                Vector256.Store(MulAdd(sv, Vector256.Create(-fx), MulAdd(m0, wy0, MulAdd(m1, wy1, m2 * wy2))), sum + L);
                Vector256.Store(MulAdd(sv, Vector256.Create(-fy), MulAdd(s2, wy2d, s1 * wy1)), sum + 2 * L);
                float vx = sum[MpmGrid.LaneMomX], vy = sum[MpmGrid.LaneMomY];
                c00[p] = k4 * sum[L + MpmGrid.LaneMomX]; c01[p] = k4 * sum[2 * L + MpmGrid.LaneMomX];
                c10[p] = k4 * sum[L + MpmGrid.LaneMomY]; c11[p] = k4 * sum[2 * L + MpmGrid.LaneMomY];
                float massSum = sum[MpmGrid.LaneMass], volSum = sum[MpmGrid.LaneVolume];
                // Divisions only where a material uses them (a divide costs more than the rest of the tail).
                float coolantFraction = lava ? sum[MpmGrid.LaneCoolant] / MathF.Max(massSum, 1e-30f) : 0f;
                float mixDensity = drift ? massSum / MathF.Max(volSum, 1e-30f) : 0f;
                // Occupancy counts the obstacle as full (its wall-volume lane), the mixture density does not.
                float occupancy = (volSum + sum[MpmGrid.LaneWall]) * invArea;
                if (fluid) J[p] = 1f / MathF.Max(occupancy, 0.05f);   // J = 1/φ
                float x = X[p], y = Y[p], t = T[p];
                float sp2 = FinishParticle(ref ctx, par, ref x, ref y, ref vx, ref vy, ref t, coolantFraction, mixDensity, occupancy);
                X[p] = x; Y[p] = y; VX[p] = vx; VY[p] = vy; T[p] = t;
                if (sp2 > maxSpeedSq) maxSpeedSq = sp2;
            }
            return MathF.Sqrt(maxSpeedSq);
        }

        // ------------------------------------------------------------------ shared particle finish

        /// <summary>
        /// Bilinear access to the obstacle distance field and its normals at particle positions (the grid's
        /// node-based fields interpolated between nodes).
        /// </summary>
        internal readonly struct GridSampler
        {
            /// <summary>φ (cells) and outward normals at the nodes.</summary>
            private readonly float* _phi, _nx, _ny;
            /// <summary>Grid size in nodes.</summary>
            private readonly int _w, _h;
            /// <summary>1/Δx, Δx, and the position clamp that keeps every 3×3 stencil inside the node array.</summary>
            private readonly float _invDx, _dx, _lo, _hiX, _hiY;

            /// <summary>Captures the grid's field pointers.</summary>
            /// <param name="grid">The grid.</param>
            public GridSampler(MpmGrid grid)
            {
                _phi = grid.Phi.Unsafe.Pointer<float>();
                _nx = grid.NormalX.Unsafe.Pointer<float>();
                _ny = grid.NormalY.Unsafe.Pointer<float>();
                _w = grid.Nx; _h = grid.Ny;
                _invDx = grid.InvDx; _dx = grid.Dx;
                // Keep particles where their 3×3 stencil stays inside the node array.
                _lo = 1.5f * _dx; _hiX = (_w - 1.5f) * _dx - 1e-5f; _hiY = (_h - 1.5f) * _dx - 1e-5f;
            }

            /// <summary>Cell size.</summary>
            public float Dx => _dx;

            /// <summary>Clamps a position into the valid stencil range (a backstop — walls normally keep particles far inside).</summary>
            /// <param name="x">X (updated).</param>
            /// <param name="y">Y (updated).</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void ClampToDomain(ref float x, ref float y)
            {
                x = Math.Clamp(x, _lo, _hiX);
                y = Math.Clamp(y, _lo, _hiY);
            }

            /// <summary>
            /// φ at the node nearest to a position — one load. φ changes by at most ~1 cell per cell, so a
            /// nearest-node value above 1.5 proves the bilinear value is above ½ and the particle cannot need a
            /// push-out; most particles take this early exit.
            /// </summary>
            /// <param name="x">X (units).</param>
            /// <param name="y">Y (units).</param>
            /// <returns>φ at the nearest node (cells).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float Nearest(float x, float y) => _phi[(int)(y * _invDx + 0.5f) * _w + (int)(x * _invDx + 0.5f)];

            /// <summary>Samples φ (cells) and the outward normal at a position.</summary>
            /// <param name="x">X (units).</param>
            /// <param name="y">Y (units).</param>
            /// <param name="nx">Normal x.</param>
            /// <param name="ny">Normal y.</param>
            /// <returns>Signed distance in cells (positive in free space).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float Sample(float x, float y, out float nx, out float ny)
            {
                float xs = x * _invDx, ys = y * _invDx;
                int i = (int)xs, j = (int)ys;
                float tx = xs - i, ty = ys - j;
                int k = j * _w + i;
                float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
                nx = _nx[k] * w00 + _nx[k + 1] * w10 + _nx[k + _w] * w01 + _nx[k + _w + 1] * w11;
                ny = _ny[k] * w00 + _ny[k + 1] * w10 + _ny[k + _w] * w01 + _ny[k + _w + 1] * w11;
                return _phi[k] * w00 + _phi[k + 1] * w10 + _phi[k + _w] * w01 + _phi[k + _w + 1] * w11;
            }
        }

        /// <summary>
        /// The per-particle tail of G2P shared by both transfer paths: clamp the speed (CFL safety net), advect with
        /// the grid velocity plus the buoyant drift, push out of any obstacle the particle ended up inside
        /// (projecting away the velocity into the wall), clamp to the domain, and cool lava (faster when
        /// surrounded by water).
        /// </summary>
        /// <param name="s">Obstacle sampler.</param>
        /// <param name="ap">Advection parameters.</param>
        /// <param name="x">Position x (updated).</param>
        /// <param name="y">Position y (updated).</param>
        /// <param name="vx">Velocity x (updated).</param>
        /// <param name="vy">Velocity y (updated).</param>
        /// <param name="temp">Temperature (updated for lava).</param>
        /// <param name="coolantFraction">Mass fraction of water around the particle (0..1).</param>
        /// <param name="mixDensity">Local mixture density ρ_mix = Σw·m / Σw·V at the particle.</param>
        /// <param name="occupancy">Local occupancy φ = Σw·(V + V_wall) / Δx² (1 = packed; ≈0.6 at a free surface).</param>
        /// <returns>The particle's final speed SQUARED (callers track the maximum and take one square root).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float FinishParticle(ref GridSampler s, in AdvectParams ap, ref float x, ref float y,
            ref float vx, ref float vy, ref float temp, float coolantFraction, float mixDensity, float occupancy)
        {
            float sp2 = vx * vx + vy * vy;
            if (sp2 > ap.MaxSpeed * ap.MaxSpeed)
            {
                float k = ap.MaxSpeed / MathF.Sqrt(sp2);
                vx *= k; vy *= k;
            }
            float px = vx, py = vy;
            if (ap.Mobility > 0f)
            {
                // Drift-flux: relative settling/rising velocity from the density contrast with the local mixture,
                // bounded so a nearly-empty node (tiny volume) can never fling a particle. It fades out toward a
                // free surface (φ falling to its surface value ≈ 0.6): separation happens INSIDE a mixture, and a
                // light particle at the top of a thin layer — which still "sees" the heavy liquid under it through
                // the 3-cell kernel — would otherwise be driven straight out of the liquid, over and over.
                float contrast = Math.Clamp((ap.Density - mixDensity) / ap.Density, -1f, 1f);
                float inside = Math.Clamp((occupancy - 0.6f) * 2.5f, 0f, 1f);
                float k = ap.Mobility * contrast * inside;
                px += k * ap.GravityX;
                py += k * ap.GravityY;
            }
            x += ap.Dt * px;
            y += ap.Dt * py;
            s.ClampToDomain(ref x, ref y);
            if (s.Nearest(x, y) < 1.5f)
            {
                float phi = s.Sample(x, y, out float nx, out float ny);
                if (phi < 0.5f)
                {
                    // Inside (or skimming) an obstacle: move out along the normal and drop the inward velocity.
                    float push = (0.5f - phi) * s.Dx;
                    x += push * nx; y += push * ny;
                    float vn = vx * nx + vy * ny;
                    if (vn < 0f) { vx -= vn * nx; vy -= vn * ny; }
                    s.ClampToDomain(ref x, ref y);
                }
            }
            if (ap.IsLava)
                temp = MathF.Max(0f, temp - ap.Dt * (ap.CoolRate + ap.QuenchRate * coolantFraction));
            return vx * vx + vy * vy;
        }

        // ------------------------------------------------------------------ reference (pure NumSharp) path

        /// <summary>
        /// Reference P2G over ALL groups as pure NumSharp array math — the specification the fused kernel is
        /// verified against. Per group: the 3×3 stencil's weights and node indices are built as broadcast
        /// <c>(3, 3, N)</c> arrays by fused expressions, the momentum contributions likewise, and
        /// <c>np.bincount(index, weights, minlength: G)</c> performs each scatter-add. The sums are written into the
        /// node lanes.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <param name="groups">Particle groups with their A matrices computed.</param>
        /// <param name="masses">Particle mass per group (same order).</param>
        /// <param name="volume">Particle volume V₀.</param>
        /// <param name="coolant">Coolant flag per group.</param>
        public static void P2GReference(MpmGrid grid, ParticleGroup[] groups, float[] masses, float volume, bool[] coolant)
        {
            int G = grid.G;
            // Everything that must OUTLIVE the scope is created before it opens: the groups' cached views and the
            // stencil's static offset arrays (a static initializer running inside the scope would hand the scope
            // ownership of those arrays, and it would dispose them on exit).
            foreach (var g in groups) g.PrepareViews();
            Stencil.Warm();
            using var scope = NDScope.Open();
            var gm = np.zeros(new Shape(G), NPTypeCode.Double);
            var gmx = np.zeros(new Shape(G), NPTypeCode.Double);
            var gmy = np.zeros(new Shape(G), NPTypeCode.Double);
            var gvol = np.zeros(new Shape(G), NPTypeCode.Double);
            var gcl = np.zeros(new Shape(G), NPTypeCode.Double);
            for (int gi = 0; gi < groups.Length; gi++)
            {
                var g = groups[gi];
                if (g.Count == 0) continue;
                var st = Stencil.Build(grid, g);
                double m = masses[gi], dx = grid.Dx;
                int n = g.Count;
                NDArray R(NDArray a) => a.reshape(1, 1, n);
                var mx9 = np.evaluate(Arr(st.W9) * (m * Arr(R(g[PField.VX])) + (Arr(R(g[SField.A00])) * (Arr(Stencil.Io) - R(st.Fx)) + Arr(R(g[SField.A01])) * (Arr(Stencil.Jo) - R(st.Fy))) * dx));
                var my9 = np.evaluate(Arr(st.W9) * (m * Arr(R(g[PField.VY])) + (Arr(R(g[SField.A10])) * (Arr(Stencil.Io) - R(st.Fx)) + Arr(R(g[SField.A11])) * (Arr(Stencil.Jo) - R(st.Fy))) * dx));
                var idx = st.Index.ravel();
                var w = np.bincount(idx, st.W9.ravel(), G);   // Σw per node — mass and volume are its multiples
                var wm = w * m;
                np.add(gm, wm, @out: gm);
                np.add(gvol, w * (double)volume, @out: gvol);
                np.add(gmx, np.bincount(idx, mx9.ravel(), G), @out: gmx);
                np.add(gmy, np.bincount(idx, my9.ravel(), G), @out: gmy);
                if (coolant[gi]) np.add(gcl, wm, @out: gcl);
            }
            int ny = grid.Ny, nxg = grid.Nx;
            np.copyto(grid.Nodes[$":, :, {MpmGrid.LaneMass}"], gm.reshape(ny, nxg));
            np.copyto(grid.Nodes[$":, :, {MpmGrid.LaneMomX}"], gmx.reshape(ny, nxg));
            np.copyto(grid.Nodes[$":, :, {MpmGrid.LaneMomY}"], gmy.reshape(ny, nxg));
            np.copyto(grid.Nodes[$":, :, {MpmGrid.LaneVolume}"], gvol.reshape(ny, nxg));
            np.copyto(grid.Nodes[$":, :, {MpmGrid.LaneCoolant}"], gcl.reshape(ny, nxg));
        }

        /// <summary>
        /// Reference G2P for one group as pure NumSharp: gathers the 3×3 node lanes with <c>np.take</c> into
        /// <c>(9, N)</c> arrays and reduces them with fused axis-0 sums into v, C, the mixture density and the water
        /// fraction, then runs the same per-particle finish as the fused path.
        /// </summary>
        /// <param name="grid">The grid after the grid update.</param>
        /// <param name="g">The group.</param>
        /// <param name="ap">Advection parameters.</param>
        /// <returns>The largest particle speed after the update.</returns>
        public static float G2PReference(MpmGrid grid, ParticleGroup g, in AdvectParams ap)
        {
            if (g.Count == 0) return 0f;
            int n = g.Count;
            g.PrepareViews();   // see P2GReference: long-lived arrays are never created inside the scope
            Stencil.Warm();
            using (var scope = NDScope.Open())
            {
                var st = Stencil.Build(grid, g);
                var w9 = st.W9.reshape(9, n);
                NDArray Lane(int lane) => np.take(grid.Nodes[$":, :, {lane}"], st.Index).reshape(9, n);
                var gvx = Lane(MpmGrid.LaneMomX);
                var gvy = Lane(MpmGrid.LaneMomY);
                var gm = Lane(MpmGrid.LaneMass);
                var gvol = Lane(MpmGrid.LaneVolume);
                var gcl = Lane(MpmGrid.LaneCoolant);
                var gwall = Lane(MpmGrid.LaneWall);
                var fx1 = st.Fx.reshape(1, n); var fy1 = st.Fy.reshape(1, n);
                double k4 = 4.0 * grid.InvDx;
                np.evaluate(Sum(Arr(w9) * gvx, 0), @out: g[PField.VX]);
                np.evaluate(Sum(Arr(w9) * gvy, 0), @out: g[PField.VY]);
                np.evaluate(Sum(k4 * Arr(w9) * gvx * (Arr(Stencil.Io9) - fx1), 0), @out: g[PField.C00]);
                np.evaluate(Sum(k4 * Arr(w9) * gvx * (Arr(Stencil.Jo9) - fy1), 0), @out: g[PField.C01]);
                np.evaluate(Sum(k4 * Arr(w9) * gvy * (Arr(Stencil.Io9) - fx1), 0), @out: g[PField.C10]);
                np.evaluate(Sum(k4 * Arr(w9) * gvy * (Arr(Stencil.Jo9) - fy1), 0), @out: g[PField.C11]);
                // Water fraction and mixture density — two evaluations each, because a reduction must be the ROOT of
                // an expression; the divisions then run as their own fused passes (into the E0/E1 scratch).
                var sm = np.evaluate(Sum(Arr(w9) * gm, 0));
                var sc = np.evaluate(Sum(Arr(w9) * gcl, 0));
                var sv = np.evaluate(Sum(Arr(w9) * gvol, 0));
                np.evaluate(Arr(sc) / Maximum(Arr(sm), 1e-30), @out: g[SField.E0]);
                np.evaluate(Arr(sm) / Maximum(Arr(sv), 1e-30), @out: g[SField.E1]);
                // Occupancy volume: particles plus the wall lane (the obstacle reads as full, see MpmGrid.LaneWall).
                np.evaluate(Arr(sv) + np.evaluate(Sum(Arr(w9) * gwall, 0)), @out: g[SField.E2]);
            }
            var ctx = new GridSampler(grid);
            float* X = g.Ptr(PField.X), Y = g.Ptr(PField.Y), VX = g.Ptr(PField.VX), VY = g.Ptr(PField.VY), T = g.Ptr(PField.Temp);
            float* cf = g.Ptr(SField.E0), rho = g.Ptr(SField.E1), occ = g.Ptr(SField.E2), J = g.Ptr(PField.Jp);
            float maxSpeedSq = 0f;
            for (int p = 0; p < n; p++)
            {
                float occupancy = occ[p] * ap.InvCellArea;
                if (ap.IsFluid) J[p] = 1f / MathF.Max(occupancy, 0.05f);   // J = 1/φ, as the fused path
                float x = X[p], y = Y[p], vx = VX[p], vy = VY[p], t = T[p];
                float sp2 = FinishParticle(ref ctx, ap, ref x, ref y, ref vx, ref vy, ref t, cf[p], rho[p], occupancy);
                X[p] = x; Y[p] = y; VX[p] = vx; VY[p] = vy; T[p] = t;
                if (sp2 > maxSpeedSq) maxSpeedSq = sp2;
            }
            return MathF.Sqrt(maxSpeedSq);
        }

        /// <summary>
        /// The quadratic-B-spline stencil of a group as NumSharp arrays: fractional offsets, the <c>(3,3,N)</c>
        /// weights w[i,j,p] = wₓ[i,p]·w_y[j,p] and flat node indices (by+j)·Nx + bx+i.
        /// </summary>
        private readonly struct Stencil
        {
            /// <summary>Column offsets i as (3,1,1) (broadcast against (1,1,N)).</summary>
            public static readonly NDArray Io = np.array(new float[] { 0, 1, 2 }).reshape(3, 1, 1);
            /// <summary>Row offsets j as (1,3,1).</summary>
            public static readonly NDArray Jo = np.array(new float[] { 0, 1, 2 }).reshape(1, 3, 1);
            /// <summary>Column offset i per flattened stencil row k = 3i + j, as (9,1).</summary>
            public static readonly NDArray Io9 = np.array(new float[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 }).reshape(9, 1);
            /// <summary>Row offset j per flattened stencil row, as (9,1).</summary>
            public static readonly NDArray Jo9 = np.array(new float[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 }).reshape(9, 1);
            /// <summary>Column offsets i as int64 (3,1,1), for node indices.</summary>
            private static readonly NDArray IoL = np.array(new long[] { 0, 1, 2 }).reshape(3, 1, 1);
            /// <summary>Row offsets j as int64 (1,3,1), for node indices.</summary>
            private static readonly NDArray JoL = np.array(new long[] { 0, 1, 2 }).reshape(1, 3, 1);

            /// <summary>
            /// Forces the static offset arrays to exist. MUST be called before opening an <see cref="NDScope"/> that
            /// uses the stencil: arrays constructed while a scope is open are owned (and later disposed) by that
            /// scope, which would leave these process-lifetime statics disposed.
            /// </summary>
            /// <returns>A value derived from the arrays (so the call cannot be optimized away).</returns>
            public static int Warm() => Io.ndim + Jo.ndim + Io9.ndim + Jo9.ndim + IoL.ndim + JoL.ndim;

            /// <summary>Fractional offset f_x ∈ [½, 3/2), shape (N,).</summary>
            public NDArray Fx { get; init; }
            /// <summary>Fractional offset f_y, shape (N,).</summary>
            public NDArray Fy { get; init; }
            /// <summary>Weights, shape (3,3,N).</summary>
            public NDArray W9 { get; init; }
            /// <summary>Flat node indices (int64), shape (3,3,N).</summary>
            public NDArray Index { get; init; }

            /// <summary>Builds the stencil of every particle of <paramref name="g"/> with fused NumSharp expressions.</summary>
            /// <param name="grid">The grid.</param>
            /// <param name="g">The group.</param>
            /// <returns>The stencil arrays.</returns>
            public static Stencil Build(MpmGrid grid, ParticleGroup g)
            {
                int n = g.Count;
                double inv = grid.InvDx;
                NDExpr xs = Arr(g[PField.X]) * inv, ys = Arr(g[PField.Y]) * inv;
                var fx = np.evaluate(xs - Floor(xs - 0.5));
                var fy = np.evaluate(ys - Floor(ys - 0.5));
                var wx = np.empty(new Shape(3, n), NPTypeCode.Single);
                var wy = np.empty(new Shape(3, n), NPTypeCode.Single);
                np.evaluate(0.5 * Square(1.5 - Arr(fx)), @out: wx["0"]);
                np.evaluate(0.75 - Square(Arr(fx) - 1.0), @out: wx["1"]);
                np.evaluate(0.5 * Square(Arr(fx) - 0.5), @out: wx["2"]);
                np.evaluate(0.5 * Square(1.5 - Arr(fy)), @out: wy["0"]);
                np.evaluate(0.75 - Square(Arr(fy) - 1.0), @out: wy["1"]);
                np.evaluate(0.5 * Square(Arr(fy) - 0.5), @out: wy["2"]);
                var w9 = np.evaluate(Arr(wx.reshape(3, 1, n)) * wy.reshape(1, 3, n));
                var px3 = g[PField.X].reshape(1, 1, n); var py3 = g[PField.Y].reshape(1, 1, n);
                var index = np.evaluate(Cast(Floor(Arr(px3) * inv - 0.5), NPTypeCode.Int64) + IoL
                                      + (Cast(Floor(Arr(py3) * inv - 0.5), NPTypeCode.Int64) + JoL) * (long)grid.Nx);
                return new Stencil { Fx = fx, Fy = fy, W9 = w9, Index = index };
            }
        }
    }
}
