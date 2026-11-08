using System;
using System.Collections.Generic;
using NumSharp;
using NumSharp.Backends.Iteration;
using static NumSharp.Backends.Iteration.NDExpr;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>
    /// The runtime scalars every constitutive kernel reads, held as 0-d float32 NumSharp arrays. A 0-d array
    /// leaf in a compiled <see cref="NDExpr"/> is a HOISTED PARAMETER: its value is read once per evaluation
    /// and never streamed, so changing the time step is a single store — no kernel is recompiled, and the
    /// per-element cost is zero (a literal would bake the value into the IL and force a recompile per value).
    /// </summary>
    public sealed class SolverParams : IDisposable
    {
        /// <summary>The substep Δt.</summary>
        public NDArray Dt { get; } = NDArray.Scalar(1e-4f);

        /// <summary>
        /// The MLS-MPM stress scale −Δt·V₀·4/Δx², which reduces to −Δt because every particle starts with volume
        /// V₀ = (Δx/2)² (2×2 particles per cell).
        /// </summary>
        public NDArray StressScale { get; } = NDArray.Scalar(-1e-4f);

        /// <summary>Per-material effective viscosity, clamped to the explicit stability limit at the current Δt (index = <see cref="MaterialId"/>).</summary>
        public NDArray[] Viscosity { get; }

        /// <summary>Per-material viscous stability cap at the current Δt — the ceiling a temperature-thickened viscosity (cooling lava) may reach.</summary>
        public NDArray[] ViscosityCap { get; }

        /// <summary>Creates the parameter set with one viscosity slot per material.</summary>
        public SolverParams()
        {
            Viscosity = new NDArray[Materials.Count];
            ViscosityCap = new NDArray[Materials.Count];
            for (int i = 0; i < Viscosity.Length; i++)
            {
                Viscosity[i] = NDArray.Scalar(0f);
                ViscosityCap[i] = NDArray.Scalar(0f);
            }
        }

        /// <summary>Publishes a new time step (and the viscosities it allows) to every kernel.</summary>
        /// <param name="dt">Substep length in seconds.</param>
        /// <param name="dx">Grid cell size.</param>
        public void Update(float dt, float dx)
        {
            Dt.SetValue(dt);
            StressScale.SetValue(-dt);
            for (int i = 0; i < Viscosity.Length; i++)
            {
                var m = Materials.All[i];
                float cap = m.MaxStableViscosity(dt, dx);
                ViscosityCap[i].SetValue(cap);
                Viscosity[i].SetValue(Math.Min(m.Viscosity, cap));
            }
        }

        /// <summary>Releases the parameter arrays.</summary>
        public void Dispose()
        {
            Dt.Dispose();
            StressScale.Dispose();
            foreach (var v in Viscosity) v.Dispose();
            foreach (var v in ViscosityCap) v.Dispose();
        }
    }

    /// <summary>
    /// The material physics of one particle group, as a pipeline of fused NumSharp expressions. Each stage
    /// is one <see cref="NDExpr"/> tree over the group's live views, compiled once
    /// (<see cref="NDExpr.Compile()"/>) and evaluated in place (<c>out=</c>) every substep — so a whole
    /// constitutive update is a handful of single-pass SIMD kernels with no temporaries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>The σ-space formulation.</b> Plasticity needs the singular values of the trial deformation gradient
    /// T = (I + Δt·C)·F. In 2-D they have a closed form with no trigonometry: with S = T·Tᵀ = [[p,q],[q,r]],
    /// the eigen-gap is D = √((p−r)² + 4q²), σ₁ = √((p+r+D)/2) and σ₂ = det T / σ₁ (signed, so inversion is
    /// represented), and the left rotation U enters only through cos 2φ = (p−r)/D and sin 2φ = 2q/D. Any
    /// U·diag(k₁,k₂)·Uᵀ is then h·I + g·[[cos 2φ, sin 2φ],[sin 2φ, −cos 2φ]] with h = (k₁+k₂)/2,
    /// g = (k₁−k₂)/2. That one identity gives both the plastic return (F = U·diag(σ′/σ)·Uᵀ·T) and the
    /// Kirchhoff stress (τ = U·diag(t)·Uᵀ) as plain arithmetic — ideal for fused array expressions.
    /// </para>
    /// <para>
    /// <b>Rebuild rule.</b> The trees reference the group's views, which change whenever the particle count
    /// changes; the stages are rebuilt then. Rebuilding is cheap — NumSharp's structural program cache
    /// recognizes the same tree shape and reuses the already-JIT-compiled kernel.
    /// </para>
    /// <para>
    /// <b>Hoist every transcendental into its own stage.</b> A fused tree evaluates each node where it appears —
    /// there is no common-subexpression elimination — so a subtree used five times is computed five times, and
    /// <c>Log</c>/<c>Exp</c> cost ~1.1–1.4 ns per element against ~0.2 ns for plain SIMD arithmetic (measured
    /// on 6,000-element stages; the overflow-safe <c>Hypot</c> is worse still, so magnitudes are written as
    /// <c>Sqrt(a² + b²)</c>). The Drucker–Prager projection written inline reads ln σ about ten times: 14.3 ns
    /// per element. Each such value is therefore written ONCE into a scratch lane (the A-matrix lanes are free
    /// until the affine stage at the very end) and read back as a plain operand — the granular constitutive
    /// update got 3.5× faster. (A transcendental that sits in only ONE branch of a <c>Where</c> is the
    /// exception: there it runs only where that branch is taken, and hoisting would compute it everywhere.)
    /// </para>
    /// </remarks>
    public sealed class ConstitutiveKernels
    {
        /// <summary>The group whose particles these stages update.</summary>
        private readonly ParticleGroup _g;
        /// <summary>The shared runtime parameters (Δt, stress scale, viscosity caps) the stages read as 0-d leaves.</summary>
        private readonly SolverParams _p;
        /// <summary>The group's material (its constants are baked into the trees).</summary>
        private readonly MaterialDef _m;
        /// <summary>Particle mass m = ρ·V₀ (the APIC term m·C).</summary>
        private readonly float _mass;
        /// <summary>The compiled stages in evaluation order, each with the view it writes.</summary>
        private readonly List<(CompiledExpression Kernel, NDArray Out)> _stages = new List<(CompiledExpression, NDArray)>();
        /// <summary>Plain copies run after the stages (the corotated models' T → F).</summary>
        private readonly List<(NDArray Dst, NDArray Src)> _copies = new List<(NDArray, NDArray)>();
        /// <summary>Particle count and buffer generation the stages were compiled over; a change triggers a rebuild.</summary>
        private int _builtCount = -1, _builtGeneration = -1;

        /// <summary>Creates the kernel set for a group.</summary>
        /// <param name="group">The particle group.</param>
        /// <param name="parameters">Shared runtime parameters.</param>
        /// <param name="particleMass">Mass of one particle of this material (V₀·ρ).</param>
        public ConstitutiveKernels(ParticleGroup group, SolverParams parameters, float particleMass)
        {
            _g = group;
            _p = parameters;
            _m = group.Material;
            _mass = particleMass;
        }

        /// <summary>Number of fused kernels one substep runs for this material (for the HUD / docs).</summary>
        public int StageCount => _stages.Count + _copies.Count;

        /// <summary>
        /// Runs the constitutive update for every particle of the group: advances the deformation state by
        /// the APIC velocity gradient C, applies the material's plasticity, and writes the affine momentum
        /// matrix A = stress term + m·C that the particle-to-grid transfer scatters.
        /// </summary>
        public void Apply()
        {
            if (_g.Count == 0) return;
            if (_g.Count != _builtCount || _g.Generation != _builtGeneration) Build();
            foreach (var (kernel, output) in _stages) kernel.Evaluate(output);
            foreach (var (dst, src) in _copies) np.copyto(dst, src);
        }

        /// <summary>(Re)compiles the stage list over the group's current views.</summary>
        private void Build()
        {
            _stages.Clear();
            _copies.Clear();
            _g.PrepareViews();
            switch (_m.Model)
            {
                case ConstitutiveModel.Fluid: BuildFluid(); break;
                case ConstitutiveModel.Corotated: BuildCorotated(); break;
                case ConstitutiveModel.SnowPlastic: BuildSnow(); break;
                case ConstitutiveModel.DruckerPrager: BuildSandOrClay(clay: false); break;
                case ConstitutiveModel.VonMises: BuildSandOrClay(clay: true); break;
                default: throw new NotSupportedException(_m.Model.ToString());
            }
            _builtCount = _g.Count;
            _builtGeneration = _g.Generation;
        }

        /// <summary>Compiles one stage writing into <paramref name="output"/>.</summary>
        /// <param name="expr">The tree (over this group's views and the shared parameters).</param>
        /// <param name="output">Destination view (may alias an input — elementwise in place is safe).</param>
        private void Stage(NDExpr expr, NDArray output) => _stages.Add((expr.Compile(), output));

        // Leaf helpers — views of the group's live range.
        /// <summary>Leaf for a persistent field's live view.</summary>
        /// <param name="f">Field.</param>
        /// <returns>The expression leaf.</returns>
        private NDExpr V(PField f) => Arr(_g[f]);
        /// <summary>Leaf for a scratch field's live view.</summary>
        /// <param name="s">Scratch field.</param>
        /// <returns>The expression leaf.</returns>
        private NDExpr V(SField s) => Arr(_g[s]);

        /// <summary>
        /// Trial deformation gradient T = (I + Δt·C)·F → T0..T3. Four stages, because every output component
        /// reads all four old components (an in-place update would read half-updated values).
        /// </summary>
        private void BuildTrialF()
        {
            NDExpr dt = Arr(_p.Dt);
            NDExpr c00 = V(PField.C00), c01 = V(PField.C01), c10 = V(PField.C10), c11 = V(PField.C11);
            NDExpr f00 = V(PField.F00), f01 = V(PField.F01), f10 = V(PField.F10), f11 = V(PField.F11);
            Stage((1.0 + dt * c00) * f00 + dt * c01 * f10, _g[SField.T0]);
            Stage((1.0 + dt * c00) * f01 + dt * c01 * f11, _g[SField.T1]);
            Stage(dt * c10 * f00 + (1.0 + dt * c11) * f10, _g[SField.T2]);
            Stage(dt * c10 * f01 + (1.0 + dt * c11) * f11, _g[SField.T3]);
        }

        /// <summary>
        /// The closed-form 2×2 SVD of T in σ-space (see the class remarks): E0 = eigen-gap D, S0 = σ₁,
        /// S1 = σ₂ (signed), S2 = cos 2φ, S3 = sin 2φ.
        /// </summary>
        private void BuildSigmaSpace()
        {
            NDExpr t0 = V(SField.T0), t1 = V(SField.T1), t2 = V(SField.T2), t3 = V(SField.T3);
            NDExpr p = t0 * t0 + t1 * t1, r = t2 * t2 + t3 * t3, q = t0 * t2 + t1 * t3;
            Stage(Sqrt(Square(p - r) + 4.0 * q * q), _g[SField.E0]);
            NDExpr d = V(SField.E0);
            Stage(Sqrt(Maximum(0.5 * (p + r + d), 1e-12)), _g[SField.S0]);
            Stage((t0 * t3 - t1 * t2) / V(SField.S0), _g[SField.S1]);
            Stage((p - r) / (d + 1e-12), _g[SField.S2]);
            Stage(2.0 * q / (d + 1e-12), _g[SField.S3]);
        }

        /// <summary>σ₂ kept away from zero (sign preserved) so the ratio σ₂′/σ₂ stays finite for a collapsed particle.</summary>
        private NDExpr SafeSigma2 => CopySign(Maximum(Abs(V(SField.S1)), 1e-6), V(SField.S1));

        /// <summary>
        /// Plastic return in σ-space: given k₁ = σ₁′/σ₁ and k₂ = σ₂′/σ₂, writes F = U·diag(k)·Uᵀ·T through
        /// h (E2) and g (E3).
        /// </summary>
        /// <param name="k1">Expression for σ₁′/σ₁.</param>
        /// <param name="k2">Expression for σ₂′/σ₂.</param>
        private void BuildPlasticReturn(NDExpr k1, NDExpr k2)
        {
            Stage(0.5 * (k1 + k2), _g[SField.E2]);
            Stage(0.5 * (k1 - k2), _g[SField.E3]);
            NDExpr h = V(SField.E2), g = V(SField.E3), c2 = V(SField.S2), s2 = V(SField.S3);
            NDExpr t0 = V(SField.T0), t1 = V(SField.T1), t2 = V(SField.T2), t3 = V(SField.T3);
            Stage((h + g * c2) * t0 + g * s2 * t2, _g[PField.F00]);
            Stage((h + g * c2) * t1 + g * s2 * t3, _g[PField.F01]);
            Stage(g * s2 * t0 + (h - g * c2) * t2, _g[PField.F10]);
            Stage(g * s2 * t1 + (h - g * c2) * t3, _g[PField.F11]);
        }

        /// <summary>
        /// Affine momentum from a Kirchhoff stress given in σ-space as τ = ht·I + gt·[[c2, s2],[s2, −c2]]:
        /// A = stressScale·τ + m·C.
        /// </summary>
        /// <param name="ht">Isotropic part (t₁+t₂)/2.</param>
        /// <param name="gt">Deviatoric part (t₁−t₂)/2.</param>
        private void BuildAffineFromSigmaStress(NDExpr ht, NDExpr gt)
        {
            NDExpr s = Arr(_p.StressScale);
            double m = _mass;
            NDExpr c2 = V(SField.S2), s2 = V(SField.S3);
            Stage(s * (ht + gt * c2) + m * V(PField.C00), _g[SField.A00]);
            Stage(s * (gt * s2) + m * V(PField.C01), _g[SField.A01]);
            Stage(s * (gt * s2) + m * V(PField.C10), _g[SField.A10]);
            Stage(s * (ht - gt * c2) + m * V(PField.C11), _g[SField.A11]);
        }

        /// <summary>
        /// Fluids (water, oil, honey, lava): the volume ratio J — which the previous particle update set to 1/φ
        /// from the gathered occupancy φ (see <see cref="Transfers"/>) — is advanced over the step the particle
        /// has just been advected through (J·(1 + Δt·tr C)) and clamped to [0.6, 1 + tension limit]; the
        /// Kirchhoff stress is the equation-of-state pressure K·J·(J−1) plus Newtonian viscosity μ·J·(C + Cᵀ).
        /// Lava's viscosity rises steeply as it cools.
        /// </summary>
        /// <remarks>
        /// The upper clamp is the liquid's cohesion: a free-surface or spray particle reads φ &lt; 1 (J &gt; 1), and
        /// the clamp turns that into at most K·(1+t)·t of tension — enough to hold sheets and droplets together,
        /// never enough to pull a free surface into clumps. The lower clamp bounds the pressure a violent impact
        /// can produce within one substep.
        /// </remarks>
        private void BuildFluid()
        {
            NDExpr dt = Arr(_p.Dt), s = Arr(_p.StressScale), mu = Arr(_p.Viscosity[(int)_m.Id]);
            NDExpr j = V(PField.Jp), c00 = V(PField.C00), c01 = V(PField.C01), c10 = V(PField.C10), c11 = V(PField.C11);
            Stage(Clamp(j * (1.0 + dt * (c00 + c11)), 0.6, 1.0 + _m.TensionLimit), _g[PField.Jp]);
            if (_m.Id == MaterialId.Lava)
            {
                // Cooling lava thickens: viscosity × (1 + 200·(1−T)²), capped at the explicit viscous
                // stability limit for the current Δt — molten lava runs, crusting lava crawls, and neither
                // can destabilize the substep.
                NDExpr cool = 1.0 - V(PField.Temp);
                mu = Minimum(mu * (1.0 + 200.0 * cool * cool), Arr(_p.ViscosityCap[(int)MaterialId.Lava]));
            }
            double m = _mass, k = _m.BulkModulus;
            NDExpr pressure = k * j * (j - 1.0);
            NDExpr shear = mu * j * (c01 + c10);
            Stage(s * (pressure + 2.0 * mu * j * c00) + m * c00, _g[SField.A00]);
            Stage(s * shear + m * c01, _g[SField.A01]);
            Stage(s * shear + m * c10, _g[SField.A10]);
            Stage(s * (pressure + 2.0 * mu * j * c11) + m * c11, _g[SField.A11]);
        }

        /// <summary>
        /// Fixed-corotated elasticity (jelly, rubber, rock): τ = 2μ·(T−R)·Tᵀ + λ·J·(J−1)·I with R the polar
        /// rotation of T, which in 2-D is the normalized (T₀₀+T₁₁, T₁₀−T₀₁) — no SVD needed. No plasticity,
        /// so the trial gradient becomes the new F.
        /// </summary>
        private void BuildCorotated()
        {
            BuildTrialF();
            NDExpr t0 = V(SField.T0), t1 = V(SField.T1), t2 = V(SField.T2), t3 = V(SField.T3);
            // |(T₀₀+T₁₁, T₁₀−T₀₁)| once (E0 scratch), then both rotation components divide by it. Sqrt of the sum
            // of squares, not Hypot: the entries are O(1), and NumSharp's overflow-safe Hypot is ~19× slower.
            NDExpr ra = t0 + t3, rb = t2 - t1;
            Stage(Maximum(Sqrt(ra * ra + rb * rb), 1e-12), _g[SField.E0]);
            NDExpr rr = V(SField.E0);
            Stage((t0 + t3) / rr, _g[SField.S0]);
            Stage((t2 - t1) / rr, _g[SField.S1]);
            NDExpr co = V(SField.S0), si = V(SField.S1);
            NDExpr s = Arr(_p.StressScale);
            double m = _mass, twoMu = 2.0 * _m.Mu, la = _m.Lambda;
            NDExpr jj = t0 * t3 - t1 * t2;
            NDExpr vol = la * jj * (jj - 1.0);
            Stage(s * (twoMu * ((t0 - co) * t0 + (t1 + si) * t1) + vol) + m * V(PField.C00), _g[SField.A00]);
            Stage(s * (twoMu * ((t0 - co) * t2 + (t1 + si) * t3)) + m * V(PField.C01), _g[SField.A01]);
            Stage(s * (twoMu * ((t2 - si) * t0 + (t3 - co) * t1)) + m * V(PField.C10), _g[SField.A10]);
            Stage(s * (twoMu * ((t2 - si) * t2 + (t3 - co) * t3) + vol) + m * V(PField.C11), _g[SField.A11]);
            _copies.Add((_g[PField.F00], _g[SField.T0]));
            _copies.Add((_g[PField.F01], _g[SField.T1]));
            _copies.Add((_g[PField.F10], _g[SField.T2]));
            _copies.Add((_g[PField.F11], _g[SField.T3]));
        }

        /// <summary>
        /// Snow (Stomakhin et al. 2013): singular values clamped to [1−θc, 1+θs] (the excess is plastic flow
        /// recorded in Jp), stiffness hardened by exp(ξ·(1−Jp)) — compacted snow gets stronger, stretched snow
        /// crumbles — and fixed-corotated stress on the elastic part.
        /// </summary>
        private void BuildSnow()
        {
            BuildTrialF();
            BuildSigmaSpace();
            double lo = 1.0 - _m.SnowThetaC, hi = 1.0 + _m.SnowThetaS;
            NDExpr s1 = V(SField.S0), s2 = V(SField.S1);
            NDExpr s1c = Clamp(s1, lo, hi), s2c = Clamp(s2, lo, hi);
            // Plastic volume change goes into Jp (bounded so hardening stays inside the stable range).
            Stage(Clamp(V(PField.Jp) * Abs(s1 * s2) / (s1c * s2c), 0.3, 3.0), _g[PField.Jp]);
            BuildPlasticReturn(s1c / s1, s2c / SafeSigma2);
            // The hardening multiplier exp(ξ(1−Jp)) once, into the A00 scratch (free until the affine stage).
            Stage(Clamp(Exp(_m.SnowHardening * (1.0 - V(PField.Jp))), 0.1, _m.SnowHardeningMax), _g[SField.A00]);
            NDExpr hc = V(SField.A00);
            NDExpr mu = _m.Mu * hc, la = _m.Lambda * hc;
            NDExpr jj = s1c * s2c;
            NDExpr a1 = s1c * (s1c - 1.0), a2 = s2c * (s2c - 1.0);
            Stage(mu * (a1 + a2) + la * jj * (jj - 1.0), _g[SField.E2]);
            Stage(mu * (a1 - a2), _g[SField.E3]);
            BuildAffineFromSigmaStress(V(SField.E2), V(SField.E3));
        }

        /// <summary>
        /// Sand (Klár et al. 2016 Drucker–Prager) or clay (von Mises), both on the Hencky strain ε = ln σ with
        /// St. Venant–Kirchhoff-Hencky stress τ = U·(2μ·ε + λ·tr ε·I)·Uᵀ.
        /// <list type="bullet">
        /// <item><description><b>Sand:</b> in tension (tr ε ≥ 0) the grains separate — project to the cone tip
        /// ε = 0 (no stress). In compression the shear allowed is proportional to the pressure (friction):
        /// δγ = ‖ε̂‖ + β·tr ε with β = α(λ+μ)/μ and α = √(2/3)·2 sin φ/(3 − sin φ); if δγ &gt; 0 the deviatoric
        /// part shrinks by δγ.</description></item>
        /// <item><description><b>Clay:</b> the deviatoric strain magnitude is capped at the yield strain,
        /// pressure-independent — so it dents and keeps the dent.</description></item>
        /// </list>
        /// </summary>
        /// <param name="clay">True for the von Mises (clay) law, false for Drucker–Prager (sand).</param>
        private void BuildSandOrClay(bool clay)
        {
            BuildTrialF();
            BuildSigmaSpace();
            const double invSqrt2 = 0.70710678118654752;
            // Hencky strains ln σ₁, ln σ₂ once each (A00/A01 scratch): the projection below reads them ~10 times.
            Stage(Log(Maximum(V(SField.S0), 1e-6)), _g[SField.A00]);
            Stage(Log(Maximum(V(SField.S1), 1e-6)), _g[SField.A01]);
            NDExpr e1 = V(SField.A00);
            NDExpr e2 = V(SField.A01);
            NDExpr tr = e1 + e2, diff = e1 - e2;
            double mu = _m.Mu, la = _m.Lambda;
            if (!clay)
            {
                double sinPhi = Math.Sin(_m.FrictionAngleDeg * Math.PI / 180.0);
                double alpha = Math.Sqrt(2.0 / 3.0) * 2.0 * sinPhi / (3.0 - sinPhi);
                double beta = alpha * (la + mu) / mu;
                NDExpr dgamma = diff * invSqrt2 + beta * tr;
                Stage(Where(tr >= 0.0, 0.0, Where(dgamma > 0.0, e1 - dgamma * invSqrt2, e1)), _g[SField.E0]);
                Stage(Where(tr >= 0.0, 0.0, Where(dgamma > 0.0, e2 + dgamma * invSqrt2, e2)), _g[SField.E1]);
            }
            else
            {
                NDExpr capped = Minimum(diff, Math.Sqrt(2.0) * _m.YieldStrain);
                Stage(0.5 * (tr + capped), _g[SField.E0]);
                Stage(0.5 * (tr - capped), _g[SField.E1]);
            }
            NDExpr p1 = V(SField.E0), p2 = V(SField.E1);
            // Return ratios exp(ε′)/σ once each (A10/A11 scratch) — the plastic return reads each twice.
            Stage(Exp(p1) / V(SField.S0), _g[SField.A10]);
            Stage(Exp(p2) / SafeSigma2, _g[SField.A11]);
            BuildPlasticReturn(V(SField.A10), V(SField.A11));
            BuildAffineFromSigmaStress((mu + la) * (p1 + p2), mu * (p1 - p2));
        }
    }
}
