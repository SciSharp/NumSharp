using System;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>The materials the lab simulates. Values are stable indices into <see cref="Materials.All"/> (and the renderer's color tables).</summary>
    public enum MaterialId : byte
    {
        /// <summary>Weakly-compressible liquid, near-zero viscosity.</summary>
        Water = 0,
        /// <summary>A lighter, slightly viscous liquid — floats on water by buoyancy alone.</summary>
        Oil = 1,
        /// <summary>A dense, very viscous liquid that coils and folds.</summary>
        Honey = 2,
        /// <summary>A glowing, viscous liquid that cools (by age and on contact with water) into <see cref="Rock"/>.</summary>
        Lava = 3,
        /// <summary>Dry granular material — Drucker–Prager plasticity forms piles at its friction angle.</summary>
        Sand = 4,
        /// <summary>Compacting granular solid — plasticity with hardening (clumps, packs, fractures).</summary>
        Snow = 5,
        /// <summary>Soft, bouncy elastic solid.</summary>
        Jelly = 6,
        /// <summary>Stiffer elastic solid.</summary>
        Rubber = 7,
        /// <summary>Plastic solid (von Mises) — deforms permanently like modelling clay.</summary>
        Clay = 8,
        /// <summary>Stiff elastic solid produced when lava cools.</summary>
        Rock = 9,
    }

    /// <summary>How a material is drawn — which shading pathway of the renderer it takes.</summary>
    public enum MaterialCategory : byte
    {
        /// <summary>Continuous liquid surface (smoothed field, refraction, specular).</summary>
        Fluid,
        /// <summary>Individually visible grains (sprites with per-grain color).</summary>
        Granular,
        /// <summary>Continuous solid body (smoothed field, opaque/translucent shading).</summary>
        Solid,
    }

    /// <summary>The constitutive model — which stress/plasticity law the NumSharp kernels apply.</summary>
    public enum ConstitutiveModel : byte
    {
        /// <summary>Equation-of-state pressure from the volume ratio J = 1/φ (inverse local occupancy), plus Newtonian viscosity from the APIC velocity gradient.</summary>
        Fluid,
        /// <summary>Fixed-corotated hyperelasticity (no plasticity).</summary>
        Corotated,
        /// <summary>Stomakhin et al. 2013 snow: singular values clamped, plastic compaction hardens the material.</summary>
        SnowPlastic,
        /// <summary>Klár et al. 2016 Drucker–Prager sand on the Hencky (log) strain.</summary>
        DruckerPrager,
        /// <summary>Von Mises plasticity on the deviatoric Hencky strain (clay/plasticine).</summary>
        VonMises,
    }

    /// <summary>
    /// The physical and visual constants of one material. All physics values are in the lab's normalized
    /// units: the domain is 1 unit tall, density 1 = water, time in seconds; moduli are in density ×
    /// (units/s)² so the wave speed is <c>sqrt(modulus / density)</c> units per second.
    /// </summary>
    /// <remarks>
    /// The stiffness values are deliberately "film-real", not literal: MPM's explicit time step is bounded by
    /// the fastest elastic wave (<see cref="WaveSpeed"/>), so a physically stiff rock would force hundreds
    /// of substeps per frame. The chosen moduli keep every wave well above the flow speeds (so liquids stay
    /// nearly incompressible and solids read as solid) while the whole scene still runs in real time.
    /// </remarks>
    public sealed class MaterialDef
    {
        /// <summary>Identity.</summary>
        public MaterialId Id { get; init; }
        /// <summary>Display name.</summary>
        public string Name { get; init; } = "";
        /// <summary>Renderer pathway.</summary>
        public MaterialCategory Category { get; init; }
        /// <summary>Constitutive law.</summary>
        public ConstitutiveModel Model { get; init; }
        /// <summary>Density relative to water (sets particle mass, hence buoyancy).</summary>
        public float Density { get; init; } = 1f;
        /// <summary>Young's modulus (solids/granular).</summary>
        public float YoungsModulus { get; init; }
        /// <summary>Poisson's ratio (solids/granular).</summary>
        public float Poisson { get; init; } = 0.3f;
        /// <summary>Bulk modulus K of the fluid equation of state (fluids).</summary>
        public float BulkModulus { get; init; }
        /// <summary>Dynamic viscosity μ (fluids). Bounded by the explicit viscous limit — see <see cref="MaxStableViscosity"/>.</summary>
        public float Viscosity { get; init; }
        /// <summary>
        /// How far a fluid may stretch (J−1) before its pressure stops pulling back — the liquid's cohesion.
        /// </summary>
        /// <remarks>
        /// Keep it at or near zero. A liquid's J is its inverse occupancy (J = 1/φ), so EVERY free-surface particle
        /// (φ ≈ 0.6) sits at this limit and pulls on its neighbours with K·(1+t)·t until the surface layer has
        /// clumped back to full density — the particle-method tensile instability: a bumpy, crusty surface and
        /// over-packed clumps that kick droplets out. Liquids this size have negligible surface tension anyway; a
        /// few thousandths give viscous liquids (honey, lava) a rope-like coherence without the crust.
        /// </remarks>
        public float TensionLimit { get; init; }
        /// <summary>Drucker–Prager friction angle in degrees (sand) — the angle of repose of a pile.</summary>
        public float FrictionAngleDeg { get; init; }
        /// <summary>Snow: critical compression (singular values clamped below 1−θc).</summary>
        public float SnowThetaC { get; init; }
        /// <summary>Snow: critical stretch (singular values clamped above 1+θs).</summary>
        public float SnowThetaS { get; init; }
        /// <summary>Snow: hardening coefficient ξ in <c>exp(ξ(1−Jp))</c>.</summary>
        public float SnowHardening { get; init; }
        /// <summary>Snow: upper clamp on the hardening multiplier (keeps the stiffest state inside the stable time step).</summary>
        public float SnowHardeningMax { get; init; } = 2.5f;
        /// <summary>Clay: the deviatoric Hencky strain magnitude at which it yields (smaller = softer clay).</summary>
        public float YieldStrain { get; init; }
        /// <summary>
        /// Drift-flux mobility τ (s): how fast this material separates from a mixture by buoyancy — its particles
        /// move relative to the shared grid velocity at τ·(ρₚ − ρ_mix)/ρₚ·g. Zero for coherent solids, whose
        /// buoyancy acts on the whole body through the grid pressure instead.
        /// </summary>
        /// <remarks>
        /// Keep it SMALL. A blob that is resolved on the grid already rises or sinks as a whole — the grid pressure
        /// gradient is set by the surrounding liquid, so a lighter region accelerates upward on its own. The drift
        /// only has to un-mix what is mixed below the grid scale (a stirred emulsion, sand in suspension). Every
        /// blob's boundary layer also reads a mixed density (the kernel is 3 cells wide), so a large τ strips that
        /// layer off faster than the blob moves and a rising oil blob dissolves into a cloud.
        /// </remarks>
        public float DriftMobility { get; init; }
        /// <summary>How readily this liquid shows spray/foam when stretched (water 1, oil a little, viscous liquids none).</summary>
        public float Foaminess { get; init; }

        /// <summary>Base albedo (linear RGB) used by the renderer.</summary>
        public (float R, float G, float B) Color { get; init; }
        /// <summary>Per-particle random brightness variation (grains look like grains).</summary>
        public float ColorJitter { get; init; }
        /// <summary>Emitted light (lava glows).</summary>
        public float Emission { get; init; }
        /// <summary>How much light passes through (0 opaque … 1 clear): drives refraction and Beer–Lambert absorption in the renderer.</summary>
        public float Translucency { get; init; }
        /// <summary>Surface glossiness (0 matte … 1 mirror-like specular).</summary>
        public float Gloss { get; init; }

        /// <summary>Lamé μ (shear modulus) from E and ν.</summary>
        public float Mu => YoungsModulus / (2f * (1f + Poisson));
        /// <summary>Lamé λ from E and ν.</summary>
        public float Lambda => YoungsModulus * Poisson / ((1f + Poisson) * (1f - 2f * Poisson));

        /// <summary>
        /// The fastest elastic wave in this material (units/s) — pressure waves for fluids
        /// (<c>sqrt(K/ρ)</c>), P-waves for solids (<c>sqrt((λ+2μ)/ρ)</c>), including snow's maximum hardening.
        /// The solver's time step is <c>CFL·dx / max(WaveSpeed)</c> over the materials present.
        /// </summary>
        public float WaveSpeed
        {
            get
            {
                if (Model == ConstitutiveModel.Fluid) return MathF.Sqrt(BulkModulus / Density);
                float h = Model == ConstitutiveModel.SnowPlastic ? SnowHardeningMax : 1f;
                return MathF.Sqrt(h * (Lambda + 2f * Mu) / Density);
            }
        }

        /// <summary>
        /// The largest viscosity the explicit update can carry at time step <paramref name="dt"/> and cell size
        /// <paramref name="dx"/> (the viscous diffusion limit <c>ν·dt/dx² ≤ ~0.2</c>). The solver clamps to it so a
        /// very viscous honey never destabilizes a coarse grid.
        /// </summary>
        /// <param name="dt">Time step.</param>
        /// <param name="dx">Cell size.</param>
        /// <returns>The maximum dynamic viscosity.</returns>
        public float MaxStableViscosity(float dt, float dx) => 0.2f * Density * dx * dx / dt;
    }

    /// <summary>
    /// The material table. Every constant here was tuned in the running lab for a "2 m tank" feel at
    /// ~17 substeps per 60 fps frame: liquids near-incompressible (Mach ≲ 0.3), sand at a 35° repose angle,
    /// jelly soft enough to wobble visibly, rubber bouncy, clay dentable, snow packing into clumps.
    /// </summary>
    public static class Materials
    {
        /// <summary>All materials, index = <see cref="MaterialId"/>.</summary>
        public static readonly MaterialDef[] All =
        {
            new MaterialDef
            {
                Id = MaterialId.Water, Name = "Water", Category = MaterialCategory.Fluid, Model = ConstitutiveModel.Fluid,
                Density = 1.0f, BulkModulus = 1500f, Viscosity = 0.0f, TensionLimit = 0.0f,
                Color = (0.06f, 0.36f, 0.80f), ColorJitter = 0.0f,
                Translucency = 0.92f, Gloss = 1.0f,
                DriftMobility = 0.006f, Foaminess = 1.0f,
            },
            new MaterialDef
            {
                Id = MaterialId.Oil, Name = "Oil", Category = MaterialCategory.Fluid, Model = ConstitutiveModel.Fluid,
                Density = 0.72f, BulkModulus = 1100f, Viscosity = 0.02f, TensionLimit = 0.0f,
                Color = (0.50f, 0.27f, 0.035f), ColorJitter = 0.0f,
                Translucency = 0.55f, Gloss = 1.0f,
                DriftMobility = 0.006f, Foaminess = 0.25f,
            },
            new MaterialDef
            {
                Id = MaterialId.Honey, Name = "Honey", Category = MaterialCategory.Fluid, Model = ConstitutiveModel.Fluid,
                Density = 1.4f, BulkModulus = 2000f, Viscosity = 1.0f, TensionLimit = 0.002f,
                Color = (0.92f, 0.56f, 0.06f), ColorJitter = 0.02f,
                Translucency = 0.68f, Gloss = 0.95f,
                DriftMobility = 0.004f,
            },
            new MaterialDef
            {
                Id = MaterialId.Lava, Name = "Lava", Category = MaterialCategory.Fluid, Model = ConstitutiveModel.Fluid,
                Density = 1.8f, BulkModulus = 2400f, Viscosity = 0.01f, TensionLimit = 0.002f,
                Color = (1.0f, 0.32f, 0.04f), ColorJitter = 0.05f, Emission = 1.0f,
                Translucency = 0.0f, Gloss = 0.35f,
                DriftMobility = 0.004f,
            },
            new MaterialDef
            {
                Id = MaterialId.Sand, Name = "Sand", Category = MaterialCategory.Granular, Model = ConstitutiveModel.DruckerPrager,
                Density = 1.6f, YoungsModulus = 3500f, Poisson = 0.3f, FrictionAngleDeg = 35f,
                Color = (0.84f, 0.69f, 0.45f), ColorJitter = 0.14f,
                Translucency = 0.0f, Gloss = 0.05f,
                DriftMobility = 0.008f,
            },
            new MaterialDef
            {
                Id = MaterialId.Snow, Name = "Snow", Category = MaterialCategory.Granular, Model = ConstitutiveModel.SnowPlastic,
                // Light (floats on water) and soft: with hardening capped at 2x its P-wave speed stays ~55 units/s,
                // in line with the other materials, so snow does not dictate a smaller time step.
                Density = 0.4f, YoungsModulus = 550f, Poisson = 0.2f,
                SnowThetaC = 0.025f, SnowThetaS = 0.0075f, SnowHardening = 10f, SnowHardeningMax = 2.0f,
                Color = (0.93f, 0.96f, 1.0f), ColorJitter = 0.05f,
                Translucency = 0.0f, Gloss = 0.2f,
                DriftMobility = 0.008f,
            },
            new MaterialDef
            {
                Id = MaterialId.Jelly, Name = "Jelly", Category = MaterialCategory.Solid, Model = ConstitutiveModel.Corotated,
                Density = 1.0f, YoungsModulus = 1800f, Poisson = 0.25f,
                Color = (0.30f, 0.92f, 0.36f), ColorJitter = 0.0f,
                Translucency = 0.62f, Gloss = 0.92f,
            },
            new MaterialDef
            {
                Id = MaterialId.Rubber, Name = "Rubber", Category = MaterialCategory.Solid, Model = ConstitutiveModel.Corotated,
                Density = 1.2f, YoungsModulus = 3000f, Poisson = 0.3f,
                Color = (0.55f, 0.22f, 0.88f), ColorJitter = 0.0f,
                Translucency = 0.0f, Gloss = 0.65f,
            },
            new MaterialDef
            {
                Id = MaterialId.Clay, Name = "Clay", Category = MaterialCategory.Solid, Model = ConstitutiveModel.VonMises,
                Density = 1.5f, YoungsModulus = 3500f, Poisson = 0.3f, YieldStrain = 0.015f,
                Color = (0.72f, 0.36f, 0.22f), ColorJitter = 0.04f,
                Translucency = 0.0f, Gloss = 0.12f,
            },
            new MaterialDef
            {
                Id = MaterialId.Rock, Name = "Rock", Category = MaterialCategory.Solid, Model = ConstitutiveModel.Corotated,
                Density = 1.8f, YoungsModulus = 5000f, Poisson = 0.25f,
                // Basalt reflects ~6 % of light: an albedo stated in LINEAR light (0.2 encodes to a pale concrete grey
                // on screen). The jitter gives the vesicular, patchy look of fresh lava rock.
                Color = (0.075f, 0.068f, 0.064f), ColorJitter = 0.12f,
                Translucency = 0.0f, Gloss = 0.25f,
            },
        };

        /// <summary>Number of materials.</summary>
        public static int Count => All.Length;

        /// <summary>Looks up a material definition.</summary>
        /// <param name="id">Material id.</param>
        /// <returns>The definition.</returns>
        public static MaterialDef Get(MaterialId id) => All[(int)id];
    }
}
