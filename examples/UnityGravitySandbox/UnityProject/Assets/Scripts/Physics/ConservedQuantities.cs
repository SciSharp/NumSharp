using System;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// A snapshot of a system's conserved (or nearly conserved) quantities at one instant. Displaying
    /// these live is what turns the sandbox from a pretty animation into a physics instrument: a correct
    /// Newtonian simulation holds total energy, linear momentum and angular momentum constant, so watching
    /// them NOT drift is direct evidence the integrator and force law are faithful.
    /// </summary>
    public readonly struct SystemDiagnostics
    {
        /// <summary>Total kinetic energy <c>½ Σ mᵢ|vᵢ|²</c>.</summary>
        public readonly double Kinetic;
        /// <summary>Total (softened) gravitational potential energy <c>−G Σ_{i&lt;j} mᵢmⱼ/√(rᵢⱼ²+ε²)</c>, always ≤ 0.</summary>
        public readonly double Potential;
        /// <summary>Total mechanical energy <c>Kinetic + Potential</c> — the headline conserved quantity.</summary>
        public readonly double Total;
        /// <summary>Total linear momentum <c>Σ mᵢvᵢ</c>; conserved exactly (no external forces).</summary>
        public readonly Vector3d LinearMomentum;
        /// <summary>Total angular momentum about the origin <c>Σ mᵢ(rᵢ×vᵢ)</c>; conserved exactly.</summary>
        public readonly Vector3d AngularMomentum;
        /// <summary>The center of mass <c>(Σ mᵢrᵢ)/Σmᵢ</c>; moves at constant velocity (a straight line).</summary>
        public readonly Vector3d CenterOfMass;
        /// <summary>The total mass <c>Σ mᵢ</c>; changes only when bodies merge.</summary>
        public readonly double TotalMass;

        /// <summary>Bundles the seven computed quantities. Normally produced by <see cref="ConservedQuantities.Compute"/>.</summary>
        /// <param name="kinetic">Kinetic energy.</param>
        /// <param name="potential">Potential energy.</param>
        /// <param name="linearMomentum">Total linear momentum.</param>
        /// <param name="angularMomentum">Total angular momentum about the origin.</param>
        /// <param name="centerOfMass">Center of mass.</param>
        /// <param name="totalMass">Total mass.</param>
        public SystemDiagnostics(double kinetic, double potential, Vector3d linearMomentum,
                                 Vector3d angularMomentum, Vector3d centerOfMass, double totalMass)
        {
            Kinetic = kinetic;
            Potential = potential;
            Total = kinetic + potential;
            LinearMomentum = linearMomentum;
            AngularMomentum = angularMomentum;
            CenterOfMass = centerOfMass;
            TotalMass = totalMass;
        }
    }

    /// <summary>
    /// Computes the conserved quantities of an N-body system with vectorized NumSharp reductions. Every
    /// quantity here is a whole-array operation, mirroring how the solver evaluates forces — the same
    /// broadcasting that computes pairwise forces computes pairwise potential energy.
    /// </summary>
    public static class ConservedQuantities
    {
        /// <summary>
        /// Computes all diagnostics for a state in one pass. The potential energy is evaluated with the
        /// SAME softening the solver uses, which is essential: the softened force is conservative only
        /// with respect to the matching softened potential, so measuring energy with the wrong ε would
        /// show a false drift.
        /// </summary>
        /// <param name="state">The system to measure.</param>
        /// <param name="solver">The force model, read only for its <see cref="GravitySolver.G"/> and
        /// <see cref="GravitySolver.Softening"/> so the potential matches the force.</param>
        /// <returns>A filled <see cref="SystemDiagnostics"/> snapshot.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="state"/> or <paramref name="solver"/> is null.</exception>
        public static SystemDiagnostics Compute(NBodyState state, GravitySolver solver)
        {
            if (state is null) throw new ArgumentNullException(nameof(state));
            if (solver is null) throw new ArgumentNullException(nameof(solver));

            double ke = KineticEnergy(state);
            double pe = PotentialEnergy(state, solver.G, solver.Softening);
            Vector3d p = LinearMomentum(state);
            Vector3d l = AngularMomentum(state);
            Vector3d com = CenterOfMass(state, out double totalMass);
            return new SystemDiagnostics(ke, pe, p, l, com, totalMass);
        }

        /// <summary>
        /// Total kinetic energy <c>½ Σ mᵢ|vᵢ|²</c> via a per-body speed-squared reduction folded against
        /// the masses.
        /// </summary>
        /// <param name="state">The system.</param>
        /// <returns>The kinetic energy, ≥ 0.</returns>
        public static double KineticEnergy(NBodyState state)
        {
            var speed2 = np.sum(state.Velocities * state.Velocities, axis: 1);   // (N,) |vᵢ|²
            return (double)np.sum(state.Masses * speed2) * 0.5;
        }

        /// <summary>
        /// Total softened gravitational potential energy <c>−G Σ_{i&lt;j} mᵢmⱼ/√(rᵢⱼ²+ε²)</c>. Built from
        /// the same pairwise displacement tensor the solver uses; the full symmetric matrix is summed and
        /// halved (rather than explicitly taking the upper triangle) because the diagonal is zeroed and
        /// each unordered pair then appears exactly twice.
        /// </summary>
        /// <param name="state">The system.</param>
        /// <param name="g">Gravitational constant (must match the solver).</param>
        /// <param name="softening">Softening length ε (must match the solver).</param>
        /// <returns>The potential energy, ≤ 0.</returns>
        public static double PotentialEnergy(NBodyState state, double g, double softening)
        {
            int n = state.Count;
            var ri = state.Positions.reshape(n, 1, 3);
            var rj = state.Positions.reshape(1, n, 3);
            var disp = rj - ri;                                       // (N,N,3)
            var r2 = np.sum(disp * disp, axis: 2) + (softening * softening);  // (N,N)
            var invR = np.power(r2, -0.5);                            // (N,N) 1/√(r²+ε²)

            // Outer product of masses gives mᵢmⱼ for every pair; the pairwise energy is −G·mᵢmⱼ/r.
            var mimj = state.Masses.reshape(n, 1) * state.Masses.reshape(1, n);  // (N,N)
            var pairEnergy = mimj * invR * (-g);                     // (N,N)
            np.fill_diagonal(pairEnergy, 0.0);                       // drop self-energy (i==j)

            // The matrix is symmetric with a zeroed diagonal, so every {i,j} pair is counted twice;
            // halving the full sum yields the Σ_{i<j} we want without materializing a triangle.
            return (double)np.sum(pairEnergy) * 0.5;
        }

        /// <summary>Total linear momentum <c>Σ mᵢvᵢ</c>.</summary>
        /// <param name="state">The system.</param>
        /// <returns>The momentum vector.</returns>
        public static Vector3d LinearMomentum(NBodyState state)
        {
            int n = state.Count;
            var p = np.sum(state.Masses.reshape(n, 1) * state.Velocities, axis: 0);  // (3,)
            return new Vector3d(p.GetDouble(0), p.GetDouble(1), p.GetDouble(2));
        }

        /// <summary>
        /// Total angular momentum about the origin <c>Σ mᵢ(rᵢ×vᵢ)</c>. The per-body cross products are one
        /// vectorized <see cref="np.cross(NDArray, NDArray, int, int, int, int)"/> over the whole
        /// <c>(N,3)</c> arrays; only the final mass-weighted sum crosses back to a scalar vector.
        /// </summary>
        /// <param name="state">The system.</param>
        /// <returns>The angular-momentum vector.</returns>
        public static Vector3d AngularMomentum(NBodyState state)
        {
            int n = state.Count;
            var cross = np.cross(state.Positions, state.Velocities);                 // (N,3) rᵢ×vᵢ
            var l = np.sum(state.Masses.reshape(n, 1) * cross, axis: 0);             // (3,)
            return new Vector3d(l.GetDouble(0), l.GetDouble(1), l.GetDouble(2));
        }

        /// <summary>Center of mass <c>(Σ mᵢrᵢ)/Σmᵢ</c>, and the total mass as a by-product.</summary>
        /// <param name="state">The system.</param>
        /// <param name="totalMass">Receives the total mass <c>Σ mᵢ</c>.</param>
        /// <returns>The center-of-mass position.</returns>
        public static Vector3d CenterOfMass(NBodyState state, out double totalMass)
        {
            int n = state.Count;
            totalMass = (double)np.sum(state.Masses);
            var weighted = np.sum(state.Masses.reshape(n, 1) * state.Positions, axis: 0);  // (3,)
            double inv = totalMass != 0 ? 1.0 / totalMass : 0.0;   // guard the (physically impossible) massless system
            return new Vector3d(weighted.GetDouble(0) * inv, weighted.GetDouble(1) * inv, weighted.GetDouble(2) * inv);
        }
    }
}
