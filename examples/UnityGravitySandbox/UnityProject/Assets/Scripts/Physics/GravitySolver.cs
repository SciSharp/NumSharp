using System;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// Which gravitational force law the <see cref="GravitySolver"/> evaluates. Selecting a model is the
    /// single knob that turns the sandbox from "textbook Newton" into "matches a measurable feature of
    /// reality" (Mercury's anomalous perihelion advance), so it is worth understanding the trade-off.
    /// </summary>
    public enum ForceModel
    {
        /// <summary>
        /// Pure Newtonian inverse-square gravity, <c>a = -G·Σ m_j (r_i-r_j)/|r_i-r_j|³</c>. Velocity-
        /// independent, so the leapfrog integrator is exactly symplectic and total energy is conserved
        /// to round-off over astronomically many orbits. This is the default and the one the verification
        /// harness pins to machine precision.
        /// </summary>
        Newtonian,

        /// <summary>
        /// Newtonian gravity plus the leading-order (1PN) general-relativistic correction
        /// <c>×(1 + 3h²/(c²r²))</c>, where <c>h</c> is a pair's specific angular momentum. This makes a
        /// tight eccentric orbit PRECESS at the Schwarzschild rate <c>Δϖ = 6πGM/(c²a(1-e²))</c> per orbit
        /// — the effect that famously accounts for the 43″/century in Mercury's orbit that Newton cannot.
        /// The correction is velocity-dependent, so the integrator is no longer strictly symplectic and
        /// energy is only approximately conserved; that is physically correct (a precessing orbit is not
        /// closed) and the drift stays tiny for realistic <c>c</c>. Use it for the Mercury scenario, not
        /// for long-term stability demos.
        /// </summary>
        RelativisticPrecession
    }

    /// <summary>
    /// Computes the gravitational acceleration on every body at once, as a single chain of NumSharp
    /// broadcasts and reductions — the piece of the engine that most directly demonstrates why an
    /// array library is the right tool for physics. The naive alternative is an <c>O(N²)</c> double loop
    /// over body pairs; here the same <c>N×N</c> interaction matrix is formed and reduced with vectorized
    /// operations, so the C# side issues a handful of array calls regardless of <c>N</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Softening.</b> The point-mass force diverges as two bodies coincide, which would blow the
    /// integrator up on a close encounter. A Plummer softening length <see cref="Softening"/> replaces
    /// <c>1/r²</c> with <c>1/(r²+ε²)</c>, capping the force at close range. It is a physical model choice,
    /// not a hack: the matching softened potential (used by <see cref="ConservedQuantities"/>) keeps the
    /// force conservative so energy is still conserved. Set it small relative to inter-body distances.
    /// </para>
    /// <para>
    /// <b>Complexity.</b> This is a direct (Barnes–Hut-free) all-pairs solver: memory and time are
    /// <c>O(N²)</c>. It is ideal up to a few hundred bodies — the regime an interactive sandbox lives in —
    /// and is deliberately NOT a galaxy-scale tree code.
    /// </para>
    /// </remarks>
    public sealed class GravitySolver
    {
        /// <summary>
        /// The gravitational constant in the scenario's unit system. It couples masses, distances and
        /// times, so it MUST match the units the scenario chose (e.g. <c>4π²</c> for AU / solar-mass /
        /// year, or <c>1</c> for the dimensionless figure-eight). Getting it inconsistent with the masses
        /// silently rescales time, not "breaks" anything, which is why it is worth stating per scenario.
        /// </summary>
        public double G { get; set; }

        /// <summary>
        /// The Plummer softening length <c>ε</c> in the scenario's distance unit. Larger values make close
        /// encounters gentler and the simulation more forgiving at the cost of fidelity; a value near zero
        /// reproduces exact point-mass gravity but demands a small time step near collisions.
        /// </summary>
        public double Softening { get; set; }

        /// <summary>The force law to evaluate. See <see cref="ForceModel"/> for the trade-off.</summary>
        public ForceModel Model { get; set; }

        /// <summary>
        /// The speed of light in the scenario's unit system, used ONLY by
        /// <see cref="ForceModel.RelativisticPrecession"/>. Its value sets how strong the relativistic
        /// correction is (the correction scales as <c>1/c²</c>); pick it so orbital speeds are a realistic
        /// small fraction of it. Ignored entirely under <see cref="ForceModel.Newtonian"/>.
        /// </summary>
        public double SpeedOfLight { get; set; } = double.PositiveInfinity;

        /// <summary>
        /// Creates a solver with an explicit configuration.
        /// </summary>
        /// <param name="g">Gravitational constant in the scenario's units (see <see cref="G"/>).</param>
        /// <param name="softening">Plummer softening length ε ≥ 0 (see <see cref="Softening"/>).</param>
        /// <param name="model">The force law (default Newtonian).</param>
        /// <param name="speedOfLight">Speed of light for the relativistic model; leave at +∞ for Newtonian.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="g"/> is not finite, or <paramref name="softening"/> is negative or not finite.</exception>
        public GravitySolver(double g, double softening, ForceModel model = ForceModel.Newtonian, double speedOfLight = double.PositiveInfinity)
        {
            if (!double.IsFinite(g)) throw new ArgumentOutOfRangeException(nameof(g), "G must be finite.");
            if (!(softening >= 0) || double.IsPositiveInfinity(softening)) throw new ArgumentOutOfRangeException(nameof(softening), "Softening must be finite and non-negative.");
            G = g;
            Softening = softening;
            Model = model;
            SpeedOfLight = speedOfLight;
        }

        /// <summary>
        /// Evaluates the acceleration on every body from the mutual gravity of all the others, returning a
        /// fresh <c>(N, 3)</c> array. This is a pure function of the inputs — it does not mutate them —
        /// which is what lets the integrator call it repeatedly on trial positions.
        /// </summary>
        /// <param name="positions">Current positions, <c>(N, 3)</c> float64.</param>
        /// <param name="velocities">
        /// Current velocities, <c>(N, 3)</c> float64. Used only by <see cref="ForceModel.RelativisticPrecession"/>;
        /// under Newtonian gravity the argument is ignored but still required so the signature is uniform.
        /// </param>
        /// <param name="masses">Body masses, <c>(N,)</c> float64.</param>
        /// <returns>Accelerations, a new <c>(N, 3)</c> float64 array row-aligned with <paramref name="positions"/>.</returns>
        /// <exception cref="ArgumentNullException">Any argument is null.</exception>
        public NDArray Acceleration(NDArray positions, NDArray velocities, NDArray masses)
        {
            if (positions is null) throw new ArgumentNullException(nameof(positions));
            if (velocities is null) throw new ArgumentNullException(nameof(velocities));
            if (masses is null) throw new ArgumentNullException(nameof(masses));

            int n = (int)positions.shape[0];   // NumSharp shapes are long-valued; a body count fits int

            // Form the full pairwise displacement tensor disp[i,j] = r_j - r_i by broadcasting the (N,1,3)
            // "source" positions against the (1,N,3) "target" positions. This one subtraction replaces the
            // inner O(N²) loop; disp points FROM body i TOWARD body j, the direction i is pulled.
            var ri = positions.reshape(n, 1, 3);
            var rj = positions.reshape(1, n, 3);
            var disp = rj - ri;                              // (N,N,3)

            // Squared separation per pair via a reduction over the coordinate axis; softened so the force
            // stays finite when two bodies nearly coincide. rawR2 (un-softened) is kept for the GR factor,
            // which needs the true geometric distance.
            var rawR2 = np.sum(disp * disp, axis: 2);        // (N,N)
            var softR2 = rawR2 + (Softening * Softening);    // (N,N)
            var invR3 = np.power(softR2, -1.5);              // (N,N): the 1/(r²+ε²)^{3/2} weight

            // The per-pair scalar weight that multiplies disp. Under Newton it is just invR3; under the
            // relativistic model it gains the (1 + 3h²/(c²r²)) factor per pair.
            NDArray weight = invR3;
            if (Model == ForceModel.RelativisticPrecession && double.IsFinite(SpeedOfLight))
                weight = invR3 * RelativisticFactor(disp, velocities, rawR2, n);

            // A body exerts no force on itself; the diagonal of the pairwise weight must be exactly zero.
            // We zero it AFTER any GR multiply because the GR factor's diagonal is a finite 1 (see
            // RelativisticFactor), and 0×anything-finite is 0 — but zeroing here also protects the plain
            // Newtonian path, whose invR3 diagonal is (0+ε²)^{-3/2}, a large but real self-term.
            np.fill_diagonal(weight, 0.0);

            // Fold each interacting mass m_j into the weight (scale column j by m_j), then contract with
            // disp over the "source" axis j: a[i] = G · Σ_j weight[i,j]·m_j·disp[i,j].
            var wm = weight * masses.reshape(1, n);          // (N,N), column j scaled by m_j
            var acc = np.sum(wm.reshape(n, n, 1) * disp, axis: 1) * G;   // (N,3)
            return acc;
        }

        /// <summary>
        /// Builds the per-pair relativistic correction factor <c>1 + 3·h²/(c²·r²)</c> as an <c>(N, N)</c>
        /// array, where <c>h²</c> is each pair's squared specific angular momentum
        /// <c>|Δr × Δv|²</c>. The cross product is avoided in favour of Lagrange's identity
        /// <c>|u×v|² = |u|²|v|² − (u·v)²</c>, which is both cheaper and lets the whole thing stay as three
        /// vectorized reductions.
        /// </summary>
        /// <param name="disp">The pairwise displacement tensor <c>(N,N,3)</c>, <c>r_j − r_i</c>.</param>
        /// <param name="velocities">Body velocities <c>(N,3)</c>.</param>
        /// <param name="rawR2">Un-softened squared separations <c>(N,N)</c> (reused, not recomputed).</param>
        /// <param name="n">The body count.</param>
        /// <returns>The correction factor per pair, <c>(N,N)</c>, with a finite <c>1</c> on the diagonal.</returns>
        private NDArray RelativisticFactor(NDArray disp, NDArray velocities, NDArray rawR2, int n)
        {
            // Relative velocity per pair, oriented consistently with disp (target minus source).
            var vi = velocities.reshape(n, 1, 3);
            var vj = velocities.reshape(1, n, 3);
            var dvel = vj - vi;                              // (N,N,3)

            var v2 = np.sum(dvel * dvel, axis: 2);           // (N,N) |Δv|²
            var uv = np.sum(disp * dvel, axis: 2);           // (N,N) Δr·Δv
            var h2 = rawR2 * v2 - uv * uv;                   // (N,N) |Δr×Δv|² by Lagrange identity

            // The denominator carries a tiny epsilon so the diagonal (rawR2 = 0) evaluates to 0/tiny = 0
            // rather than 0/0 = NaN; h² is also 0 on the diagonal, so the factor there is exactly 1 —
            // finite, and about to be multiplied by a zeroed weight anyway.
            double k = 3.0 / (SpeedOfLight * SpeedOfLight);
            var factor = 1.0 + (h2 / (rawR2 + 1e-300)) * k;  // (N,N)
            return factor;
        }
    }
}
