using System;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// The mutable dynamical state of an N-body gravitational system — every body's position,
    /// velocity and (fixed) mass — held as three NumSharp <see cref="NDArray"/>s so the whole system
    /// advances through vectorized array math instead of a per-body loop.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Layout contract (load-bearing).</b> <see cref="Positions"/> and <see cref="Velocities"/> are
    /// <c>(N, 3)</c> float64 arrays and <see cref="Masses"/> is <c>(N,)</c> float64, all row-aligned:
    /// row <c>i</c> of each is body <c>i</c>. The whole engine — <see cref="GravitySolver"/>,
    /// <see cref="Integrator"/>, <see cref="ConservedQuantities"/> — depends on that shape and on
    /// float64 (double) precision. Double is used deliberately: gravitational dynamics is chaotic and a
    /// symplectic integrator only demonstrably conserves energy when the arithmetic carries enough
    /// significant digits; float32 (Unity's native <c>Vector3</c>) visibly drifts within seconds. The
    /// Unity view layer down-casts to float only at the last moment, for rendering.
    /// </para>
    /// <para>
    /// <b>Mutability.</b> The state is mutated in place across a simulation step: the integrator
    /// REASSIGNS <see cref="Positions"/> and <see cref="Velocities"/> to freshly computed arrays each
    /// substep (NumSharp array ops return new arrays rather than writing in place, so reassignment is
    /// the idiom). <see cref="Masses"/> only changes on a structural edit (spawn / remove / merge),
    /// which rebuilds all three arrays. Treat an <see cref="NBodyState"/> as owned by a single
    /// <see cref="NBodyWorld"/>; it is not thread-safe.
    /// </para>
    /// </remarks>
    public sealed class NBodyState
    {
        /// <summary>
        /// Per-body positions as an <c>(N, 3)</c> float64 array; row <c>i</c> is body <c>i</c>'s
        /// <c>(x, y, z)</c>. Reassigned each integrator substep. Never null and never ragged.
        /// </summary>
        public NDArray Positions { get; internal set; }

        /// <summary>
        /// Per-body velocities as an <c>(N, 3)</c> float64 array, row-aligned with <see cref="Positions"/>.
        /// Reassigned each integrator substep.
        /// </summary>
        public NDArray Velocities { get; internal set; }

        /// <summary>
        /// Per-body masses as a <c>(N,)</c> float64 array, row-aligned with <see cref="Positions"/>.
        /// Constant during integration; only a structural edit rebuilds it. Units are scenario-defined
        /// (e.g. solar masses in the astronomer's unit system) and must be consistent with the
        /// gravitational constant carried by the <see cref="GravitySolver"/>.
        /// </summary>
        public NDArray Masses { get; internal set; }

        /// <summary>The number of bodies currently in the system (the leading dimension of every array).</summary>
        // NumSharp shapes are long-valued (NDArray can exceed int.MaxValue elements); an N-body system is
        // never that large, so narrowing to int here is safe and keeps the per-body index API idiomatic.
        public int Count => (int)Positions.shape[0];

        /// <summary>
        /// Wraps three pre-built, mutually consistent arrays as a state. Validates the shape contract up
        /// front because a mismatch would otherwise surface deep inside a broadcast in the solver as an
        /// opaque shape error mid-frame.
        /// </summary>
        /// <param name="positions">An <c>(N, 3)</c> array of positions. Cast to float64 if not already.</param>
        /// <param name="velocities">An <c>(N, 3)</c> array of velocities. Cast to float64 if not already.</param>
        /// <param name="masses">A <c>(N,)</c> array of masses. Cast to float64 if not already.</param>
        /// <exception cref="ArgumentNullException">Any argument is null.</exception>
        /// <exception cref="ArgumentException">
        /// The arrays are not 2-D <c>(N, 3)</c> / 1-D <c>(N,)</c>, or their body counts disagree, or
        /// <paramref name="positions"/> is empty (a zero-body system has no dynamics to simulate).
        /// </exception>
        public NBodyState(NDArray positions, NDArray velocities, NDArray masses)
        {
            if (positions is null) throw new ArgumentNullException(nameof(positions));
            if (velocities is null) throw new ArgumentNullException(nameof(velocities));
            if (masses is null) throw new ArgumentNullException(nameof(masses));

            // Validate BEFORE storing so the invariant the rest of the engine relies on holds by
            // construction — every downstream reshape/broadcast assumes exactly (N,3)/(N,3)/(N,).
            if (positions.ndim != 2 || positions.shape[1] != 3)
                throw new ArgumentException($"positions must be (N,3); got shape ({string.Join(",", positions.shape)}).", nameof(positions));
            if (velocities.ndim != 2 || velocities.shape[1] != 3)
                throw new ArgumentException($"velocities must be (N,3); got shape ({string.Join(",", velocities.shape)}).", nameof(velocities));
            if (masses.ndim != 1)
                throw new ArgumentException($"masses must be 1-D (N,); got shape ({string.Join(",", masses.shape)}).", nameof(masses));

            int n = (int)positions.shape[0];   // shapes are long-valued; an N-body count fits int
            if (n == 0)
                throw new ArgumentException("An N-body system needs at least one body.", nameof(positions));
            if (velocities.shape[0] != n || masses.shape[0] != n)
                throw new ArgumentException(
                    $"Row counts disagree: positions={n}, velocities={velocities.shape[0]}, masses={masses.shape[0]}.");

            // np.astype is a no-op when the dtype already matches, so this is free for double inputs and
            // a safety net for a scenario authored with an int/float literal array.
            Positions = positions.astype(NPTypeCode.Double);
            Velocities = velocities.astype(NPTypeCode.Double);
            Masses = masses.astype(NPTypeCode.Double);
        }

        /// <summary>
        /// Builds a state from plain managed arrays — the convenient entry point for hand-authored
        /// scenarios where you write out coordinates as C# literals rather than constructing NDArrays.
        /// </summary>
        /// <param name="positions">
        /// An <c>N x 3</c> rectangular array of <c>(x, y, z)</c> rows. The second dimension MUST be 3.
        /// </param>
        /// <param name="velocities">An <c>N x 3</c> rectangular array of velocity rows, row-aligned with <paramref name="positions"/>.</param>
        /// <param name="masses">A length-<c>N</c> array of masses, aligned with the rows.</param>
        /// <returns>A validated <see cref="NBodyState"/> ready to hand to an <see cref="NBodyWorld"/>.</returns>
        /// <exception cref="ArgumentNullException">Any argument is null.</exception>
        /// <exception cref="ArgumentException">The dimensions do not satisfy the <c>(N,3)/(N,3)/(N,)</c> contract.</exception>
        public static NBodyState FromArrays(double[,] positions, double[,] velocities, double[] masses)
        {
            if (positions is null) throw new ArgumentNullException(nameof(positions));
            if (velocities is null) throw new ArgumentNullException(nameof(velocities));
            if (masses is null) throw new ArgumentNullException(nameof(masses));

            // np.array over a rectangular double[,] yields the desired (N,3) float64 array directly; the
            // ctor re-validates, so this method only needs the null guards above.
            return new NBodyState(np.array(positions), np.array(velocities), np.array(masses));
        }

        /// <summary>
        /// Reads one body's position out of the array into three doubles. Intended for the view layer,
        /// which needs a handful of scalar reads per frame to place GameObjects; it is NOT how the
        /// physics moves bodies (that stays fully vectorized).
        /// </summary>
        /// <param name="index">The body row, in <c>[0, Count)</c>.</param>
        /// <returns>The body's position.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside <c>[0, Count)</c>.</exception>
        public Vector3d PositionOf(int index)
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            return new Vector3d(Positions.GetDouble(index, 0), Positions.GetDouble(index, 1), Positions.GetDouble(index, 2));
        }

        /// <summary>
        /// Reads one body's velocity into three doubles (view/inspection helper; see <see cref="PositionOf"/>).
        /// </summary>
        /// <param name="index">The body row, in <c>[0, Count)</c>.</param>
        /// <returns>The body's velocity.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside <c>[0, Count)</c>.</exception>
        public Vector3d VelocityOf(int index)
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            return new Vector3d(Velocities.GetDouble(index, 0), Velocities.GetDouble(index, 1), Velocities.GetDouble(index, 2));
        }

        /// <summary>Reads one body's mass.</summary>
        /// <param name="index">The body row, in <c>[0, Count)</c>.</param>
        /// <returns>The body's mass in the scenario's mass unit.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside <c>[0, Count)</c>.</exception>
        public double MassOf(int index)
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            return Masses.GetDouble(index);
        }

        /// <summary>
        /// Produces an independent deep copy — used to snapshot a scenario's initial conditions so a
        /// reset can restore them exactly. NumSharp arithmetic returns fresh arrays during stepping, so
        /// the live state would otherwise share no memory with its origin and the initial conditions
        /// would be irrecoverable.
        /// </summary>
        /// <returns>A new <see cref="NBodyState"/> whose arrays are copies, sharing no memory with this one.</returns>
        public NBodyState Clone() =>
            new NBodyState(Positions.copy(), Velocities.copy(), Masses.copy());
    }
}
