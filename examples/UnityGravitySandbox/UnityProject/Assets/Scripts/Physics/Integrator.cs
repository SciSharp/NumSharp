using System;
using NumSharp;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// The time-integration scheme used to advance the system by one step. The choice is not cosmetic:
    /// it decides whether energy is conserved over long runs, which is the whole game in orbital
    /// mechanics. Exposed so the sandbox can let the player switch schemes and WATCH the difference.
    /// </summary>
    public enum IntegrationMethod
    {
        /// <summary>
        /// Symplectic (semi-implicit) Euler: first order, one force evaluation per step, and — crucially —
        /// symplectic, so energy oscillates within a bounded band instead of drifting away. The cheapest
        /// scheme that does not blow orbits up. Good when force evaluations are the bottleneck.
        /// </summary>
        SymplecticEuler,

        /// <summary>
        /// Leapfrog / velocity-Verlet in kick-drift-kick form: second order, time-reversible and
        /// symplectic, at one NEW force evaluation per step (the end-of-step acceleration is cached and
        /// reused as the next step's start). This is the default and the right choice for essentially
        /// every gravity scenario — it is what real N-body codes use for the same reason.
        /// </summary>
        Leapfrog,

        /// <summary>
        /// Classical fourth-order Runge–Kutta: very accurate per step (four force evaluations), but NOT
        /// symplectic, so energy drifts secularly — an orbit slowly spirals in or out over many periods.
        /// Included precisely so that contrast is visible: RK4 looks best on a short clip and worst on a
        /// long one. Do not use it for stability demonstrations.
        /// </summary>
        Rk4
    }

    /// <summary>
    /// Advances an <see cref="NBodyState"/> in time using a chosen <see cref="IntegrationMethod"/>, driving
    /// the whole update through vectorized NumSharp array arithmetic. The instance caches the acceleration
    /// at the current positions so leapfrog costs only one new force evaluation per step; that cache MUST
    /// be invalidated (via <see cref="Reset"/>) whenever the set of bodies changes underneath it.
    /// </summary>
    public sealed class Integrator
    {
        /// <summary>The scheme this integrator applies on each <see cref="Step"/>. May be changed live.</summary>
        public IntegrationMethod Method { get; set; }

        // The cached acceleration at the state's CURRENT positions, valid only while _primed is true and
        // only for the body set it was computed against. Reused by leapfrog as the start-of-step kick;
        // recomputed from scratch by RK4/Euler (they don't rely on it) but kept coherent for a later switch.
        private NDArray _accel;
        private bool _primed;

        /// <summary>Creates an integrator with the given scheme (default <see cref="IntegrationMethod.Leapfrog"/>).</summary>
        /// <param name="method">The integration scheme to use.</param>
        public Integrator(IntegrationMethod method = IntegrationMethod.Leapfrog)
        {
            Method = method;
        }

        /// <summary>
        /// Discards the cached acceleration. Call this after ANY change to the body set (spawn, remove,
        /// merge, scenario reset) or after editing the solver's force model — otherwise leapfrog would
        /// kick the new configuration with a stale acceleration computed for the old one, injecting a
        /// spurious impulse on the next step.
        /// </summary>
        public void Reset()
        {
            _accel = null;
            _primed = false;
        }

        /// <summary>
        /// Advances <paramref name="state"/> in place by <paramref name="dt"/> using <see cref="Method"/>.
        /// </summary>
        /// <param name="state">The system to advance; its <see cref="NBodyState.Positions"/> and
        /// <see cref="NBodyState.Velocities"/> are reassigned.</param>
        /// <param name="solver">The force model that supplies accelerations.</param>
        /// <param name="dt">The time step in the scenario's time unit. Should be positive and small
        /// relative to the fastest orbital period; a step too large will make even a symplectic scheme
        /// unstable.</param>
        /// <exception cref="ArgumentNullException"><paramref name="state"/> or <paramref name="solver"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="dt"/> is not a finite, non-zero number.</exception>
        public void Step(NBodyState state, GravitySolver solver, double dt)
        {
            if (state is null) throw new ArgumentNullException(nameof(state));
            if (solver is null) throw new ArgumentNullException(nameof(solver));
            if (!double.IsFinite(dt) || dt == 0.0) throw new ArgumentOutOfRangeException(nameof(dt), "dt must be finite and non-zero.");

            switch (Method)
            {
                case IntegrationMethod.SymplecticEuler: StepSymplecticEuler(state, solver, dt); break;
                case IntegrationMethod.Leapfrog: StepLeapfrog(state, solver, dt); break;
                case IntegrationMethod.Rk4: StepRk4(state, solver, dt); break;
                default: throw new ArgumentOutOfRangeException(nameof(Method), Method, "Unknown integration method.");
            }
        }

        /// <summary>
        /// Symplectic Euler: update velocity from the current acceleration, then position from the UPDATED
        /// velocity. Updating position with the new velocity (not the old one) is exactly what makes this
        /// variant symplectic rather than the energy-hemorrhaging forward Euler.
        /// </summary>
        private void StepSymplecticEuler(NBodyState state, GravitySolver solver, double dt)
        {
            var a = solver.Acceleration(state.Positions, state.Velocities, state.Masses);
            state.Velocities = state.Velocities + a * dt;
            state.Positions = state.Positions + state.Velocities * dt;   // uses the just-updated velocity
            // The cached acceleration is now stale (positions moved); leave it invalid for a clean handoff
            // if the caller later switches to leapfrog.
            _primed = false;
        }

        /// <summary>
        /// Leapfrog in kick-drift-kick form. The trailing acceleration is cached because it equals the
        /// acceleration at the new positions, i.e. the next step's leading kick — so steady stepping costs
        /// one force evaluation per step despite the scheme naming two.
        /// </summary>
        private void StepLeapfrog(NBodyState state, GravitySolver solver, double dt)
        {
            // Prime the cache on the first step (or after Reset) with the acceleration at the start point.
            if (!_primed)
            {
                _accel = solver.Acceleration(state.Positions, state.Velocities, state.Masses);
                _primed = true;
            }

            double half = 0.5 * dt;
            state.Velocities = state.Velocities + _accel * half;              // kick (half step)
            state.Positions = state.Positions + state.Velocities * dt;        // drift (full step)
            _accel = solver.Acceleration(state.Positions, state.Velocities, state.Masses); // force at new r
            state.Velocities = state.Velocities + _accel * half;             // kick (half step)
        }

        /// <summary>
        /// Classical RK4 over the first-order system <c>y' = (v, a(r, v))</c>. Four force evaluations at
        /// staged trial states are blended with the 1-2-2-1 weights. High local accuracy, no symplectic
        /// guarantee — energy drifts over long runs, which is the point of offering it alongside leapfrog.
        /// </summary>
        private void StepRk4(NBodyState state, GravitySolver solver, double dt)
        {
            var x0 = state.Positions;
            var v0 = state.Velocities;
            var m = state.Masses;

            // Stage 1: derivatives at the current state.
            var a1 = solver.Acceleration(x0, v0, m);
            var kx1 = v0;

            // Stage 2: derivatives at the half-step trial state advanced by stage-1 slopes.
            var x2 = x0 + kx1 * (0.5 * dt);
            var v2 = v0 + a1 * (0.5 * dt);
            var a2 = solver.Acceleration(x2, v2, m);
            var kx2 = v2;

            // Stage 3: another half-step trial state, now advanced by stage-2 slopes.
            var x3 = x0 + kx2 * (0.5 * dt);
            var v3 = v0 + a2 * (0.5 * dt);
            var a3 = solver.Acceleration(x3, v3, m);
            var kx3 = v3;

            // Stage 4: full-step trial state advanced by stage-3 slopes.
            var x4 = x0 + kx3 * dt;
            var v4 = v0 + a3 * dt;
            var a4 = solver.Acceleration(x4, v4, m);
            var kx4 = v4;

            // Weighted blend (Simpson-like 1,2,2,1)/6 of the four slope estimates.
            double sixth = dt / 6.0;
            state.Positions = x0 + (kx1 + kx2 * 2.0 + kx3 * 2.0 + kx4) * sixth;
            state.Velocities = v0 + (a1 + a2 * 2.0 + a3 * 2.0 + a4) * sixth;
            _primed = false;   // RK4 kept no reusable cache
        }
    }
}
