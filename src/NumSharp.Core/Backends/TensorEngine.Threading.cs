using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     Process-wide threading configuration — one named, environment-variable-backed knob per
        ///     threading domain (NumSharp's own kernels, and the native BLAS / OpenMP runtimes NumPy and
        ///     the rest of the ecosystem thread through: OpenBLAS, MKL, BLIS, NumExpr, vecLib). It is the
        ///     single surface for reading and setting them, and it is <b>extensible per module</b>: a
        ///     module registers (or upgrades) its own knob via <see cref="Register"/> — that is how
        ///     <c>NumSharp.Interop.OpenBLAS</c> attaches a native applier to the <see cref="OpenBlas"/>
        ///     knob so a change reaches the loaded library, and how any future backend adds its own.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>Two invariants govern every knob.</b>
        ///     </para>
        ///     <para>
        ///     <b>1. Source of truth.</b> A variable already present in the environment when the knob is
        ///     first observed is authoritative: its value seeds the knob (<see cref="Variable.IsEnvSourced"/>
        ///     becomes true) and is pushed to the knob's applier. A module-supplied default never overwrites
        ///     it (see <see cref="TrySetDefault"/>). An explicit <see cref="SetThreads"/> / <see cref="SetAll"/>
        ///     is a deliberate in-process override and does take effect.
        ///     </para>
        ///     <para>
        ///     <b>2. Writes only apply to the process.</b> Every write this class performs targets
        ///     <see cref="EnvironmentVariableTarget.Process"/> — the caller's persistent (User / Machine)
        ///     environment is never modified.
        ///     </para>
        ///     <para>
        ///     <b>Reaching a native runtime.</b> Core is 100 % managed and can only write the managed
        ///     process-env table. A native library reads a variable through its own C runtime's
        ///     <c>getenv</c>, which a managed write does NOT reach (documented on <c>OpenBlasEngine</c>);
        ///     and most native BLAS read their <c>*_NUM_THREADS</c> variable once, at load. So for the
        ///     native knobs the Core write is a best-effort managed-table update for diagnostics and child
        ///     processes; the value only takes hold in the running library when a module has registered an
        ///     applier that pushes it there (through the CRT and the runtime's own set-thread-count call).
        ///     </para>
        /// </remarks>
        public static class Threading
        {
            /// <summary>NumSharp's own multithreaded kernels — the knob whose value is
            /// <see cref="Backends.MultiThread.MaxThreads"/>. Enabling/disabling those kernels is the
            /// companion <c>NUMSHARP_MULTITHREADING</c> / <see cref="np.multithreading(bool,int)"/> switch
            /// on <see cref="Backends.MultiThread.Enabled"/> — a bool, not a thread count, so it is not a
            /// knob here.</summary>
            public const string NumSharp = "NumSharp";

            /// <summary>OpenBLAS (<c>OPENBLAS_NUM_THREADS</c>). Core only touches the managed env table;
            /// referencing <c>NumSharp.Interop.OpenBLAS</c> upgrades this knob with a native applier.</summary>
            public const string OpenBlas = "OpenBLAS";

            /// <summary>OpenMP (<c>OMP_NUM_THREADS</c>) — the threading layer inside OpenBLAS / MKL / BLIS.</summary>
            public const string OpenMp = "OpenMP";

            /// <summary>Intel MKL (<c>MKL_NUM_THREADS</c>).</summary>
            public const string Mkl = "MKL";

            /// <summary>BLIS (<c>BLIS_NUM_THREADS</c>).</summary>
            public const string Blis = "BLIS";

            /// <summary>NumExpr (<c>NUMEXPR_NUM_THREADS</c>).</summary>
            public const string NumExpr = "NumExpr";

            /// <summary>Apple Accelerate / vecLib (<c>VECLIB_MAXIMUM_THREADS</c>).</summary>
            public const string VecLib = "vecLib";

            private static readonly object _gate = new object();

            private static readonly Dictionary<string, Variable> _vars =
                new Dictionary<string, Variable>(StringComparer.OrdinalIgnoreCase);

            static Threading()
            {
                // NumSharp's own kernels: the value IS MultiThread.MaxThreads (read and written live),
                // env-seeded from NUMSHARP_NUM_THREADS as the source of truth.
                Register(NumSharp, "NUMSHARP_NUM_THREADS",
                    reader: () => Backends.MultiThread.MaxThreads,
                    applier: v => { if (v.HasValue) Backends.MultiThread.MaxThreads = v.Value; });

                // The native BLAS / OpenMP runtimes. No Core applier — the value lives in the managed
                // env table until a module (NumSharp.Interop.OpenBLAS) re-registers with one that reaches
                // the native library. Registering them here means they are configurable and inspectable
                // out of the box, and a native module only has to add the applier, not the whole knob.
                Register(OpenBlas, "OPENBLAS_NUM_THREADS");
                Register(OpenMp, "OMP_NUM_THREADS");
                Register(Mkl, "MKL_NUM_THREADS");
                Register(Blis, "BLIS_NUM_THREADS");
                Register(NumExpr, "NUMEXPR_NUM_THREADS");
                Register(VecLib, "VECLIB_MAXIMUM_THREADS");
            }

            /// <summary>
            ///     Registers a threading knob, or upgrades an existing one of the same
            ///     <paramref name="name"/> (this is the per-module extension point — a module attaches its
            ///     <paramref name="reader"/> / <paramref name="applier"/> to a knob Core already declared).
            ///     A non-null argument replaces that facet; a null one leaves it unchanged.
            /// </summary>
            /// <param name="name">Logical knob name (one of the constants above, or a module's own).</param>
            /// <param name="envVar">Backing environment variable, or null for a value with no env backing.</param>
            /// <param name="reader">
            ///     Live value source. When set, <see cref="Variable.Threads"/> returns
            ///     <c>reader()</c> — used to bind a knob to live state (e.g. the managed
            ///     <see cref="Backends.MultiThread.MaxThreads"/>, or a native
            ///     <c>openblas_get_num_threads()</c>) instead of the cached value.
            /// </param>
            /// <param name="applier">
            ///     Invoked after every value change (and once, at registration, with a source-of-truth env
            ///     value) to push the count where it takes effect — a managed field, a native
            ///     <c>set_num_threads</c>, the CRT env, etc.
            /// </param>
            /// <returns>The registered (or upgraded) knob.</returns>
            public static Variable Register(string name, string envVar = null, Func<int?> reader = null, Action<int?> applier = null)
            {
                if (string.IsNullOrWhiteSpace(name))
                    throw new ArgumentException("Threading variable name must be non-empty.", nameof(name));

                Variable v;
                Action<int?> seedApplier = null;
                int? seedValue = null;

                lock (_gate)
                {
                    if (!_vars.TryGetValue(name, out v))
                    {
                        v = new Variable(name);
                        _vars[name] = v;
                    }

                    if (envVar != null) v.EnvVar = envVar;
                    if (reader != null) v.Reader = reader;
                    if (applier != null) v.Applier = applier;

                    // Source of truth: a value already in the environment seeds the knob and is pushed to
                    // the applier. It is NOT written back (it is already there) and never clobbered by a
                    // later default. Re-registration re-seeds, so a module attaching an applier receives
                    // the source-of-truth value.
                    if (v.EnvVar != null)
                    {
                        var parsed = ParseThreads(Environment.GetEnvironmentVariable(v.EnvVar));
                        if (parsed.HasValue)
                        {
                            v.Cached = parsed;
                            v.IsEnvSourced = true;
                            seedApplier = v.Applier;
                            seedValue = parsed;
                        }
                    }
                }

                // Push the source-of-truth value to the applier OUTSIDE the lock — an applier may call
                // into native code (or back into this class) and must not run under _gate.
                seedApplier?.Invoke(seedValue);

                return v;
            }

            /// <summary>The registered knob named <paramref name="name"/>.</summary>
            /// <exception cref="ArgumentException">No knob of that name is registered.</exception>
            public static Variable Get(string name)
            {
                if (TryGet(name, out var v))
                    return v;
                throw new ArgumentException(
                    $"Unknown threading variable '{name}'. Register it via TensorEngine.Threading.Register first.",
                    nameof(name));
            }

            /// <summary>The knob named <paramref name="name"/>, if one is registered.</summary>
            public static bool TryGet(string name, out Variable variable)
            {
                lock (_gate)
                    return _vars.TryGetValue(name ?? string.Empty, out variable);
            }

            /// <summary>A snapshot of every registered knob.</summary>
            public static IReadOnlyList<Variable> Variables
            {
                get { lock (_gate) return _vars.Values.ToList(); }
            }

            /// <summary>The current thread count of <paramref name="name"/> (null when unset / auto).</summary>
            public static int? GetThreads(string name) => Get(name).Threads;

            /// <summary>
            ///     Sets <paramref name="name"/> to <paramref name="threads"/> — a deliberate in-process
            ///     override. Writes the backing env var (process scope only), records the value, and runs
            ///     the applier. Passing null clears the env var / requests auto.
            /// </summary>
            public static void SetThreads(string name, int? threads) => Apply(Get(name), threads);

            /// <summary>
            ///     Applies <paramref name="threads"/> as a DEFAULT — honouring the source-of-truth rule:
            ///     does nothing (returns false) if the knob was seeded from the environment or has already
            ///     been set explicitly; otherwise behaves like <see cref="SetThreads"/> and returns true.
            ///     Modules use this to suggest a value without overriding what the user configured.
            /// </summary>
            public static bool TrySetDefault(string name, int? threads)
            {
                var v = Get(name);
                lock (_gate)
                {
                    if (v.IsEnvSourced || v.IsExplicitlySet)
                        return false;
                }

                Apply(v, threads);
                return true;
            }

            /// <summary>
            ///     Sets every registered knob to <paramref name="threads"/> — a deliberate override of all
            ///     (e.g. pin the whole process to one thread). Process-scoped writes, per knob.
            /// </summary>
            public static void SetAll(int threads)
            {
                foreach (var v in Variables)
                    Apply(v, threads);
            }

            private static void Apply(Variable v, int? threads)
            {
                Action<int?> applier;
                lock (_gate)
                {
                    if (v.EnvVar != null)
                        // Process scope ONLY — the persistent User / Machine environment is never touched.
                        Environment.SetEnvironmentVariable(
                            v.EnvVar,
                            threads?.ToString(CultureInfo.InvariantCulture),
                            EnvironmentVariableTarget.Process);

                    v.Cached = threads;
                    v.IsExplicitlySet = true;
                    applier = v.Applier;
                }

                applier?.Invoke(threads);
            }

            private static int? ParseThreads(string raw)
            {
                if (string.IsNullOrWhiteSpace(raw))
                    return null;
                return int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n)
                    ? n
                    : (int?)null;
            }

            /// <summary>
            ///     One threading knob: a logical name, an optional backing environment variable, and the
            ///     module hooks that read and apply its value. Read-only from outside; mutated only through
            ///     <see cref="Threading"/>.
            /// </summary>
            public sealed class Variable
            {
                internal Variable(string name) => Name = name;

                /// <summary>Logical knob name.</summary>
                public string Name { get; }

                /// <summary>Backing environment variable, or null for a knob with no env backing.</summary>
                public string EnvVar { get; internal set; }

                /// <summary>True when the value was seeded from a pre-existing env var (the source of truth).</summary>
                public bool IsEnvSourced { get; internal set; }

                /// <summary>True once the value has been set through the API (as opposed to only seeded).</summary>
                public bool IsExplicitlySet { get; internal set; }

                internal Func<int?> Reader;
                internal Action<int?> Applier;
                internal int? Cached;

                /// <summary>
                ///     The current thread count (null = unset / auto). A knob with a live
                ///     <see cref="Reader"/> reports that; otherwise an env-backed knob reports the env var's
                ///     current value (so an externally-changed env var stays authoritative); otherwise the
                ///     last value set through the API.
                /// </summary>
                public int? Threads
                {
                    get
                    {
                        if (Reader != null)
                            return Reader();

                        if (EnvVar != null)
                        {
                            var raw = Environment.GetEnvironmentVariable(EnvVar);
                            if (!string.IsNullOrWhiteSpace(raw) &&
                                int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n))
                                return n;
                        }

                        return Cached;
                    }
                }

                public override string ToString() =>
                    $"{Name}={(Threads?.ToString(CultureInfo.InvariantCulture) ?? "auto")}"
                    + (EnvVar != null ? $" [{EnvVar}]" : string.Empty)
                    + (IsEnvSourced ? " (env)" : string.Empty);
            }
        }
    }
}
