using System;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    /// TensorEngine.Threading — the unified, per-module-extensible threading configuration surface,
    /// its two invariants (env-var source of truth; process-scoped writes only), and the
    /// np.multithreading integration.
    /// </summary>
    [TestClass]
    public class TensorEngineThreadingTests : TestClass
    {
        [TestMethod]
        public void WellKnownKnobs_AreRegistered_WithTheirEnvVars()
        {
            TensorEngine.Threading.Get(TensorEngine.Threading.NumSharp).EnvVar.Should().Be("NUMSHARP_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.OpenBlas).EnvVar.Should().Be("OPENBLAS_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.OpenMp).EnvVar.Should().Be("OMP_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.Mkl).EnvVar.Should().Be("MKL_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.Blis).EnvVar.Should().Be("BLIS_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.NumExpr).EnvVar.Should().Be("NUMEXPR_NUM_THREADS");
            TensorEngine.Threading.Get(TensorEngine.Threading.VecLib).EnvVar.Should().Be("VECLIB_MAXIMUM_THREADS");
        }

        [TestMethod]
        public void UnknownKnob_Throws_ButTryGetDoesNot()
        {
            Assert.ThrowsExactly<ArgumentException>(() => TensorEngine.Threading.Get("no-such-knob"));
            TensorEngine.Threading.TryGet("no-such-knob", out var v).Should().BeFalse();
            v.Should().BeNull();
        }

        // The NumSharp knob's value IS MultiThread.MaxThreads, live in both directions.
        [TestMethod]
        public void NumSharpKnob_IsBoundLiveTo_MultiThreadMaxThreads()
        {
            try
            {
                TensorEngine.Threading.SetThreads(TensorEngine.Threading.NumSharp, 4);
                MultiThread.MaxThreads.Should().Be(4);
                TensorEngine.Threading.GetThreads(TensorEngine.Threading.NumSharp).Should().Be(4);

                // Reader is live — a direct write to MultiThread shows through the knob.
                MultiThread.MaxThreads = 6;
                TensorEngine.Threading.GetThreads(TensorEngine.Threading.NumSharp).Should().Be(6);
            }
            finally
            {
                np.multithreading(false);
                MultiThread.MaxThreads = 8;
                Environment.SetEnvironmentVariable("NUMSHARP_NUM_THREADS", null);
            }
        }

        // np.multithreading routes its thread cap through the surface (writes the process env too).
        [TestMethod]
        public void Multithreading_Integrates_WithThreadingSurface()
        {
            try
            {
                np.multithreading(true, 4);

                MultiThread.Enabled.Should().BeTrue();
                MultiThread.MaxThreads.Should().Be(4);
                TensorEngine.Threading.GetThreads(TensorEngine.Threading.NumSharp).Should().Be(4);
                Environment.GetEnvironmentVariable("NUMSHARP_NUM_THREADS").Should().Be("4");
            }
            finally
            {
                np.multithreading(false);
                MultiThread.MaxThreads = 8;
                Environment.SetEnvironmentVariable("NUMSHARP_NUM_THREADS", null);
            }
        }

        // Invariant 2: writes target the PROCESS scope only.
        [TestMethod]
        public void SetThreads_Writes_ProcessScopeOnly()
        {
            const string env = "NUMSHARP_TEST_PROCSCOPE";
            try
            {
                TensorEngine.Threading.Register("nst-procscope", env);
                TensorEngine.Threading.SetThreads("nst-procscope", 3);

                Environment.GetEnvironmentVariable(env, EnvironmentVariableTarget.Process).Should().Be("3");
                TensorEngine.Threading.GetThreads("nst-procscope").Should().Be(3);
            }
            finally
            {
                Environment.SetEnvironmentVariable(env, null);
            }
        }

        [TestMethod]
        public void SetThreads_Null_ClearsTheEnvVar()
        {
            const string env = "NUMSHARP_TEST_CLEAR";
            try
            {
                TensorEngine.Threading.Register("nst-clear", env);
                TensorEngine.Threading.SetThreads("nst-clear", 2);
                Environment.GetEnvironmentVariable(env).Should().Be("2");

                TensorEngine.Threading.SetThreads("nst-clear", null);
                Environment.GetEnvironmentVariable(env).Should().BeNull();
                TensorEngine.Threading.GetThreads("nst-clear").Should().BeNull();
            }
            finally
            {
                Environment.SetEnvironmentVariable(env, null);
            }
        }

        // Invariant 1: a pre-existing env var is the source of truth — it seeds the knob and is pushed
        // to the applier at registration.
        [TestMethod]
        public void SourceOfTruth_SeedsFromPreexistingEnv_AndPushesToApplier()
        {
            const string env = "NUMSHARP_TEST_SEED";
            try
            {
                Environment.SetEnvironmentVariable(env, "5");

                int? applied = null;
                var v = TensorEngine.Threading.Register("nst-seed", env, applier: n => applied = n);

                v.IsEnvSourced.Should().BeTrue();
                v.Threads.Should().Be(5);
                applied.Should().Be(5);
            }
            finally
            {
                Environment.SetEnvironmentVariable(env, null);
            }
        }

        // TrySetDefault must not override a value that came from the environment (source of truth),
        // nor one already set explicitly — but does apply to an untouched knob.
        [TestMethod]
        public void TrySetDefault_RespectsSourceOfTruth_AndExplicitSets()
        {
            const string seededEnv = "NUMSHARP_TEST_DEF_SEED";
            const string freshEnv = "NUMSHARP_TEST_DEF_FRESH";
            try
            {
                Environment.SetEnvironmentVariable(seededEnv, "7");
                TensorEngine.Threading.Register("nst-def-seed", seededEnv);
                TensorEngine.Threading.TrySetDefault("nst-def-seed", 1).Should().BeFalse();
                TensorEngine.Threading.GetThreads("nst-def-seed").Should().Be(7); // untouched

                TensorEngine.Threading.Register("nst-def-fresh", freshEnv);
                TensorEngine.Threading.TrySetDefault("nst-def-fresh", 2).Should().BeTrue();
                TensorEngine.Threading.GetThreads("nst-def-fresh").Should().Be(2);
                // Now explicitly set -> a second default is refused.
                TensorEngine.Threading.TrySetDefault("nst-def-fresh", 9).Should().BeFalse();
                TensorEngine.Threading.GetThreads("nst-def-fresh").Should().Be(2);
            }
            finally
            {
                Environment.SetEnvironmentVariable(seededEnv, null);
                Environment.SetEnvironmentVariable(freshEnv, null);
            }
        }

        // The per-module extension point: re-registering upgrades a knob with reader/applier, and a
        // subsequent SetThreads drives the applier.
        [TestMethod]
        public void Register_Upserts_ReaderAndApplier()
        {
            const string env = "NUMSHARP_TEST_UPSERT";
            try
            {
                TensorEngine.Threading.Register("nst-upsert", env);

                int? applied = null;
                TensorEngine.Threading.Register("nst-upsert", applier: n => applied = n);

                TensorEngine.Threading.SetThreads("nst-upsert", 4);
                applied.Should().Be(4);
            }
            finally
            {
                Environment.SetEnvironmentVariable(env, null);
            }
        }
    }
}
