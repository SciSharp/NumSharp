using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // The [NDScoped]/[NDScopedAsync] TARGET gate (COVERAGE_PLAN §4). Extends GateScenarios.cs with the
    // negative half the suite lacked — the SUPPORTED carriers that must stay CLEAN (no gate error) —
    // plus two more rejections. Gate diagnostics are ERRORS; this project's .editorconfig silences them
    // for the fixture build, and NumSharp.Tests.Build.Analyzer asserts them in-process. Tags sit on the
    // METHOD/PROPERTY declaration line (the symbol location the gate reports).
    public abstract class GateNegativeScenarios
    {
        // ---------------------------------------------------------------- MUST be gated (positives)

        [NDScoped]
        public static void RefArrayEgress(ref NDArray[] a)            // [NDW002]  ref NDArray[] is a hidden egress
            => a = null;

        [NDScoped]
        public static NDArray InEgress(in NDArray a)                  // [NDW002]  an `in` NDArray is a hidden egress too
            => a + 1.0;

        [NDScopedAsync]
        public static async Task<List<NDArray>> TaskOfUnsupported(NDArray a) // [NDW003]  List<NDArray> inside the Task is unsupported
        {
            await Task.Yield();
            return new List<NDArray> { a + 1.0 };
        }

        [NDScoped]
        public static List<NDArray> GetterUnsupported                 // [NDW003]  a getter returning List<NDArray>
            => new List<NDArray> { _a + 1.0 };

        private static NDArray _a;

        // ---------------------------------------------------------------- MUST stay CLEAN (negatives)

        // A 3-tuple of NDArrays is a supported carrier.
        [NDScoped]
        public static (NDArray, NDArray, NDArray) Tuple3(NDArray a)
            => (a + 1.0, a - 1.0, a * 2.0);

        // A 5-tuple rides the ITuple path — still supported.
        [NDScoped]
        public static (NDArray, NDArray, NDArray, NDArray, NDArray) Tuple5(NDArray a)
            => (a + 1.0, a - 1.0, a * 2.0, a / 2.0, a + 3.0);

        // A bare IArraySlice is a supported carrier.
        [NDScoped]
        public static IArraySlice SliceCarrier(NDArray a) => default;

        // NDArray<T> (a subclass) is a supported carrier.
        [NDScoped]
        public static NumSharp.Generic.NDArray<double> GenericCarrier(NDArray a)
            => (a + 1.0).MakeGeneric<double>();

        // Task<NDArray[]> under [NDScopedAsync] — a supported task-of-carrier shape.
        [NDScopedAsync]
        public static async Task<NDArray[]> TaskOfArray(NDArray a)
        {
            await Task.Yield();
            return new[] { a + 1.0 };
        }

        // An async iterator of NDArrays.
        [NDScopedAsync]
        public static async IAsyncEnumerable<NDArray> AsyncEnumerable(NDArray a)
        {
            await Task.Yield();
            yield return a + 1.0;
        }

        // A synchronous iterator under [NDScoped] must NOT be mistaken for a wrong-attribute (NDW010).
        [NDScoped]
        public static IEnumerable<NDArray> SyncEnumerable(NDArray a)
        {
            yield return a + 1.0;
        }
    }
}
