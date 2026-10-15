using Python.Runtime;

// The classes of the tree carry Python's module names, and those are lower case. The compiler warns that
// a lower-case type name can become a keyword of a later C# version (CS8981); the names stay as Python has them.
#pragma warning disable CS8981

namespace Pythonic;

static partial class Python
{
    /// <summary>
    ///     <c>import gc</c>: Python's collector for reference cycles. Python frees most objects at once, when
    ///     their last reference goes away. Objects that refer to each other in a circle have no such moment:
    ///     they stay until this collector finds them.
    /// </summary>
    internal static class gc
    {
        /// <summary>The module. It belongs to the process: it is not disposed.</summary>
        private static PyObject Module => PythonRuntime.Import("gc");

        /// <summary><c>gc.collect(generation)</c>: runs the collector now and frees the unreachable cycles it finds.</summary>
        /// <param name="generation">
        ///     0 looks at the newest objects only and takes less than a millisecond. 2, Python's default,
        ///     looks at every object: about 25 milliseconds with scikit-learn imported.
        /// </param>
        /// <returns>The number of unreachable objects that the collector found.</returns>
        /// <exception cref="PythonInteropException">No usable Python was found.</exception>
        /// <exception cref="PythonException">The generation is not 0, 1 or 2.</exception>
        public static long collect(int generation = 2)
        {
            using Py.GILState gil = PythonRuntime.GIL();
            using PyObject found = Call(Module, "collect", [generation]);
            return found.As<long>();
        }
    }
}
