using System;

namespace NumSharp
{
    /// <summary>
    ///     Raised when an operation needs a pluggable compute backend — an
    ///     <see cref="NumSharp.Backends.IBlasBackend"/> assigned to
    ///     <see cref="NumSharp.Backends.TensorEngine.Blas"/> — and none is installed, or the installed
    ///     one declined these operands.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Derives from <see cref="NotSupportedException"/> so an existing
    ///     <c>catch (NotSupportedException)</c> keeps catching it, and also implements
    ///     <see cref="INumSharpException"/> like the rest of the house exceptions — the same dual shape
    ///     <see cref="ValueError"/> and <see cref="AxisError"/> use to sit under a BCL type AND the
    ///     NumSharp marker at once.
    ///     </para>
    ///     <para>
    ///     This is the GENERAL case. The concrete backend a caller is expected to reference is named by
    ///     the derived <see cref="OpenBlasMissingBackendException"/>.
    ///     </para>
    /// </remarks>
    public class MissingBackendException : NotSupportedException, INumSharpException
    {
        public MissingBackendException() { }

        public MissingBackendException(string message) : base(message) { }

        public MissingBackendException(string message, Exception innerException) : base(message, innerException) { }
    }
}
