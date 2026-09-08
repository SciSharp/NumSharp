using System;

namespace NumSharp
{
    /// <summary>
    ///     Flags describing a <see cref="DTypeMeta"/> — the DType CLASS — mirroring NumPy's <c>NPY_DT_*</c> bits
    ///     (<c>dtypemeta.h</c> / <c>dtype_api.h</c>): <c>NPY_DT_LEGACY = 1</c>, <c>NPY_DT_ABSTRACT = 2</c>,
    ///     <c>NPY_DT_PARAMETRIC = 4</c>, <c>NPY_DT_NUMERIC = 8</c>.
    /// </summary>
    [Flags]
    public enum DTypeFlags
    {
        /// <summary>No flags.</summary>
        None = 0,

        /// <summary>
        ///     A "legacy" dtype in NumPy's sense: one of the builtin classes whose descriptor layout predates NEP 41/42
        ///     (every storage-backed NumSharp dtype and the datetime pair carry it; NEP 55's <c>StringDType</c> would not).
        /// </summary>
        Legacy = 1 << 0,

        /// <summary>
        ///     An abstract DType: it has no instances and exists only to participate in promotion — the NEP 50
        ///     Python-scalar classes <c>_PyLongDType</c> / <c>_PyFloatDType</c> / <c>_PyComplexDType</c>.
        /// </summary>
        Abstract = 1 << 1,

        /// <summary>
        ///     A parametric DType: its instances carry parameters beyond the class (a datetime unit, a string length), so
        ///     <c>promote_types</c> must run <c>common_instance</c> instead of returning the class default.
        /// </summary>
        Parametric = 1 << 2,

        /// <summary>A numeric DType (<c>PyTypeNum_ISNUMBER</c>): bool, integers, floats, complex.</summary>
        Numeric = 1 << 3,
    }
}
