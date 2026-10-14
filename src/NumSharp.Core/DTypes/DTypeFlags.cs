using System;

namespace NumSharp
{
    /// <summary>
    ///     Flags describing a <see cref="DTypeMeta"/> — the DType CLASS — under NumPy's own names, the <c>NPY_DT_*</c>
    ///     bits of <c>dtypemeta.h</c> / <c>dtype_api.h</c> with NumPy's values (<c>NPY_DT_LEGACY = 1</c>,
    ///     <c>NPY_DT_ABSTRACT = 2</c>, <c>NPY_DT_PARAMETRIC = 4</c>, <c>NPY_DT_NUMERIC = 8</c>), so a flags word compares
    ///     bit for bit with NumPy's <c>PyArray_DTypeMeta.flags</c>.
    /// </summary>
    /// <remarks>
    ///     The members keep NumPy's C spellings, as <see cref="NumSharp.Backends.Iteration.NPY_CASTING"/> and <see cref="NPY_TYPES"/> do, because the
    ///     NumPy surface of NumSharp is case-sensitive to NumPy. Python reads the same bits through the class attributes
    ///     <c>_legacy</c>, <c>_abstract</c>, <c>_parametric</c> and <c>_is_numeric</c>, which <see cref="DTypeMeta"/>
    ///     exposes under those names. <see cref="None"/> is the C# flags-enum zero; NumPy has no name for an empty word.
    /// </remarks>
    [Flags]
    public enum DTypeFlags
    {
        /// <summary>No flags (the C# zero value; NumPy names no empty word).</summary>
        None = 0,

        /// <summary>
        ///     <c>NPY_DT_LEGACY</c>: one of the builtin classes whose descriptor layout predates NEP 41/42 (every
        ///     storage-backed NumSharp dtype and the datetime pair carry it; NEP 55's <c>StringDType</c> would not).
        ///     Read in Python as <c>DType._legacy</c>.
        /// </summary>
        NPY_DT_LEGACY = 1 << 0,

        /// <summary>
        ///     <c>NPY_DT_ABSTRACT</c>: the class has no instances and exists only to take part in promotion — the NEP 50
        ///     Python-scalar classes <c>_PyLongDType</c> / <c>_PyFloatDType</c> / <c>_PyComplexDType</c>. Read in Python
        ///     as <c>DType._abstract</c>.
        /// </summary>
        NPY_DT_ABSTRACT = 1 << 1,

        /// <summary>
        ///     <c>NPY_DT_PARAMETRIC</c>: the instances carry parameters beyond the class (a datetime unit, a string
        ///     length), so <c>promote_types</c> runs <c>common_instance</c> instead of returning the class default. Read in
        ///     Python as <c>DType._parametric</c>.
        /// </summary>
        NPY_DT_PARAMETRIC = 1 << 2,

        /// <summary>
        ///     <c>NPY_DT_NUMERIC</c> (<c>PyTypeNum_ISNUMBER</c>): bool, the integers, the floats and complex. Read in
        ///     Python as <c>DType._is_numeric</c>.
        /// </summary>
        NPY_DT_NUMERIC = 1 << 3,
    }
}
