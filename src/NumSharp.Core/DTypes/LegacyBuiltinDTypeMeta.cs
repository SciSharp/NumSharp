using System;
using System.Collections.Generic;

namespace NumSharp
{
    /// <summary>
    ///     The DType class of a storage-backed NumSharp dtype — the 15 element types (plus the vestigial
    ///     <see cref="NPTypeCode.String"/>), wrapped the way NumPy's <c>dtypemeta_wrap_legacy_descriptor</c> wraps its
    ///     builtin descriptors into <c>numpy.dtypes.*DType</c> classes. Data-driven: the registry constructs one per
    ///     <see cref="NPTypeCode"/> with its NumPy type number, kind, char, size, alignment and aliases; the behaviour
    ///     slots are the builtin ones — the singleton is the default descriptor, and <see cref="common_dtype"/> is NumPy's
    ///     <c>default_builtin_common_dtype</c> over NumSharp's frozen promotion table
    ///     (<see cref="np._nptypemap_arr_arr"/>), so every existing <c>promote_types</c> answer is reproduced bit-for-bit.
    /// </summary>
    public sealed class LegacyBuiltinDTypeMeta : DTypeMeta
    {
        /// <summary>Builds one builtin class; only <see cref="DTypeRegistry"/> calls this, once per <see cref="NPTypeCode"/>.</summary>
        /// <param name="name">NumPy's class name (<c>"Int32DType"</c>).</param>
        /// <param name="modulePath"><c>"numpy.dtypes"</c>, or <c>"numsharp.dtypes"</c> for the NumSharp-only classes.</param>
        /// <param name="typeNum">The class's <c>NPY_TYPES</c> number.</param>
        /// <param name="typeCode">The storage discriminator.</param>
        /// <param name="scalarType">The C# element type.</param>
        /// <param name="scalarName">NumPy's scalar type name (<c>"int32"</c>).</param>
        /// <param name="kind">NumPy's kind character.</param>
        /// <param name="typeChar">NumPy's type character.</param>
        /// <param name="itemSize">Element size in bytes.</param>
        /// <param name="alignment">Required alignment in bytes.</param>
        /// <param name="flags">The <c>NPY_DT_*</c> flags.</param>
        /// <param name="aliases">The alias class names NumPy exports for this class.</param>
        internal LegacyBuiltinDTypeMeta(string name, string modulePath, int typeNum, NPTypeCode typeCode, Type scalarType,
            string scalarName, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, IReadOnlyList<string> aliases)
            : base(name, modulePath, typeNum, scalarType, scalarName, typeCode, kind, typeChar, itemSize, alignment, flags, aliases)
        {
        }

        /// <summary>NumPy's builtin <c>default_descr</c>: the singleton — a non-parametric class has exactly one canonical descriptor.</summary>
        /// <returns><see cref="DTypeMeta.singleton"/>.</returns>
        public override DType default_descr() => singleton;

        /// <summary>
        ///     NumPy's <c>default_builtin_common_dtype</c>: the NEP 50 Python-scalar rules first (a weak float adopts a
        ///     float/complex class, a weak int adopts any numeric class, a weak complex lifts a real float to complex),
        ///     then defer to the class with the larger type number, then the frozen promotion table.
        /// </summary>
        /// <param name="other">The other operand's class.</param>
        /// <returns>The common class, or null (NotImplemented) when the other class must answer.</returns>
        public override DTypeMeta common_dtype(DTypeMeta other) => DTypePromotion.DefaultBuiltinCommonDType(this, other);

        /// <summary>
        ///     The slot <c>is_known_scalar_type</c> for a builtin class: the C# element type, and a C# array of it
        ///     (<c>np.array(int[])</c> is int32). NumPy's legacy slot (<c>python_builtins_are_known_scalar_types</c>)
        ///     accepts Python's own scalars instead; C# has no such shared scalar set, so each class claims its element type.
        /// </summary>
        /// <param name="type">The C# type of the scalar.</param>
        /// <returns>True when <paramref name="type"/> belongs to this class.</returns>
        public override bool is_known_scalar_type(Type type)
        {
            if (type == null)
                return false;
            if (this.type != null && type == this.type)
                return true;
            // C# array types of the scalar map to the same class (np.array(int[]) is int32).
            return type.IsArray && type.GetTypeCode() == TypeCode && TypeCode != NPTypeCode.Empty;
        }
    }
}
