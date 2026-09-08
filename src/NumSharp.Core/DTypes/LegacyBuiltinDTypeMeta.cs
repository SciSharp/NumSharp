using System;
using System.Collections.Generic;

namespace NumSharp
{
    /// <summary>
    ///     The DType class of a storage-backed NumSharp dtype — the 15 element types (plus the vestigial
    ///     <see cref="NPTypeCode.String"/>), wrapped the way NumPy's <c>dtypemeta_wrap_legacy_descriptor</c> wraps its
    ///     builtin descriptors into <c>numpy.dtypes.*DType</c> classes. Data-driven: the registry constructs one per
    ///     <see cref="NPTypeCode"/> with its NumPy type number, kind, char, size, alignment and aliases; the behaviour
    ///     slots are the builtin ones — the singleton is the default descriptor, and <see cref="CommonDType"/> is NumPy's
    ///     <c>default_builtin_common_dtype</c> over NumSharp's frozen promotion table
    ///     (<see cref="np._nptypemap_arr_arr"/>), so every existing <c>promote_types</c> answer is reproduced bit-for-bit.
    /// </summary>
    public sealed class LegacyBuiltinDTypeMeta : DTypeMeta
    {
        internal LegacyBuiltinDTypeMeta(string name, string modulePath, int typeNum, NPTypeCode typeCode, Type scalarType,
            string scalarName, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, IReadOnlyList<string> aliases)
            : base(name, modulePath, typeNum, scalarType, scalarName, typeCode, kind, typeChar, itemSize, alignment, flags, aliases)
        {
        }

        /// <summary>The singleton — a non-parametric class has exactly one canonical descriptor.</summary>
        public override DType DefaultDescr() => Singleton;

        /// <summary>
        ///     NumPy's <c>default_builtin_common_dtype</c>: the NEP 50 Python-scalar rules first (a weak float adopts a
        ///     float/complex class, a weak int adopts any numeric class, a weak complex lifts a real float to complex),
        ///     then defer to the class with the larger type number, then the frozen promotion table.
        /// </summary>
        public override DTypeMeta CommonDType(DTypeMeta other) => DTypePromotion.DefaultBuiltinCommonDType(this, other);

        /// <inheritdoc/>
        public override bool IsKnownScalarType(Type type)
        {
            if (type == null)
                return false;
            if (ScalarType != null && type == ScalarType)
                return true;
            // C# array types of the scalar map to the same class (np.array(int[]) is int32).
            return type.IsArray && type.GetTypeCode() == TypeCode && TypeCode != NPTypeCode.Empty;
        }
    }
}
