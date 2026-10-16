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
    /// <remarks>
    ///     <para>
    ///     <b>Lane-sharing classes.</b> NumPy keeps a class per C type, and several C types can have one width: Windows'
    ///     <c>int</c> and <c>long</c> are both 32-bit (<c>IntDType</c> beside <c>Int32DType</c>), Linux's <c>long</c> and
    ///     <c>long long</c> both 64-bit (<c>LongLongDType</c> beside <c>Int64DType</c>), and NumSharp's <c>long double</c> is
    ///     a 64-bit double on every platform (<c>LongDoubleDType</c> beside <c>Float64DType</c>, <c>CLongDoubleDType</c>
    ///     beside <c>Complex128DType</c>). Such a class stores its elements in the storage lane of its width (the kernels
    ///     are the lane's) but does not OWN the lane: <see cref="DTypeRegistry.FromTypeCode"/> and
    ///     <see cref="DTypeRegistry.FromScalarType"/> keep returning the lane's own class, and an array keeps the
    ///     lane-sharing descriptor only where it is carried explicitly (creation with that dtype, views, copies, casts).
    ///     </para>
    ///     <para>
    ///     NumPy calls two such classes EQUIVALENT (<c>np.dtype('g') == np.dtype('d')</c> on Windows, a cast between them
    ///     is a no-op), and so does NumSharp: <see cref="DType.Equals(DType)"/>, <see cref="BuiltinCastingImpl"/> and the
    ///     promotion table treat every two NumPy classes of one kind and size that way.
    ///     </para>
    /// </remarks>
    public sealed class LegacyBuiltinDTypeMeta : DTypeMeta
    {
        /// <summary>Builds one builtin class; only <see cref="DTypeRegistry"/> calls this, once per class.</summary>
        /// <param name="name">NumPy's class name (<c>"Int32DType"</c>).</param>
        /// <param name="modulePath"><c>"numpy.dtypes"</c>, or <c>"numsharp.dtypes"</c> for the NumSharp-only classes.</param>
        /// <param name="typeNum">The class's <c>NPY_TYPES</c> number.</param>
        /// <param name="typeCode">The storage discriminator.</param>
        /// <param name="scalarType">The C# element type.</param>
        /// <param name="scalarName">NumPy's scalar type name (<c>"int32"</c>, <c>"intc"</c>, <c>"longdouble"</c>).</param>
        /// <param name="kind">NumPy's kind character.</param>
        /// <param name="typeChar">NumPy's type character.</param>
        /// <param name="itemSize">Element size in bytes.</param>
        /// <param name="alignment">Required alignment in bytes.</param>
        /// <param name="flags">The <c>NPY_DT_*</c> flags.</param>
        /// <param name="aliases">The alias class names NumPy exports for this class.</param>
        /// <param name="ownsLane">
        ///     True (the default) for the class that OWNS its storage lane (the registry maps the lane's
        ///     <see cref="NPTypeCode"/> and C# element type back to it); false for a lane-sharing C-type class (see the
        ///     remarks), which is reached by name, type character and type number only.
        /// </param>
        internal LegacyBuiltinDTypeMeta(string name, string modulePath, int typeNum, NPTypeCode typeCode, Type scalarType,
            string scalarName, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, IReadOnlyList<string> aliases,
            bool ownsLane = true)
            : base(name, modulePath, typeNum, scalarType, scalarName, typeCode, kind, typeChar, itemSize, alignment, flags, aliases)
        {
            _ownsLane = ownsLane;
            // Decided once: DType.Equals / GetHashCode read it on every comparison of two descriptors.
            _isNumPyNumeric = modulePath == "numpy.dtypes" && (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f' || kind == 'c');
        }

        /// <summary>Whether this class owns its storage lane (see the constructor's <c>ownsLane</c>).</summary>
        private readonly bool _ownsLane;

        /// <summary>The cached answer of <see cref="IsNumPyNumeric"/> (the module and kind never change).</summary>
        private readonly bool _isNumPyNumeric;

        /// <summary>
        ///     The lane owner maps its <see cref="NPTypeCode"/> back to itself; a lane-sharing C-type class does not, so
        ///     <c>(DType)NPTypeCode.Double</c> stays <c>float64</c> and an array computed in the lane is labelled with the
        ///     lane's own class.
        /// </summary>
        internal override bool OwnsTypeCode => _ownsLane;

        /// <summary>
        ///     The lane owner claims the C# element type; a lane-sharing class does not, so <c>typeof(double)</c> stays
        ///     <c>float64</c> and <c>typeof(int)</c> stays <c>int32</c>.
        /// </summary>
        internal override bool OwnsScalarType => _ownsLane;

        /// <summary>
        ///     True for a class of NumPy's own legacy numeric set (module <c>numpy.dtypes</c>, kind <c>b</c> / <c>i</c> /
        ///     <c>u</c> / <c>f</c> / <c>c</c>): the classes among which NumPy's <c>add_numeric_cast</c> makes a pair of one
        ///     kind and one size a no-op cast, and <c>dtype.__eq__</c> calls them equal. NumSharp's own decimal and char are
        ///     not in it (char and uint16 stay two classes with two kernels).
        /// </summary>
        internal bool IsNumPyNumeric => _isNumPyNumeric;

        /// <summary>
        ///     Whether a cast from this class to <paramref name="other"/> is NumPy's <c>numeric_copy_or_byteswap</c>: the
        ///     same class, or two NumPy numeric classes of one kind and one item size (<c>intc</c> ↔ <c>int32</c> on Windows,
        ///     <c>longdouble</c> ↔ <c>float64</c>), whose elements have one bit layout.
        /// </summary>
        /// <param name="other">The other class.</param>
        /// <returns>True for an equivalent pair.</returns>
        internal bool IsEquivalentNumeric(LegacyBuiltinDTypeMeta other)
            => ReferenceEquals(this, other)
               || (other != null && IsNumPyNumeric && other.IsNumPyNumeric && Kind == other.Kind && ItemSize == other.ItemSize);

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
