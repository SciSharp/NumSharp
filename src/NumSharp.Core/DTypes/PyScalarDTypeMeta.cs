using System;

namespace NumSharp
{
    /// <summary>The three NEP 50 "weak" literal categories: a Python <c>int</c>, <c>float</c>, <c>complex</c>.</summary>
    public enum PyScalarKind
    {
        /// <summary>A Python <c>int</c> literal — in C#, any integer primitive (<c>int</c>, <c>long</c>, <c>byte</c>, …).</summary>
        Int,
        /// <summary>A Python <c>float</c> literal — in C#, <c>double</c> or <c>float</c>.</summary>
        Float,
        /// <summary>A Python <c>complex</c> literal — in C#, <see cref="System.Numerics.Complex"/>.</summary>
        Complex,
    }

    /// <summary>
    ///     NumPy's abstract DType classes for Python scalars — <c>numpy.dtypes._PyLongDType</c>,
    ///     <c>_PyFloatDType</c>, <c>_PyComplexDType</c> (<c>abstractdtypes.c</c>). They are how NEP 50 "weak" promotion
    ///     works: a literal contributes only its abstract CLASS to <c>result_type</c> (no descriptor, so no value or
    ///     width), and the class's <see cref="common_dtype"/> lets the other operand's dtype win within its kind
    ///     (<c>int8 + 300 → int8</c>, <c>float32 + 1e300 → float32</c>) while still lifting across kinds
    ///     (<c>int8 + 1.5 → float64</c>). A literal that survives alone falls back to the class default
    ///     (<c>int64</c> / <c>float64</c> / <c>complex128</c>).
    /// </summary>
    /// <remarks>
    ///     C# has no Python-literal/NumPy-scalar split, so the mapping is by TYPE (the house NEP 50 rule): the integer
    ///     primitives, <c>float</c>/<c>double</c> and <c>Complex</c> are weak; <c>bool</c>, <c>char</c>, <c>Half</c>,
    ///     <c>decimal</c> and every <see cref="NDArray"/> (0-d included) are strong.
    /// </remarks>
    public sealed class PyScalarDTypeMeta : DTypeMeta
    {
        private readonly NPTypeCode _defaultCode;

        /// <summary>Builds one abstract scalar class; only <see cref="DTypeRegistry"/> calls this, once per <see cref="PyScalarKind"/>.</summary>
        /// <param name="name">NumPy's class name (<c>"_PyLongDType"</c>).</param>
        /// <param name="pyKind">Which literal category the class stands for.</param>
        /// <param name="defaultCode">The storage type a lone literal of the category resolves to.</param>
        internal PyScalarDTypeMeta(string name, PyScalarKind pyKind, NPTypeCode defaultCode)
            : base(name, "numpy.dtypes", -1, scalarType: null, scalarName: name, NPTypeCode.Empty, kind: '\0', typeChar: '\0',
                itemSize: -1, alignment: -1, DTypeFlags.NPY_DT_ABSTRACT)
        {
            PyKind = pyKind;
            _defaultCode = defaultCode;
        }

        /// <summary>Which literal category this class stands for.</summary>
        public PyScalarKind PyKind { get; }

        /// <summary>The default descriptor a lone literal resolves to: <c>int64</c> (NumPy's <c>intp</c>), <c>float64</c>, <c>complex128</c>.</summary>
        /// <returns>The singleton of the default class.</returns>
        public override DType default_descr() => DTypeRegistry.FromTypeCode(_defaultCode).singleton;

        /// <summary>
        ///     NumPy's <c>discover_descriptor_from_py{long,float,complex}</c> (<c>abstractdtypes.c</c>): the dtype an array
        ///     made from ONE literal gets, which is what <c>np.result_type(literal)</c> answers (a lone operand skips
        ///     promotion and returns the dtype of <c>np.asarray(literal)</c>). A Python int is read by VALUE: the default
        ///     integer while it fits <c>int64</c>, <c>uint64</c> above that (<c>np.result_type(2**63)</c> is uint64); a
        ///     float is float64 and a complex complex128. As an operand among others the literal stays weak and its value
        ///     never matters (<c>np.result_type(np.uint8, 2**63)</c> is uint8).
        /// </summary>
        /// <param name="obj">The literal: a boxed C# integer primitive, <c>float</c> / <c>double</c>, or <see cref="System.Numerics.Complex"/>.</param>
        /// <returns>The singleton of the discovered class.</returns>
        /// <remarks>
        ///     NumPy's third outcome, an <c>object</c> array for an int past <c>2**64 - 1</c>, cannot arise: no C# integer
        ///     primitive holds such a value.
        /// </remarks>
        public override DType discover_descr_from_pyobject(object obj)
        {
            // Only a ulong can exceed int64 among the C# integers; NumPy's PyLong_AsLongLong fails for it and
            // PyLong_AsUnsignedLongLong succeeds, which is NPY_ULONGLONG (uint64).
            if (PyKind == PyScalarKind.Int && obj is ulong u && u > long.MaxValue)
                return DTypeRegistry.UInt64.singleton;
            return default_descr();
        }

        /// <summary>Weak literals map onto the C# primitives of their category.</summary>
        /// <param name="type">The C# type of the literal.</param>
        /// <returns>True when <paramref name="type"/> is a C# primitive of this class's category.</returns>
        public override bool is_known_scalar_type(Type type)
        {
            if (type == null)
                return false;
            switch (PyKind)
            {
                case PyScalarKind.Int:
                    return type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(sbyte)
                           || type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);
                case PyScalarKind.Float:
                    return type == typeof(double) || type == typeof(float);
                default:
                    return type == typeof(System.Numerics.Complex);
            }
        }

        /// <summary>
        ///     Ports of <c>int_common_dtype</c> / <c>float_common_dtype</c> / <c>complex_common_dtype</c>. The builtin
        ///     classes handle most pairs themselves (their <c>common_dtype</c> is asked FIRST and answers for a literal of
        ///     their own or a lower kind); these implement the remaining directions: a weak int with <c>bool</c> is the
        ///     default integer, a weak float or complex with an integer or bool class is <c>float64</c> /
        ///     <c>complex128</c>, and two abstract classes resolve to the higher category. The NumSharp-only storage
        ///     classes (Decimal, Char) are treated as the numeric builtins they masquerade as — Char is an unsigned
        ///     integer here, so <c>result_type(char_array, 1.5)</c> is <c>float64</c> exactly like <c>uint16</c> — rather
        ///     than through NumPy's opaque-user-dtype fallback; a genuinely foreign legacy class still takes that
        ///     fallback (promote with the smallest concrete type of the literal's category and let it decide).
        /// </summary>
        /// <param name="other">The other operand's class.</param>
        /// <returns>The common class, or null (NotImplemented) when the other class must answer.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
        public override DTypeMeta common_dtype(DTypeMeta other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            bool otherIsBuiltin = other._legacy && other.HasStorage
                                  && (other.Kind == 'b' || other.Kind == 'i' || other.Kind == 'u' || other.Kind == 'f' || other.Kind == 'c');
            bool otherIsUser = other._legacy && !otherIsBuiltin && other.type_num >= 256;

            switch (PyKind)
            {
                case PyScalarKind.Int:
                    if (otherIsBuiltin)
                    {
                        if (other.Kind == 'b')
                            return DTypeRegistry.Int64; /* Use the default integer for bools */
                    }
                    else if (otherIsUser)
                    {
                        /* This is a back-compat fallback to usually do the right thing... */
                        return other.common_dtype(DTypeRegistry.UInt8)
                               ?? other.common_dtype(DTypeRegistry.Int8)
                               ?? other.common_dtype(DTypeRegistry.Int64);
                    }
                    return null;

                case PyScalarKind.Float:
                    if (otherIsBuiltin)
                    {
                        if (other.Kind == 'b' || other.Kind == 'i' || other.Kind == 'u')
                            return DTypeRegistry.Double; /* Use the default float for bools and ints */
                    }
                    else if (other is PyScalarDTypeMeta py && py.PyKind == PyScalarKind.Int)
                    {
                        return this;
                    }
                    else if (otherIsUser)
                    {
                        return other.common_dtype(DTypeRegistry.Half) ?? other.common_dtype(DTypeRegistry.Double);
                    }
                    return null;

                default:
                    if (otherIsBuiltin)
                    {
                        if (other.Kind == 'b' || other.Kind == 'i' || other.Kind == 'u')
                            return DTypeRegistry.Complex128; /* Use the default complex for bools and ints */
                    }
                    else if (otherIsUser)
                    {
                        return other.common_dtype(DTypeRegistry.Complex128);
                    }
                    else if (other is PyScalarDTypeMeta py2 && (py2.PyKind == PyScalarKind.Int || py2.PyKind == PyScalarKind.Float))
                    {
                        return this;
                    }
                    return null;
            }
        }
    }
}
