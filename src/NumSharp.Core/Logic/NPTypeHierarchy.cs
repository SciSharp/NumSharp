using System;
using System.Collections.Generic;

namespace NumSharp
{
    // ================================================================================
    // NUMPY TYPE HIERARCHY
    // ================================================================================
    //
    // This file implements NumPy's exact type hierarchy as defined in:
    //   numpy/_core/src/multiarray/multiarraymodule.c (lines 4764-4839)
    //
    // NumPy uses Python class inheritance:
    //   SINGLE_INHERIT(Number, Generic);
    //   SINGLE_INHERIT(Integer, Number);
    //   SINGLE_INHERIT(SignedInteger, Integer);
    //   ...etc
    //
    // Then issubdtype() is simply: issubclass(type1, type2)
    //
    // Since C# primitives don't have this hierarchy, we encode it explicitly
    // and provide IsSubType() which walks up the tree like issubclass().
    //
    // HIERARCHY (from NumPy):
    //
    //   Generic
    //   ├── Bool (NOT under Number!)
    //   ├── Number
    //   │   ├── Integer
    //   │   │   ├── SignedInteger
    //   │   │   │   └── int8, int16, int32, int64
    //   │   │   └── UnsignedInteger
    //   │   │       └── uint8, uint16, uint32, uint64
    //   │   └── Inexact
    //   │       ├── Floating
    //   │       │   └── float16, float32, float64
    //   │       └── ComplexFloating
    //   │           └── complex64, complex128
    //   ├── Flexible (not used in NumSharp)
    //   ├── Datetime (not used in NumSharp)
    //   └── Object (not used in NumSharp)
    //
    // ================================================================================

    /// <summary>
    /// Abstract type categories in NumPy's type hierarchy.
    /// These mirror NumPy's abstract scalar types (np.generic, np.number, etc.)
    /// </summary>
    public enum NPTypeKind
    {
        // Root
        Generic,

        // Number branch
        Number,
        Integer,
        SignedInteger,
        UnsignedInteger,
        Inexact,
        Floating,
        ComplexFloating,

        // Bool is directly under Generic (NOT Number) in NumPy 2.x
        Boolean,
    }

    /// <summary>
    /// Encodes NumPy's exact type hierarchy for use by issubdtype, isdtype, can_cast, etc.
    /// Single source of truth - all typing functions should use this class.
    /// </summary>
    internal static class NPTypeHierarchy
    {
        // Parent relationships: _parent[child] = parent
        // Walk up this tree to implement issubclass-like behavior
        private static readonly Dictionary<NPTypeKind, NPTypeKind> _abstractParent = new()
        {
            // Number branch
            [NPTypeKind.SignedInteger] = NPTypeKind.Integer,
            [NPTypeKind.UnsignedInteger] = NPTypeKind.Integer,
            [NPTypeKind.Integer] = NPTypeKind.Number,
            [NPTypeKind.Floating] = NPTypeKind.Inexact,
            [NPTypeKind.ComplexFloating] = NPTypeKind.Inexact,
            [NPTypeKind.Inexact] = NPTypeKind.Number,
            [NPTypeKind.Number] = NPTypeKind.Generic,

            // Bool is directly under Generic (NumPy 2.x behavior)
            [NPTypeKind.Boolean] = NPTypeKind.Generic,
        };

        // Map concrete types to their immediate abstract parent
        private static readonly Dictionary<NPTypeCode, NPTypeKind> _concreteParent = new()
        {
            // SignedInteger children
            [NPTypeCode.Int16] = NPTypeKind.SignedInteger,
            [NPTypeCode.Int32] = NPTypeKind.SignedInteger,
            [NPTypeCode.Int64] = NPTypeKind.SignedInteger,

            // UnsignedInteger children
            [NPTypeCode.Byte] = NPTypeKind.UnsignedInteger,
            [NPTypeCode.UInt16] = NPTypeKind.UnsignedInteger,
            [NPTypeCode.UInt32] = NPTypeKind.UnsignedInteger,
            [NPTypeCode.UInt64] = NPTypeKind.UnsignedInteger,

            // Char is treated as UnsignedInteger (like uint16)
            [NPTypeCode.Char] = NPTypeKind.UnsignedInteger,

            // Floating children
            [NPTypeCode.Single] = NPTypeKind.Floating,
            [NPTypeCode.Double] = NPTypeKind.Floating,
            [NPTypeCode.Decimal] = NPTypeKind.Floating,  // NumSharp-specific

            // ComplexFloating children
            [NPTypeCode.Complex] = NPTypeKind.ComplexFloating,

            // Bool is under Generic directly
            [NPTypeCode.Boolean] = NPTypeKind.Boolean,
        };

        /// <summary>
        /// Check if a concrete type is a subtype of an abstract category.
        /// Equivalent to NumPy's issubdtype(concrete_type, abstract_type).
        /// </summary>
        /// <param name="type">Concrete type code.</param>
        /// <param name="category">Abstract category to check against.</param>
        /// <returns>True if type is in the category's subtree.</returns>
        public static bool IsSubType(NPTypeCode type, NPTypeKind category)
        {
            if (category == NPTypeKind.Generic)
                return true;  // Everything is a subtype of generic

            if (!_concreteParent.TryGetValue(type, out var parent))
                return false;  // Unknown type

            // Walk up the tree
            var current = parent;
            while (true)
            {
                if (current == category)
                    return true;

                if (!_abstractParent.TryGetValue(current, out var nextParent))
                    return false;

                current = nextParent;
            }
        }

        /// <summary>
        /// Check if an abstract category is a subtype of another abstract category.
        /// Equivalent to NumPy's issubdtype(abstract_type1, abstract_type2).
        /// </summary>
        public static bool IsSubType(NPTypeKind child, NPTypeKind parent)
        {
            if (child == parent)
                return true;

            if (parent == NPTypeKind.Generic)
                return true;

            var current = child;
            while (_abstractParent.TryGetValue(current, out var nextParent))
            {
                if (nextParent == parent)
                    return true;
                current = nextParent;
            }
            return false;
        }

        /// <summary>
        /// Get the immediate abstract parent category for a concrete type.
        /// </summary>
        public static NPTypeKind GetImmediateKind(NPTypeCode type)
        {
            return _concreteParent.TryGetValue(type, out var kind) ? kind : NPTypeKind.Generic;
        }

        /// <summary>
        /// Check if two concrete types are in the same "kind" for same_kind casting.
        /// Same kind means they share the same immediate abstract parent (SignedInteger, UnsignedInteger, Floating, etc.)
        /// </summary>
        public static bool IsSameKind(NPTypeCode type1, NPTypeCode type2)
        {
            var kind1 = GetImmediateKind(type1);
            var kind2 = GetImmediateKind(type2);

            // Same immediate kind
            if (kind1 == kind2)
                return true;

            // SignedInteger and UnsignedInteger are both Integer (same_kind allows int<->uint)
            if ((kind1 == NPTypeKind.SignedInteger || kind1 == NPTypeKind.UnsignedInteger) &&
                (kind2 == NPTypeKind.SignedInteger || kind2 == NPTypeKind.UnsignedInteger))
                return true;

            return false;
        }

        /// <summary>
        /// Get the maximum precision type for a given kind.
        /// Used by maximum_sctype.
        /// </summary>
        /// <remarks>
        /// NumPy returns the highest precision type of the same kind:
        /// - int16/int32 -> int64
        /// - uint8/uint16/uint32 -> uint64
        /// - float32 -> float64 (or longdouble)
        /// - complex64 -> complex128 (or clongdouble)
        ///
        /// NumSharp-specific: Decimal stays Decimal (it's already max precision
        /// for decimal arithmetic, different from IEEE floating point).
        /// </remarks>
        public static NPTypeCode GetMaximumType(NPTypeCode type)
        {
            // Handle special cases first
            return type switch
            {
                // Boolean stays boolean
                NPTypeCode.Boolean => NPTypeCode.Boolean,

                // Signed integers -> int64
                NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64 => NPTypeCode.Int64,

                // Unsigned integers -> uint64
                NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 => NPTypeCode.UInt64,

                // Char treated as unsigned integer
                NPTypeCode.Char => NPTypeCode.UInt64,

                // Float types: Single/Double -> Double, Decimal stays Decimal
                NPTypeCode.Single or NPTypeCode.Double => NPTypeCode.Double,
                NPTypeCode.Decimal => NPTypeCode.Decimal,

                // Complex stays complex (we only have one complex type)
                NPTypeCode.Complex => NPTypeCode.Complex,

                // Unknown types stay as-is
                _ => type
            };
        }

        /// <summary>
        /// Parse a string category name to NPTypeKind.
        /// Supports NumPy's category names and common aliases.
        /// </summary>
        public static bool TryParseKind(string name, out NPTypeKind kind)
        {
            kind = name.ToLowerInvariant() switch
            {
                "generic" => NPTypeKind.Generic,
                "number" => NPTypeKind.Number,
                "integer" => NPTypeKind.Integer,
                "signedinteger" or "signed" or "signed integer" => NPTypeKind.SignedInteger,
                "unsignedinteger" or "unsigned" or "unsigned integer" => NPTypeKind.UnsignedInteger,
                "inexact" => NPTypeKind.Inexact,
                "floating" or "float" or "real floating" => NPTypeKind.Floating,
                "complexfloating" or "complex" or "complex floating" => NPTypeKind.ComplexFloating,
                "bool" or "boolean" => NPTypeKind.Boolean,

                // isdtype uses different names
                "integral" => NPTypeKind.Integer,
                "numeric" => NPTypeKind.Number,

                _ => NPTypeKind.Generic
            };

            // Return false for unrecognized names
            return name.ToLowerInvariant() switch
            {
                "generic" or "number" or "integer" or "signedinteger" or "signed" or "signed integer"
                or "unsignedinteger" or "unsigned" or "unsigned integer" or "inexact"
                or "floating" or "float" or "real floating" or "complexfloating" or "complex" or "complex floating"
                or "bool" or "boolean" or "integral" or "numeric" => true,
                _ => false
            };
        }

        /// <summary>
        /// Check if a type belongs to a category specified by string name.
        /// This is the core logic used by issubdtype(type, "category_name").
        /// </summary>
        public static bool IsSubType(NPTypeCode type, string categoryName)
        {
            if (!TryParseKind(categoryName, out var category))
                return false;

            // Special case: "numeric" in isdtype excludes bool
            if (categoryName.ToLowerInvariant() == "numeric")
                return type != NPTypeCode.Boolean && IsSubType(type, NPTypeKind.Number);

            return IsSubType(type, category);
        }
    }
}
