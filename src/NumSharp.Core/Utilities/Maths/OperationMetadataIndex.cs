using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;

namespace NumSharp.Utilities.Maths
{
    using OperationTypeIndex = Dictionary<NPTypeCode, Type>;

    /// <summary>
    /// Encapsulates the base operation metadata index functionality that supports dynamic <see cref="NDArray"/> operation inference.
    /// </summary>
#pragma warning disable CA1052 // Static holder types should be Static or NotInheritable
    public class OperationMetadataIndex
#pragma warning restore CA1052 // Static holder types should be Static or NotInheritable
    {
        protected const BindingFlags Filter = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;

        /// <summary>
        /// The index of types supported by <see cref="NDArray"/> operations.
        /// </summary>
        private static readonly OperationTypeIndex s_types = new Type[]
        {
            typeof(bool), typeof(char),
            typeof(sbyte), typeof(byte), typeof(short), typeof(ushort),
            typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double), typeof(decimal), typeof(Complex)
        }.ToDictionary(x => x.GetTypeCode());

        /// <summary>
        /// Translates the <see cref="NPTypeCode"/> type code in to the actual reflecton type.
        /// </summary>
        /// <param name="code">The type code to translate into the type.</param>
        /// <returns>The <see cref="NDArray"/> element <see cref="Type"/>translated.</returns>
        public static Type Translate(NPTypeCode code)
        {
            if (s_types.TryGetValue(code, out var type))
                return type;
            else
                throw new ArgumentException($"The operation index does not support requested type code: {code}.");
        }

        /// <summary>
        /// Validates the <see cref="NDArray"/> element <see cref="Type"/> against the index of supported type.
        /// </summary>
        /// <param name="type">The <see cref="Type"/> to validate agains the operation metadata index.</param>
        /// <returns>True if the <see cref="Type"/> is supported; otherwise, false.</returns>
        public static bool IsValid(Type type)
        {
            if (type is null)
                throw new ArgumentNullException(nameof(type));

            return s_types.ContainsKey(type.GetTypeCode());
        }
    }
}
