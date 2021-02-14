using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using NumSharp.Backends;

namespace NumSharp
{
    /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.dtype.html#numpy.dtype</remarks>
    public class DType
    {
        protected internal static readonly Dictionary<NPTypeCode, char> _kind_list_map = new Dictionary<NPTypeCode, char>()
        {
            {NPTypeCode.Complex, 'c'},
            {NPTypeCode.Boolean, '?'},
            {NPTypeCode.Byte, 'b'},
            {NPTypeCode.Int16, 'i'},
            {NPTypeCode.UInt16, 'u'},
            {NPTypeCode.Int32, 'i'},
            {NPTypeCode.UInt32, 'u'},
            {NPTypeCode.Int64, 'i'},
            {NPTypeCode.UInt64, 'u'},
            {NPTypeCode.Char, 'S'},
            {NPTypeCode.Double, 'f'},
            {NPTypeCode.Single, 'f'},
            {NPTypeCode.Decimal, 'f'},
            {NPTypeCode.String, 'S'},
        };

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public DType(Type type)
        {
            this.type = type ?? throw new ArgumentNullException(nameof(type));
            typecode = type.GetTypeCode();
            name = type.Name;
            byteorder = '=';
            itemsize = typecode.SizeOf();
            TYPECHAR = typecode.ToTYPECHAR();
            kind = _kind_list_map[typecode];
        }

        /// <summary>
        ///     A character indicating the byte-order of this data-type object.<br></br>
        ///     One of:<br></br>
        ///     
        ///     '='	native<br></br>
        ///     '\&lt;'	little-endian<br></br>
        ///     '&gt;'	big-endian<br></br>
        ///     '|'	not applicable<br></br>
        /// </summary>
        public char byteorder;

        /// <summary>
        ///     The size of the dtype in bytes.
        /// </summary>
        public int itemsize;

        /// <summary>
        ///     The name of this dtype.
        /// </summary>
        public string name;

        /// <summary>
        ///     The actual type this dtype represents.
        /// </summary>
        public Type type;

        /// <summary>
        ///     The NumSharp type code.
        /// </summary>
        public NPTypeCode typecode;

        /// <summary>
        ///     A unique character code for each of the 21 different built-in types.
        /// </summary>
        internal NPY_TYPECHAR TYPECHAR;

        /// <summary>
        ///     A character code (one of ‘biufcmMOSUV’) identifying the general kind of data.<br></br><br></br>
        ///     b boolean<br></br>
        ///     i signed integer<br></br>
        ///     u   unsigned integer<br></br>
        ///     f floating-point<br></br>
        ///     c   complex floating-point<br></br>
        ///     m   timedelta<br></br>
        ///     M   datetime<br></br>
        ///     O   object<br></br>
        ///     S(byte-)string<br></br>
        ///     U   Unicode<br></br>
        ///     V   void<br></br>
        /// </summary>
        public char kind;

        /// <summary>
        /// A unique character code for each of the 21 different built-in types.
        /// </summary>
        public char @char => (char)TYPECHAR;

        /// <summary>
        ///     Return a new dtype with a different byte order.
        ///     Changes are also made in all fields and sub-arrays of the data type.
        /// </summary>
        /// <param name="new_order">
        ///     Byte order to force; a value from the byte order specifications below.<br></br> The default value (‘S’) results in swapping the current byte order.<br></br> new_order codes can be any of:<br></br>
        ///     ‘S’ - swap dtype from current to opposite endian<br></br>
        ///     '='	- native order<br></br>
        ///     '\&lt;'	- little-endian<br></br>
        ///     '&gt;' - big-endian<br></br>
        ///     '|'	- ignore(no change to byte order)<br></br>
        ///     The code does a case-insensitive check on the first letter of new_order for these alternatives.<br></br>For example, any of ‘>’ or ‘B’ or ‘b’ or ‘brian’ are valid to specify big-endian.
        /// </param>
        /// <returns>New dtype object with the given change to the byte order.</returns>
        public DType newbyteorder(char new_order = 'S')
        {
            throw new NotSupportedException();
        }
    }

    public static partial class np
    {
        /// <summary>
        ///     Return the character for the minimum-size type to which given types can be safely cast.
        ///     The returned type character must represent the smallest size dtype such that an array of the returned type can handle the data from an array of all types in typechars(or if typechars is an array, then its dtype.char).
        /// </summary>
        /// <param name="typechars">every character represents a type. see <see cref="DType.@char"/></param>
        /// <param name="typeset">The set of characters that the returned character is chosen from. The default set is ‘GDFgdf’.</param>
        /// <param name="default">The default character, this is returned if none of the characters in typechars matches a character in typeset.</param>
        /// <returns>The character representing the minimum-size type that was found.</returns>
        public static char mintypecode(string typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            const string _typecodes_by_elsize = "GDFgdfQqLlIiHhBb?";

            var chars = typechars.ToCharArray();
            var intersect = chars.Intersect(typeset.ToCharArray()).ToArray();
            if (intersect.Length == 0)
                return @default;
            if (intersect.Contains('F') && intersect.Contains('d'))
                return 'D';

            return intersect.OrderBy(c => _typecodes_by_elsize.IndexOf(c)).First();
        }

        /// <summary>
        ///     Return the character for the minimum-size type to which given types can be safely cast.
        ///     The returned type character must represent the smallest size dtype such that an array of the returned type can handle the data from an array of all types in typechars(or if typechars is an array, then its dtype.char).
        /// </summary>
        /// <param name="typechars"></param>
        /// <param name="typeset">The set of characters that the returned character is chosen from. The default set is ‘GDFgdf’.</param>
        /// <param name="default">The default character, this is returned if none of the characters in typechars matches a character in typeset.</param>
        /// <returns>The character representing the minimum-size type that was found.</returns>
        public static char mintypecode(char[] typechars, string typeset = "GDFgdf", char @default = 'd')
        {
            const string _typecodes_by_elsize = "GDFgdfQqLlIiHhBb?";

            var chars = typechars;
            var intersect = chars.Intersect(typeset.ToCharArray()).ToArray();
            if (intersect.Length == 0)
                return @default;
            if (intersect.Contains('F') && intersect.Contains('d'))
                return 'D';

            return intersect.OrderBy(c => _typecodes_by_elsize.IndexOf(c)).First();
        }

        /// <summary>
        ///     Parse a string into a <see cref="DType"/>.
        /// </summary>
        /// <param name="dtype"></param>
        /// <returns>A <see cref="DType"/> based on <paramref name="dtype"/>, return can be null.</returns>
        /// <remarks>
        ///     https://docs.scipy.org/doc/numpy-1.16.0/reference/arrays.dtypes.html <br></br>
        ///     This was created to ease the porting of C++ numpy to C#.
        /// </remarks>
        public static DType dtype(string dtype)
        {
            //TODO! we parse here the string according to docs and return the relevant dtype.
            const string regex = @"^([\>\<\|S\=]?)([a-zA-Z\?]+)(\d+)?";

            if (dtype.Contains("("))
                throw new NotSupportedException("NumSharp does not support custom nested array dtypes");

            if (Enum.TryParse<NPTypeCode>(dtype, out var code))
            {
                switch (code)
                {
#if _REGEN1
	                %foreach all_dtypes%
	                case NPTypeCode.#1: return new DType(typeof(#1));
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Boolean: return new DType(typeof(Boolean));
	                case NPTypeCode.Byte: return new DType(typeof(Byte));
	                case NPTypeCode.Int32: return new DType(typeof(Int32));
	                case NPTypeCode.Int64: return new DType(typeof(Int64));
	                case NPTypeCode.Single: return new DType(typeof(Single));
	                case NPTypeCode.Double: return new DType(typeof(Double));
	                case NPTypeCode.String: return new DType(typeof(String));
	                default:
		                throw new NotSupportedException();
#endif
                }


            }

            var match = Regex.Match(dtype, regex);
            if (!match.Success)
                return null;

            var byteorder = match.Groups[1].Value;
            var type = match.Groups[2].Value;
            var size_str = match.Groups[3].Value?.Trim();

            if (string.IsNullOrEmpty(size_str))
                size_str = "-1";
            int size = int.Parse(size_str);

            //sizeless types
            switch (type)
            {
                case "c":
                case "complex":
                case "Complex":
                    return new DType(typeof(Complex));
                case "string":
                case "chars":
                case "char":
                case "S":
                case "U":
                    return new DType(typeof(char));
                case "b":
                case "byte":
                case "Byte":
                    return new DType(typeof(byte));
                case "bool":
                case "Bool":
                case "Boolean":
                case "boolean":
                case "?":
                    return new DType(typeof(bool));
            }

            //size-specific
            switch (size)
            {
                case -1:
                    switch (type)
                    {
                        case "i":
                        case "int":
                            return new DType(typeof(Int32));
                        case "u":
                        case "uint":
                            return new DType(typeof(UInt32));
                        case "f":
                        case "float":
                        case "single":
                        case "Float":
                        case "Single":
                            return new DType(typeof(float));
                        case "d":
                        case "double":
                        case "Double":
                            return new DType(typeof(double));
                    }

                    break;
                case 1:
                    switch (type)
                    {
                        case "?":
                            return new DType(typeof(bool));
                        case "b":
                        case "i":
                        case "int":
                        case "Int":
                            return new DType(typeof(byte));
                        case "u":
                        case "uint":
                        case "Uint":
                            return new DType(typeof(UInt16));
                    }

                    break;
                case 2:
                    switch (type)
                    {
                        case "i":
                        case "int":
                        case "Int":
                            return new DType(typeof(Int16));
                        case "u":
                        case "uint":
                        case "Uint":
                            return new DType(typeof(UInt16));
                        case "f":
                        case "float":
                        case "Float":
                        case "single":
                        case "Single":
                            return new DType(typeof(float));
                    }

                    break;
                case 4:
                    switch (type)
                    {
                        case "i":
                        case "int":
                            return new DType(typeof(Int32));
                        case "u":
                        case "uint":
                            return new DType(typeof(UInt32));
                        case "f":
                        case "float":
                        case "single":
                        case "Float":
                        case "Single":
                            return new DType(typeof(float));
                        case "d":
                        case "double":
                        case "Double":
                            return new DType(typeof(double));
                    }

                    break;
                case 8:
                case 16:
                    switch (type)
                    {
                        case "i":
                        case "int":
                        case "Int":
                            return new DType(typeof(Int64));
                        case "u":
                        case "uint":
                        case "Uint":
                            return new DType(typeof(UInt64));
                        case "d":
                        case "f":
                        case "float":
                        case "Float":
                        case "single":
                        case "Single":
                        case "double":
                        case "Double":
                            return new DType(typeof(double));
                    }

                    break;
            }

            throw new NotSupportedException($"NumSharp does not support this specific {type}");
        }
    }

    internal enum NPY_SCALARKIND
    {
        NPY_NOSCALAR = -1,
        NPY_BOOL_SCALAR,
        NPY_INTPOS_SCALAR,
        NPY_INTNEG_SCALAR,
        NPY_FLOAT_SCALAR,
        NPY_COMPLEX_SCALAR,
        NPY_OBJECT_SCALAR
    };

    /// <summary>
    ///     https://docs.scipy.org/doc/numpy-1.16.1/reference/c-api.dtype.html#enumerated-types
    /// </summary>
    internal enum NPY_TYPECHAR
    {
        NPY_BOOLLTR = '?',
        NPY_BYTELTR = 'b',
        NPY_UBYTELTR = 'B',
        NPY_SHORTLTR = 'h',
        NPY_USHORTLTR = 'H',
        NPY_INTLTR = 'i',
        NPY_UINTLTR = 'I',
        NPY_LONGLTR = 'l',
        NPY_ULONGLTR = 'L',
        NPY_LONGLONGLTR = 'q',
        NPY_ULONGLONGLTR = 'Q',
        NPY_HALFLTR = 'e',
        NPY_FLOATLTR = 'f',
        NPY_DOUBLELTR = 'd',
        NPY_LONGDOUBLELTR = 'g',
        NPY_CFLOATLTR = 'F',
        NPY_CDOUBLELTR = 'D',
        NPY_CLONGDOUBLELTR = 'G',
        NPY_OBJECTLTR = 'O',
        NPY_STRINGLTR = 'S',
        NPY_STRINGLTR2 = 'a',
        NPY_UNICODELTR = 'U',
        NPY_VOIDLTR = 'V',
        NPY_DATETIMELTR = 'M',
        NPY_TIMEDELTALTR = 'm',
        NPY_CHARLTR = 'c',

        /*
         * No Descriptor, just a define -- this let's
         * Python users specify an array of integers
         * large enough to hold a pointer on the
         * platform
         */
        NPY_INTPLTR = 'p',
        NPY_UINTPLTR = 'P',

        /*
         * These are for dtype 'kinds', not dtype 'typecodes'
         * as the above are for.
         */
        NPY_GENBOOLLTR = 'b',
        NPY_SIGNEDLTR = 'i',
        NPY_UNSIGNEDLTR = 'u',
        NPY_FLOATINGLTR = 'f',
        NPY_COMPLEXLTR = 'c'
    };

}
