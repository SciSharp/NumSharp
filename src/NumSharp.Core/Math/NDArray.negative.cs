using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Numerical negative, element-wise.
        /// Returns -x for each element (negates ALL values, not just positive).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.negative.html</remarks>
        public NDArray negative()
        {
            // NumPy rejects boolean negative (np.negative(bool) / unary -): there
            // is no negative loop for the bool dtype, even for empty arrays. Use
            // the `~` operator (np.invert) or np.logical_not for a boolean flip.
            if (this.GetTypeCode == NPTypeCode.Boolean)
                throw new NotSupportedException(
                    "The numpy boolean negative, the `-` operator, is not supported, " +
                    "use the `~` operator or the logical_not function instead.");

            if (this.size == 0)
                return this.Clone();

            var @out = TensorEngine.Cast(this, dtype ?? this.dtype, copy: true);

            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
                    // NPTypeCode.Boolean is rejected up-front (see guard above);
                    // NumPy has no boolean negative loop.
                    // %foreach supported_numericals_signed,supported_numericals_signed_lowercase%
	                // case NPTypeCode.#1:
	                // {
                        // var out_addr = (#2*)@out.Address;
                        // for (int i = 0; i < len; i++)
                            // out_addr[i] = (#2)(-out_addr[i]);
                        // return @out;
	                // }
	                // %
                    // %foreach supported_numericals_unsigned,supported_numericals_unsigned_lowercase,supported_numericals_unsigned_defaultvals
	                // case NPTypeCode.#1:
	                // default:
		                // throw new NotSupportedException();
	                case NPTypeCode.SByte:
	                {
                        var out_addr = (sbyte*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = (sbyte)(-out_addr[i]);
                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var out_addr = (short*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = (short)(-out_addr[i]);
                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Half:
	                {
                        var out_addr = (Half*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Complex:
	                {
                        var out_addr = (System.Numerics.Complex*)@out.Address;
                        for (long i = 0; i < len; i++)
                            out_addr[i] = -out_addr[i];
                        return @out;
	                }
	                case NPTypeCode.Byte:
	                case NPTypeCode.UInt16:
	                case NPTypeCode.UInt32:
	                case NPTypeCode.UInt64:
	                case NPTypeCode.Char:
	                default:
		                throw new NotSupportedException();
                }
            }
        }
    }
}
