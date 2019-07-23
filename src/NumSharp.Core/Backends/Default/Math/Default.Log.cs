using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Log(in NDArray nd, NPTypeCode typeCode) => Log(nd, typeCode.AsType());

        public override NDArray Log(in NDArray nd, Type dtype = null)
        {
            if (nd.size == 0)
                return nd.Clone(); 

            var @out = Cast(nd, dtype ?? nd.dtype, true);
            var len = @out.size;

            unsafe
            {
                switch (@out.GetTypeCode)
                {
#if _REGEN
                    %foreach supported_numericals,supported_numericals_lowercase%
	                case NPTypeCode.#1:
	                {
                        var out_addr = (#2*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.To#1(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                %
	                default:
		                throw new NotSupportedException();
#else
	                case NPTypeCode.Byte:
	                {
                        var out_addr = (byte*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToByte(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Int16:
	                {
                        var out_addr = (short*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt16(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.UInt16:
	                {
                        var out_addr = (ushort*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt16(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Int32:
	                {
                        var out_addr = (int*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt32(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.UInt32:
	                {
                        var out_addr = (uint*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt32(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Int64:
	                {
                        var out_addr = (long*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToInt64(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.UInt64:
	                {
                        var out_addr = (ulong*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToUInt64(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Char:
	                {
                        var out_addr = (char*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToChar(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Double:
	                {
                        var out_addr = (double*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToDouble(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Single:
	                {
                        var out_addr = (float*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToSingle(Math.Log(*(out_addr + i)));

                        return @out;
	                }
	                case NPTypeCode.Decimal:
	                {
                        var out_addr = (decimal*)@out.Address;
                        for (int i = 0; i < len; i++)
                            *(out_addr + i) = Convert.ToDecimal(Log(*(out_addr + i)));

                        return @out;
	                }
	                default:
		                throw new NotSupportedException();
#endif
                }

                decimal Log(decimal d)
                {
                    const decimal Ln10 = 2.3025850929940456840179914547m;
                    if (d < 0) throw new ArgumentException("Natural logarithm is a complex number for values less than zero!", "d");
                    if (d == 0) throw new OverflowException("Natural logarithm is defined as negative infinity at zero which the Decimal data type can't represent!");

                    if (d == 1) return 0;

                    if (d >= 1)
                    {
                        var power = 0m;

                        var x = d;
                        while (x > 1)
                        {
                            x /= 10;
                            power += 1;
                        }

                        return Log(x) + power * Ln10;
                    }

                    // See http://en.wikipedia.org/wiki/Natural_logarithm#Numerical_value
                    // for more information on this faster-converging series.

                    decimal y;
                    decimal ySquared;

                    var iteration = 0;
                    var exponent = 0m;
                    var nextAdd = 0m;
                    var result = 0m;

                    y = (d - 1) / (d + 1);
                    ySquared = y * y;

                    while (true)
                    {
                        if (iteration == 0)
                        {
                            exponent = 2 * y;
                        }
                        else
                        {
                            exponent = exponent * ySquared;
                        }

                        nextAdd = exponent / (2 * iteration + 1);

                        if (nextAdd == 0) break;

                        result += nextAdd;

                        iteration += 1;
                    }

                    return result;
                }
            }
        }
    }
}
