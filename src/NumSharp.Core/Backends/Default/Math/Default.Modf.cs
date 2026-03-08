using System;
using System.Threading.Tasks;
using DecimalMath;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, Type dtype) => ModF(nd, dtype?.GetTypeCode());

        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, NPTypeCode? typeCode = null)
        {
            var @out = Cast(nd, typeCode ?? nd.typecode, copy: true);
            var @out1 = Cast(nd, typeCode ?? nd.typecode, copy: true);
            var len = @out.size;

            // Use SIMD-optimized path for contiguous float/double arrays
            if (@out.Shape.IsContiguous && ILKernelGenerator.Enabled)
            {
                unsafe
                {
                    switch (@out.GetTypeCode)
                    {
                        case NPTypeCode.Double:
                        {
                            ILKernelGenerator.ModfHelper((double*)@out.Address, (double*)@out1.Address, len);
                            return (@out, @out1);
                        }
                        case NPTypeCode.Single:
                        {
                            ILKernelGenerator.ModfHelper((float*)@out.Address, (float*)@out1.Address, len);
                            return (@out, @out1);
                        }
                    }
                }
            }

            // Fallback path (non-contiguous or decimal)
            unsafe
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Double:
                    {
                        var out_addr = (double*)@out.Address;
                        var out1_addr = (double*)@out1.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var trunc = Math.Truncate(out_addr[i]);
                            out_addr[i] = Converts.ToDouble(out_addr[i] - trunc);
                            *(out1_addr + i) = trunc;
                        }

                        return (@out, @out1);
                    }
                    case NPTypeCode.Single:
                    {
                        var out_addr = (float*)@out.Address;
                        var out1_addr = (float*)@out1.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var trunc = Math.Truncate(out_addr[i]);
                            out_addr[i] = Converts.ToSingle(out_addr[i] - trunc);
                            *(out1_addr + i) = Convert.ToSingle(trunc);
                        }

                        return (@out, @out1);
                    }
                    case NPTypeCode.Decimal:
                    {
                        var out_addr = (decimal*)@out.Address;
                        var out1_addr = (decimal*)@out1.Address;
                        for (int i = 0; i < len; i++)
                        {
                            var trunc = Math.Truncate(out_addr[i]);
                            out_addr[i] = out_addr[i] - trunc;
                            *(out1_addr + i) = trunc;
                        }

                        return (@out, @out1);
                    }
                    default:
                        throw new NotSupportedException();
                }
            }
        }
    }
}
