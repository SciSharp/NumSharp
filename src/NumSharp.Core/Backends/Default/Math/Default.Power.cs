using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, Type dtype)
            => Power(lhs, rhs, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// Uses ExecuteBinaryOp with BinaryOp.Power for broadcasting support.
        /// NumPy rule: for integer types, the result wraps modulo the dtype range
        /// (not promoted through double, which loses precision for large exponents).
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null)
        {
            // NumPy integer pow wraps modulo the dtype range. The existing IL kernel
            // routes through Math.Pow(double, double) and loses precision for values
            // outside [-2^52, 2^52] (large int^int results). Use native integer
            // exponentiation for same-dtype integer inputs to preserve wrapping.
            if (!typeCode.HasValue
                && lhs.GetTypeCode == rhs.GetTypeCode
                && lhs.GetTypeCode.IsInteger()
                && lhs.shape.SequenceEqual(rhs.shape))
            {
                return PowerInteger(lhs, rhs);
            }

            var result = ExecuteBinaryOp(lhs, rhs, BinaryOp.Power);
            if (typeCode.HasValue && result.typecode != typeCode.Value)
                return Cast(result, typeCode.Value, copy: false);
            return result;
        }

        /// <summary>
        /// NumPy-style integer exponentiation with dtype wraparound. Matches NumPy:
        ///   - negative exponent with |base| > 1: 0 (integer reciprocal truncation)
        ///   - negative exponent with base == 1: 1
        ///   - negative exponent with base == -1: ±1 based on exp parity
        ///   - negative exponent with base == 0: NumPy raises "0 cannot be raised to a negative power"
        ///     but with seterr=ignore it returns 0; we return 0 to match seterr=ignore behavior.
        ///   - non-negative exponent: repeated multiplication with native wrapping.
        /// </summary>
        private static NDArray PowerInteger(NDArray lhs, NDArray rhs)
        {
            var tc = lhs.GetTypeCode;
            var result = new NDArray(tc, new Shape((long[])lhs.shape.Clone()), false);
            long n = lhs.size;
            unsafe
            {
                switch (tc)
                {
                    case NPTypeCode.SByte:
                    {
                        var a = (sbyte*)lhs.Unsafe.Address;
                        var b = (sbyte*)rhs.Unsafe.Address;
                        var d = (sbyte*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowSByte(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.Byte:
                    {
                        var a = (byte*)lhs.Unsafe.Address;
                        var b = (byte*)rhs.Unsafe.Address;
                        var d = (byte*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowByte(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.Int16:
                    {
                        var a = (short*)lhs.Unsafe.Address;
                        var b = (short*)rhs.Unsafe.Address;
                        var d = (short*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowInt16(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.UInt16:
                    {
                        var a = (ushort*)lhs.Unsafe.Address;
                        var b = (ushort*)rhs.Unsafe.Address;
                        var d = (ushort*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowUInt16(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.Int32:
                    {
                        var a = (int*)lhs.Unsafe.Address;
                        var b = (int*)rhs.Unsafe.Address;
                        var d = (int*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowInt32(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.UInt32:
                    {
                        var a = (uint*)lhs.Unsafe.Address;
                        var b = (uint*)rhs.Unsafe.Address;
                        var d = (uint*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowUInt32(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.Int64:
                    {
                        var a = (long*)lhs.Unsafe.Address;
                        var b = (long*)rhs.Unsafe.Address;
                        var d = (long*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowInt64(a[i], b[i]);
                        break;
                    }
                    case NPTypeCode.UInt64:
                    {
                        var a = (ulong*)lhs.Unsafe.Address;
                        var b = (ulong*)rhs.Unsafe.Address;
                        var d = (ulong*)result.Unsafe.Address;
                        for (long i = 0; i < n; i++) d[i] = PowUInt64(a[i], b[i]);
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Integer power not supported for {tc}");
                }
            }
            return result;
        }

        // Core repeated-squaring with native wrapping. Exponents cast to long to avoid
        // signed-overflow issues inside the loop counter.
        private static sbyte PowSByte(sbyte a, sbyte b)
        {
            if (b < 0) return a == 1 ? (sbyte)1 : a == -1 ? ((b & 1) == 0 ? (sbyte)1 : (sbyte)-1) : (sbyte)0;
            sbyte r = 1;
            sbyte x = a;
            long e = b;
            unchecked
            {
                while (e > 0) { if ((e & 1) == 1) r = (sbyte)(r * x); e >>= 1; if (e > 0) x = (sbyte)(x * x); }
            }
            return r;
        }
        private static byte PowByte(byte a, byte b)
        {
            byte r = 1, x = a; long e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = (byte)(r * x); e >>= 1; if (e > 0) x = (byte)(x * x); } }
            return r;
        }
        private static short PowInt16(short a, short b)
        {
            if (b < 0) return a == 1 ? (short)1 : a == -1 ? ((b & 1) == 0 ? (short)1 : (short)-1) : (short)0;
            short r = 1, x = a; long e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = (short)(r * x); e >>= 1; if (e > 0) x = (short)(x * x); } }
            return r;
        }
        private static ushort PowUInt16(ushort a, ushort b)
        {
            ushort r = 1, x = a; long e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = (ushort)(r * x); e >>= 1; if (e > 0) x = (ushort)(x * x); } }
            return r;
        }
        private static int PowInt32(int a, int b)
        {
            if (b < 0) return a == 1 ? 1 : a == -1 ? ((b & 1) == 0 ? 1 : -1) : 0;
            int r = 1, x = a; long e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = r * x; e >>= 1; if (e > 0) x = x * x; } }
            return r;
        }
        private static uint PowUInt32(uint a, uint b)
        {
            uint r = 1, x = a; long e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = r * x; e >>= 1; if (e > 0) x = x * x; } }
            return r;
        }
        private static long PowInt64(long a, long b)
        {
            if (b < 0) return a == 1 ? 1L : a == -1 ? ((b & 1) == 0 ? 1L : -1L) : 0L;
            long r = 1, x = a, e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = r * x; e >>= 1; if (e > 0) x = x * x; } }
            return r;
        }
        private static ulong PowUInt64(ulong a, ulong b)
        {
            ulong r = 1, x = a, e = b;
            unchecked { while (e > 0) { if ((e & 1) == 1) r = r * x; e >>= 1; if (e > 0) x = x * x; } }
            return r;
        }
    }
}
