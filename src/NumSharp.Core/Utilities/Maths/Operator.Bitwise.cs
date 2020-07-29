using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities.Maths
{
    internal partial class Operator
    {

        #region Bitwise AND
        // Boolean
        // The type inference is based on Python output for the following type combinations:
        // print((np.array([True], dtype=np.bool) & np.array([True])).dtype)               -> bool   : bool
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.int8)).dtype)   -> int8   : sbyte
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.uint8)).dtype)  -> uint8  : byte
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.int16)).dtype)  -> int16  : short
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.uint16)).dtype) -> uint16 : ushort
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.int32)).dtype)  -> int32  : int
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.uint32)).dtype) -> uint32 : uint
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.int64)).dtype)  -> int64  : long
        // print((np.array([True], dtype=np.bool) & np.array([1], dtype=np.uint64)).dtype) -> uint64 : ulong
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool BitwiseAnd(bool lhs, bool rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static sbyte BitwiseAnd(bool lhs, sbyte rhs) => (sbyte)(Convert.ToSByte(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(bool lhs, byte rhs) => (byte)(Convert.ToByte(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(bool lhs, short rhs) => (short)(Convert.ToInt16(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(bool lhs, ushort rhs) => (ushort)(Convert.ToUInt16(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(bool lhs, int rhs) => Convert.ToInt32(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(bool lhs, uint rhs) => Convert.ToUInt32(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(bool lhs, long rhs) => Convert.ToInt64(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(bool lhs, ulong rhs) => Convert.ToUInt64(lhs) & rhs;

        // SByte
        // print((np.array([1], dtype=np.int8) & np.array([True])).dtype)                # -> int8  : sbyte 
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.int8)).dtype)    # -> int8  : sbyte 
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.uint8)).dtype)   # -> int16 : short 
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.int16)).dtype)   # -> int16 : short  
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.uint16)).dtype)  # -> int32 : int
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.int32)).dtype)   # -> int32 : int  
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.uint32)).dtype)  # -> int64 : long
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.int64)).dtype)   # -> int64 : long  
        // print((np.array([1], dtype=np.int8) & np.array([1], dtype=np.uint64)).dtype)  # -> 'bitwise_and' not supported for the input type
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static sbyte BitwiseAnd(sbyte lhs, bool rhs) => (sbyte)(lhs & Convert.ToSByte(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static sbyte BitwiseAnd(sbyte lhs, sbyte rhs) => (sbyte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(sbyte lhs, byte rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(sbyte lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(sbyte lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(sbyte lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(sbyte lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(sbyte lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(sbyte lhs, ulong rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");

        // Byte
        // print((np.array([1], dtype=np.uint8) & np.array([True])).dtype)                # -> uint8  : byte
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.int8)).dtype)    # -> int16  : short
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.uint8)).dtype)   # -> uint8  : byte
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.int16)).dtype)   # -> int16  : short
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.uint16)).dtype)  # -> uint16 : ushort
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.int32)).dtype)   # -> int32  : int
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.uint32)).dtype)  # -> uint32 : uint
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.int64)).dtype)   # -> int64  : long
        // print((np.array([1], dtype=np.uint8) & np.array([1], dtype=np.uint64)).dtype)  # -> uint64 : ulong
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(byte lhs, bool rhs) => (byte)(lhs & Convert.ToByte(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(byte lhs, sbyte rhs) => (byte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(byte lhs, byte rhs) => (byte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(byte lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(byte lhs, ushort rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(byte lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(byte lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(byte lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(byte lhs, ulong rhs) => lhs & rhs;

        // Int16
        // print((np.array([1], dtype=np.int16) & np.array([True])).dtype)                # -> int16 : short
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.int8)).dtype)    # -> int16 : short
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.uint8)).dtype)   # -> int16 : short
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.int16)).dtype)   # -> int16 : short
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.uint16)).dtype)  # -> int32 : int
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.int32)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.uint32)).dtype)  # -> int64 : long
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.int64)).dtype)   # -> int64 : long
        // print((np.array([1], dtype=np.int16) & np.array([1], dtype=np.uint64)).dtype)  # -> 'bitwise_and' not supported for the input type
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, bool rhs) => (short)(lhs & Convert.ToInt16(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, sbyte rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, byte rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(short lhs, ushort rhs) => (lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(short lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(short lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(short lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(short lhs, ulong rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");

        // UInt16
        // print((np.array([1], dtype=np.uint16) & np.array([True])).dtype)                # -> uint16 : ushort
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.int8)).dtype)    # -> int32 : int
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.uint8)).dtype)   # -> uint16 : ushort
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.int16)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.uint16)).dtype)  # -> uint16 : ushort
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.int32)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.uint32)).dtype)  # -> uint32 : uint
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.int64)).dtype)   # -> int64 : long
        // print((np.array([1], dtype=np.uint16) & np.array([1], dtype=np.uint64)).dtype)  # -> uint64 : ulong
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, bool rhs) => (ushort)(lhs & Convert.ToUInt16(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(ushort lhs, sbyte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, byte rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(ushort lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, ushort rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(ushort lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(ushort lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(ushort lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ushort lhs, ulong rhs) => lhs & rhs;

        // Int32
        // print((np.array([1], dtype=np.int32) & np.array([True])).dtype)                # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.int8)).dtype)    # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.uint8)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.int16)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.uint16)).dtype)  # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.int32)).dtype)   # -> int32 : int
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.uint32)).dtype)  # -> int64 : long
        // print((np.array([1], dtype=np.int32) & np.array([1], dtype=np.int64)).dtype)   # -> int64 : long
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, bool rhs) => lhs & Convert.ToInt32(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, sbyte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(int lhs, uint rhs) => lhs & (int)rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(int lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(int lhs, ulong rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");

        // UInt32
        // print((np.array([1], dtype=np.uint32) & np.array([True])).dtype)                # -> uint32 : uint
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.int8)).dtype)    # -> int64  : long
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.uint8)).dtype)   # -> uint32 : uint
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.int16)).dtype)   # -> int64  : long
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.uint16)).dtype)  # -> uint32 : uint
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.int32)).dtype)   # -> int64  : long
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.uint32)).dtype)  # -> uint32 : uint
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.int64)).dtype)   # -> int64  : long
        // print((np.array([1], dtype=np.uint32) & np.array([1], dtype=np.uint64)).dtype)  # -> uint64 : ulong
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, bool rhs) => lhs & Convert.ToUInt32(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, sbyte rhs) => lhs & (uint)rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, int rhs) => (int)lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(uint lhs, ulong rhs) => lhs & rhs;

        // Int64
        // print((np.array([1], dtype=np.int64) & np.array([True])).dtype)                # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.int8)).dtype)    # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.uint8)).dtype)   # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.int16)).dtype)   # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.uint16)).dtype)  # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.int32)).dtype)   # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.uint32)).dtype)  # -> int64 : long
        // print((np.array([1], dtype=np.int64) & np.array([1], dtype=np.int64)).dtype)   # -> int64 : long
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, bool rhs) => lhs & Convert.ToInt64(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, sbyte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, ulong rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");

        // UInt64
        // print((np.array([1], dtype=np.uint64) & np.array([True])).dtype)                # -> uint64 : ulong
        // print((np.array([1], dtype=np.uint64) & np.array([1], dtype=np.uint8)).dtype)   # -> uint64 : ulong
        // print((np.array([1], dtype=np.uint64) & np.array([1], dtype=np.uint16)).dtype)  # -> uint64 : ulong
        // print((np.array([1], dtype=np.uint64) & np.array([1], dtype=np.uint32)).dtype)  # -> uint64 : ulong
        // print((np.array([1], dtype=np.uint64) & np.array([1], dtype=np.uint64)).dtype)  # -> uint64 : ulong
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, bool rhs) => lhs & Convert.ToUInt64(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, sbyte rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(ulong lhs, short rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, ushort rhs) => (lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(ulong lhs, int rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, long rhs) => throw new NotSupportedException("The operator is 'bitwise_and' not supported for the input types.");
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, ulong rhs) => lhs & rhs;
        #endregion

        public static BinaryOperatorIndex OpBitwiseAnd = new BinaryOperatorIndex(typeof(Operator), nameof(BitwiseAnd));
    }
}
