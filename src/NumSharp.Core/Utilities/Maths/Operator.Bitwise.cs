using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities.Maths
{
    internal partial class Operator
    {
        #region Bitwise AND
        // Boolean
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool BitwiseAnd(bool lhs, bool rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char BitwiseAnd(bool lhs, char rhs) => (char)(Convert.ToInt32(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(bool lhs, byte rhs) => (byte)(Convert.ToUInt32(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(bool lhs, short rhs) => (short)(Convert.ToInt32(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(bool lhs, ushort rhs) => (ushort)(Convert.ToUInt32(lhs) & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(bool lhs, int rhs) => Convert.ToInt32(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(bool lhs, uint rhs) => Convert.ToUInt32(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(bool lhs, long rhs) => Convert.ToInt64(lhs) & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(bool lhs, ulong rhs) => Convert.ToUInt64(lhs) & rhs;

        // Int8
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char BitwiseAnd(char lhs, bool rhs) => (char)(lhs & Convert.ToInt32(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char BitwiseAnd(char lhs, char rhs) => (char)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(char lhs, byte rhs) => (byte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(char lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(char lhs, ushort rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(char lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(char lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(char lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(char lhs, ulong rhs) => lhs & rhs;

        // Char
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(byte lhs, bool rhs) => (byte)(lhs & Convert.ToUInt32(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(byte lhs, char rhs) => (byte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte BitwiseAnd(byte lhs, byte rhs) => (byte)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(byte lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(byte lhs, ushort rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(byte lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(byte lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(byte lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(byte lhs, ulong rhs) => lhs & rhs;

        // Byte
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, bool rhs) => (short)(lhs & Convert.ToInt32(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, char rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, byte rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short BitwiseAnd(short lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(short lhs, ushort rhs) => (lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(short lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(short lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(short lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(short lhs, ulong rhs) => (ulong)lhs & rhs;

        // UInt16
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, bool rhs) => (ushort)(lhs & Convert.ToUInt32(rhs));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, char rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, byte rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(ushort lhs, short rhs) => (short)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort BitwiseAnd(ushort lhs, ushort rhs) => (ushort)(lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(ushort lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(ushort lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(ushort lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ushort lhs, ulong rhs) => lhs & rhs;

        // Int32
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, bool rhs) => lhs & Convert.ToInt32(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, char rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int BitwiseAnd(int lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(int lhs, uint rhs) => lhs & (int)rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(int lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(int lhs, ulong rhs) => (ulong)lhs & rhs;

        // UInt32
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, bool rhs) => lhs & Convert.ToUInt32(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, char rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, int rhs) => (int)lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint BitwiseAnd(uint lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(uint lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(uint lhs, ulong rhs) => lhs & rhs;

        // Int64
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, bool rhs) => lhs & Convert.ToInt64(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, char rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, short rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, ushort rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, int rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long BitwiseAnd(long lhs, long rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(long lhs, ulong rhs) => lhs & (long)rhs;

        // UInt64
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, bool rhs) => lhs & Convert.ToUInt64(rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, char rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, byte rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(ulong lhs, short rhs) => lhs & (ulong)rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, ushort rhs) => (lhs & rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(ulong lhs, int rhs) => lhs & (ulong)rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, uint rhs) => lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double BitwiseAnd(ulong lhs, long rhs) => (long)lhs & rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong BitwiseAnd(ulong lhs, ulong rhs) => lhs & rhs;
        #endregion

        public static BinaryOperatorIndex OpBitwiseAnd = new BinaryOperatorIndex(typeof(Operator), nameof(BitwiseAnd));
    }
}
