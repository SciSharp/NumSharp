using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities.Maths
{
    /// <summary>
    ///     A class that provides math operations that return the exact expected type according to numpy.
    /// </summary>
    [SuppressMessage("ReSharper", "UnusedMember.Local")]
    [SuppressMessage("ReSharper", "ArrangeTypeMemberModifiers")]
    internal class Operator
    {
        //all other gen
#if _REGEN //We manually fixed types that naturally do not match with casting.
        %import "C:\Users\Eli-PC\Desktop\SciSharp\NumSharp\src\NumSharp.Core\bin\Debug\netstandard2.0\NumSharp.Core.dll"
        %import NumSharp.np as np
        %ops = ["+", "-", "%", "*", "/"]
        %names = ["Add", "Subtract", "Mod", "Multiply", "Divide"]
        
        %foreach ops,names%
        %foreach supported_numericals,supported_numericals_lowercase%
        %foreach supported_numericals,supported_numericals_lowercase%
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static #(np.find_common_type(#101.ToString(), #201.ToString())) #2(#101 lhs, #201 rhs) => lhs #1 rhs;
        %
        %
        %
#else

#endif

        //boolean gen
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool Add(bool lhs, bool rhs) => (lhs || rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool Subtract(bool lhs, bool rhs) => ((lhs ? 1 : 0) - (rhs ? 1 : 0)) != 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool Multiply(bool lhs, bool rhs) => lhs && rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool Mod(bool lhs, bool rhs) => ((lhs ? 1 : 0) % (rhs ? 1 : 0)) != 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool Divide(bool lhs, bool rhs) => lhs && rhs;

        //boolean gen
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int AddBoolean(bool lhs, bool rhs) => (lhs || rhs) ? 1 : 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int SubtractBoolean(bool lhs, bool rhs) => (((lhs ? 1 : 0) - (rhs ? 1 : 0)) != 0)  ? 1 : 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int MultiplyBoolean(bool lhs, bool rhs) => (lhs && rhs) ? 1 : 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int ModBoolean(bool lhs, bool rhs) => (((lhs ? 1 : 0) % (rhs ? 1 : 0)) != 0) ? 1 : 0;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int DivideBoolean(bool lhs, bool rhs) => (lhs && rhs) ? 1 : 0;


#if _REGEN //We manually fixed types that naturally do not match with casting.
        %import "C:\Users\Eli-PC\Desktop\SciSharp\NumSharp\src\NumSharp.Core\bin\Debug\netstandard2.0\NumSharp.Core.dll"
        %import NumSharp.np as np

        %ops = ["+", "-", "%", "*", "/"]
        %names = ["Add", "Subtract", "Mod", "Multiply", "Divide"]
        //Add, Subtract, Mod, Multiply and Divide Booleanic Operators
        //bool is lhs
        %foreach ops,names%
        %foreach ["Boolean"],["bool"]%
        %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals,supported_numericals_onevales%
        |#rettype = np.find_common_type("#101", "#201")
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static #(rettype) #2(#101 lhs, #201 rhs) => (#(rettype)) ((lhs ? #204 : #203) #1 rhs);
        %
        %
        %      

        //bool is rhs
        %foreach ops,names%
        %foreach supported_numericals,supported_numericals_lowercase,supported_numericals_defaultvals,supported_numericals_onevales%
        %foreach ["Boolean"],["bool"]%
        |#rettype = np.find_common_type("#101", "#201")
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static #(rettype) #2(#101 lhs, #201 rhs) => (#(rettype)) (lhs #1 (rhs ? #104 : #103));
        %
        %
        %      
#else

        //Add, Subtract, Mod, Multiply and Divide Booleanic Operators
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Add(bool lhs, byte rhs) => (byte) ((lhs ? 1 : 0) + rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Add(bool lhs, short rhs) => (short) ((lhs ? 1 : 0) + rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Add(bool lhs, ushort rhs) => (ushort) ((lhs ? 1 : 0) + rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Add(bool lhs, int rhs) => (lhs ? 1 : 0) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Add(bool lhs, uint rhs) => (lhs ? 1u : 0u) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Add(bool lhs, long rhs) => (lhs ? 1L : 0L) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Add(bool lhs, ulong rhs) => (lhs ? 1UL : 0UL) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Add(bool lhs, char rhs) => (char) ((lhs ? 1 : 0) + rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Add(bool lhs, double rhs) => (lhs ? 1d : 0d) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Add(bool lhs, float rhs) => (lhs ? 1f : 0f) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Add(bool lhs, decimal rhs) => (lhs ? 1m : 0m) + rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Subtract(bool lhs, byte rhs) => (byte) ((lhs ? 1 : 0) - rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Subtract(bool lhs, short rhs) => (short) ((lhs ? 1 : 0) - rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Subtract(bool lhs, ushort rhs) => (ushort) ((lhs ? 1 : 0) - rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Subtract(bool lhs, int rhs) => (lhs ? 1 : 0) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Subtract(bool lhs, uint rhs) => (lhs ? 1u : 0u) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Subtract(bool lhs, long rhs) => (lhs ? 1L : 0L) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Subtract(bool lhs, ulong rhs) => (lhs ? 1UL : 0UL) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Subtract(bool lhs, char rhs) => (char) ((lhs ? 1 : 0) - rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Subtract(bool lhs, double rhs) => (lhs ? 1d : 0d) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Subtract(bool lhs, float rhs) => (lhs ? 1f : 0f) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Subtract(bool lhs, decimal rhs) => (lhs ? 1m : 0m) - rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Mod(bool lhs, byte rhs) => (byte) ((lhs ? 1 : 0) % rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Mod(bool lhs, short rhs) => (short) ((lhs ? 1 : 0) % rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Mod(bool lhs, ushort rhs) => (ushort) ((lhs ? 1 : 0) % rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Mod(bool lhs, int rhs) => (lhs ? 1 : 0) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Mod(bool lhs, uint rhs) => (lhs ? 1u : 0u) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Mod(bool lhs, long rhs) => (lhs ? 1L : 0L) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Mod(bool lhs, ulong rhs) => (lhs ? 1UL : 0UL) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Mod(bool lhs, char rhs) => (char) ((lhs ? 1 : 0) % rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Mod(bool lhs, double rhs) => (lhs ? 1d : 0d) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Mod(bool lhs, float rhs) => (lhs ? 1f : 0f) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Mod(bool lhs, decimal rhs) => (lhs ? 1m : 0m) % rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Multiply(bool lhs, byte rhs) => (byte) ((lhs ? 1 : 0) * rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Multiply(bool lhs, short rhs) => (short) ((lhs ? 1 : 0) * rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Multiply(bool lhs, ushort rhs) => (ushort) ((lhs ? 1 : 0) * rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Multiply(bool lhs, int rhs) => (lhs ? 1 : 0) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Multiply(bool lhs, uint rhs) => (lhs ? 1u : 0u) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Multiply(bool lhs, long rhs) => (lhs ? 1L : 0L) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Multiply(bool lhs, ulong rhs) => (lhs ? 1UL : 0UL) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Multiply(bool lhs, char rhs) => (char) ((lhs ? 1 : 0) * rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Multiply(bool lhs, double rhs) => (lhs ? 1d : 0d) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Multiply(bool lhs, float rhs) => (lhs ? 1f : 0f) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Multiply(bool lhs, decimal rhs) => (lhs ? 1m : 0m) * rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Divide(bool lhs, byte rhs) => (byte) ((lhs ? 1 : 0) / rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Divide(bool lhs, short rhs) => (short) ((lhs ? 1 : 0) / rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Divide(bool lhs, ushort rhs) => (ushort) ((lhs ? 1 : 0) / rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Divide(bool lhs, int rhs) => (lhs ? 1 : 0) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Divide(bool lhs, uint rhs) => (lhs ? 1u : 0u) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Divide(bool lhs, long rhs) => (lhs ? 1L : 0L) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Divide(bool lhs, ulong rhs) => (lhs ? 1UL : 0UL) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Divide(bool lhs, char rhs) => (char) ((lhs ? 1 : 0) / rhs);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Divide(bool lhs, double rhs) => (lhs ? 1d : 0d) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Divide(bool lhs, float rhs) => (lhs ? 1f : 0f) / rhs;
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Divide(bool lhs, decimal rhs) => (lhs ? 1m : 0m) / rhs;

        //bool is rhs
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Add(byte lhs, bool rhs) => (byte) (lhs + (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Add(short lhs, bool rhs) => (short) (lhs + (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Add(ushort lhs, bool rhs) => (ushort) (lhs + (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Add(int lhs, bool rhs) => lhs + (rhs ? 1 : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Add(uint lhs, bool rhs) => lhs + (rhs ? 1u : 0u);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Add(long lhs, bool rhs) => lhs + (rhs ? 1L : 0L);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Add(ulong lhs, bool rhs) => lhs + (rhs ? 1UL : 0UL);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Add(char lhs, bool rhs) => (char) (lhs + (rhs ? 1 : '\0'));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Add(double lhs, bool rhs) => lhs + (rhs ? 1d : 0d);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Add(float lhs, bool rhs) => lhs + (rhs ? 1f : 0f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Add(decimal lhs, bool rhs) => lhs + (rhs ? 1m : 0m);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Subtract(byte lhs, bool rhs) => (byte) (lhs - (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Subtract(short lhs, bool rhs) => (short) (lhs - (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Subtract(ushort lhs, bool rhs) => (ushort) (lhs - (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Subtract(int lhs, bool rhs) => lhs - (rhs ? 1 : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Subtract(uint lhs, bool rhs) => lhs - (rhs ? 1u : 0u);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Subtract(long lhs, bool rhs) => lhs - (rhs ? 1L : 0L);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Subtract(ulong lhs, bool rhs) => lhs - (rhs ? 1UL : 0UL);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Subtract(char lhs, bool rhs) => (char) (lhs - (rhs ? 1 : '\0'));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Subtract(double lhs, bool rhs) => lhs - (rhs ? 1d : 0d);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Subtract(float lhs, bool rhs) => lhs - (rhs ? 1f : 0f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Subtract(decimal lhs, bool rhs) => lhs - (rhs ? 1m : 0m);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Mod(byte lhs, bool rhs) => (byte) (lhs % (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Mod(short lhs, bool rhs) => (short) (lhs % (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Mod(ushort lhs, bool rhs) => (ushort) (lhs % (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Mod(int lhs, bool rhs) => lhs % (rhs ? 1 : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Mod(uint lhs, bool rhs) => lhs % (rhs ? 1u : 0u);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Mod(long lhs, bool rhs) => lhs % (rhs ? 1L : 0L);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Mod(ulong lhs, bool rhs) => lhs % (rhs ? 1UL : 0UL);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Mod(char lhs, bool rhs) => (char) (lhs % (rhs ? 1 : '\0'));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Mod(double lhs, bool rhs) => lhs % (rhs ? 1d : 0d);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Mod(float lhs, bool rhs) => lhs % (rhs ? 1f : 0f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Mod(decimal lhs, bool rhs) => lhs % (rhs ? 1m : 0m);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Multiply(byte lhs, bool rhs) => (byte) (lhs * (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Multiply(short lhs, bool rhs) => (short) (lhs * (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Multiply(ushort lhs, bool rhs) => (ushort) (lhs * (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Multiply(int lhs, bool rhs) => lhs * (rhs ? 1 : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Multiply(uint lhs, bool rhs) => lhs * (rhs ? 1u : 0u);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Multiply(long lhs, bool rhs) => lhs * (rhs ? 1L : 0L);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Multiply(ulong lhs, bool rhs) => lhs * (rhs ? 1UL : 0UL);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Multiply(char lhs, bool rhs) => (char) (lhs * (rhs ? 1 : '\0'));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Multiply(double lhs, bool rhs) => lhs * (rhs ? 1d : 0d);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Multiply(float lhs, bool rhs) => lhs * (rhs ? 1f : 0f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Multiply(decimal lhs, bool rhs) => lhs * (rhs ? 1m : 0m);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static byte Divide(byte lhs, bool rhs) => (byte) (lhs / (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static short Divide(short lhs, bool rhs) => (short) (lhs / (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ushort Divide(ushort lhs, bool rhs) => (ushort) (lhs / (rhs ? 1 : 0));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int Divide(int lhs, bool rhs) => lhs / (rhs ? 1 : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static uint Divide(uint lhs, bool rhs) => lhs / (rhs ? 1u : 0u);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long Divide(long lhs, bool rhs) => lhs / (rhs ? 1L : 0L);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static ulong Divide(ulong lhs, bool rhs) => lhs / (rhs ? 1UL : 0UL);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static char Divide(char lhs, bool rhs) => (char) (lhs / (rhs ? 1 : '\0'));
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Divide(double lhs, bool rhs) => lhs / (rhs ? 1d : 0d);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static float Divide(float lhs, bool rhs) => lhs / (rhs ? 1f : 0f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static decimal Divide(decimal lhs, bool rhs) => lhs / (rhs ? 1m : 0m);
#endif

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Add(byte lhs, byte rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Add(byte lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Add(byte lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(byte lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(byte lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(byte lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(byte lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Add(byte lhs, char rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(byte lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(byte lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(byte lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Add(short lhs, byte rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Add(short lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(short lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(short lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(short lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(short lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(short lhs, ulong rhs) => lhs + (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Add(short lhs, char rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(short lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(short lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(short lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Add(ushort lhs, byte rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(ushort lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Add(ushort lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(ushort lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(ushort lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(ushort lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ushort lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Add(ushort lhs, char rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ushort lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(ushort lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(ushort lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(int lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(int lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(int lhs, ulong rhs) => lhs + (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(int lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(int lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(int lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(uint lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(uint lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(uint lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(uint lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(uint lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(uint lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(uint lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(uint lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(uint lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(uint lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(uint lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(long lhs, ulong rhs) => lhs + (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(long lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(long lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(long lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(long lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ulong lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ulong lhs, short rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ulong lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ulong lhs, int rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ulong lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ulong lhs, long rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ulong lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(ulong lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ulong lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(ulong lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(ulong lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Add(char lhs, byte rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Add(char lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Add(char lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(char lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Add(char lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Add(char lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Add(char lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Add(char lhs, char rhs) => (char)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(char lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(char lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(char lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(double lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(double lhs, decimal rhs) => (decimal)lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(float lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(float lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(float lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(float lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(float lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(float lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(float lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(float lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Add(float lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Add(float lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(float lhs, decimal rhs) => (decimal)lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, double rhs) => lhs + (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, float rhs) => lhs + (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Add(decimal lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Subtract(byte lhs, byte rhs) => (byte)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Subtract(byte lhs, short rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Subtract(byte lhs, ushort rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(byte lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(byte lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(byte lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(byte lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Subtract(byte lhs, char rhs) => (byte)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(byte lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(byte lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(byte lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Subtract(short lhs, byte rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Subtract(short lhs, short rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(short lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(short lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(short lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(short lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(short lhs, ulong rhs) => lhs - (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Subtract(short lhs, char rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(short lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(short lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(short lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Subtract(ushort lhs, byte rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(ushort lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Subtract(ushort lhs, ushort rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(ushort lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(ushort lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(ushort lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ushort lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Subtract(ushort lhs, char rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ushort lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(ushort lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(ushort lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(int lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(int lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(int lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(int lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(int lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(int lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(int lhs, ulong rhs) => lhs - (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(int lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(int lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(int lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(int lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(uint lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(uint lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(uint lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(uint lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(uint lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(uint lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(uint lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(uint lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(uint lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(uint lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(uint lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(long lhs, ulong rhs) => lhs - (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(long lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(long lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(long lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(long lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ulong lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ulong lhs, short rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ulong lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ulong lhs, int rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ulong lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ulong lhs, long rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ulong lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(ulong lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ulong lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(ulong lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(ulong lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Subtract(char lhs, byte rhs) => (byte) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Subtract(char lhs, short rhs) => (short) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Subtract(char lhs, ushort rhs) => (ushort) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Subtract(char lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Subtract(char lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Subtract(char lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Subtract(char lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Subtract(char lhs, char rhs) => (char) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(char lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(char lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(char lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(double lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(double lhs, decimal rhs) => (decimal)lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(float lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(float lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(float lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(float lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(float lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(float lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(float lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(float lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Subtract(float lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtract(float lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(float lhs, decimal rhs) => (decimal)lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, double rhs) => lhs - (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, float rhs) => lhs - (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Subtract(decimal lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Mod(byte lhs, byte rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Mod(byte lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Mod(byte lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(byte lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(byte lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(byte lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(byte lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Mod(byte lhs, char rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(byte lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(byte lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(byte lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Mod(short lhs, byte rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Mod(short lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(short lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(short lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(short lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(short lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(short lhs, ulong rhs) => lhs % (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Mod(short lhs, char rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(short lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(short lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(short lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Mod(ushort lhs, byte rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(ushort lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Mod(ushort lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(ushort lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(ushort lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(ushort lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ushort lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Mod(ushort lhs, char rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ushort lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(ushort lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(ushort lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(int lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(int lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(int lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(int lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(int lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(int lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(int lhs, ulong rhs) => lhs % (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(int lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(int lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(int lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(int lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(uint lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(uint lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(uint lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(uint lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(uint lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(uint lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(uint lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(uint lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(uint lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(uint lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(uint lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(long lhs, ulong rhs) => lhs % (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(long lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(long lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(long lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(long lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ulong lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ulong lhs, short rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ulong lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ulong lhs, int rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ulong lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ulong lhs, long rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ulong lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(ulong lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ulong lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(ulong lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(ulong lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Mod(char lhs, byte rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Mod(char lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Mod(char lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Mod(char lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod(char lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Mod(char lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mod(char lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Mod(char lhs, char rhs) => (char)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(char lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(char lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(char lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(double lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(double lhs, decimal rhs) => (decimal)lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(float lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(float lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(float lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(float lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(float lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(float lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(float lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(float lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Mod(float lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Mod(float lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(float lhs, decimal rhs) => (decimal)lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, double rhs) => lhs % (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, float rhs) => lhs % (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Mod(decimal lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Multiply(byte lhs, byte rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Multiply(byte lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Multiply(byte lhs, ushort rhs) => (ushort) (lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(byte lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(byte lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(byte lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(byte lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Multiply(byte lhs, char rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(byte lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(byte lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(byte lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Multiply(short lhs, byte rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Multiply(short lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(short lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(short lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(short lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(short lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(short lhs, ulong rhs) => lhs * (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Multiply(short lhs, char rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(short lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(short lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(short lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Multiply(ushort lhs, byte rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(ushort lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Multiply(ushort lhs, ushort rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(ushort lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(ushort lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(ushort lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ushort lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Multiply(ushort lhs, char rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ushort lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(ushort lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(ushort lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(int lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(int lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(int lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(int lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(int lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(int lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(int lhs, ulong rhs) => lhs * (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(int lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(int lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(int lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(int lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(uint lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(uint lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(uint lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(uint lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(uint lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(uint lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(uint lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(uint lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(uint lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(uint lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(uint lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(long lhs, ulong rhs) => lhs * (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(long lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(long lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(long lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(long lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ulong lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ulong lhs, short rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ulong lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ulong lhs, int rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ulong lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ulong lhs, long rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ulong lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(ulong lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ulong lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(ulong lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(ulong lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Multiply(char lhs, byte rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Multiply(char lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Multiply(char lhs, ushort rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Multiply(char lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Multiply(char lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Multiply(char lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Multiply(char lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Multiply(char lhs, char rhs) => (char)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(char lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(char lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(char lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(double lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(double lhs, decimal rhs) => (decimal)lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(float lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(float lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(float lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(float lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(float lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(float lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(float lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(float lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Multiply(float lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Multiply(float lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(float lhs, decimal rhs) => (decimal)lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, double rhs) => lhs * (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, float rhs) => lhs * (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Multiply(decimal lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Divide(byte lhs, byte rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Divide(byte lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Divide(byte lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(byte lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(byte lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(byte lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(byte lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Divide(byte lhs, char rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(byte lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(byte lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(byte lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Divide(short lhs, byte rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Divide(short lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(short lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(short lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(short lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(short lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(short lhs, ulong rhs) => (double)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Divide(short lhs, char rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(short lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(short lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(short lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Divide(ushort lhs, byte rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(ushort lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Divide(ushort lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(ushort lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(ushort lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(ushort lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ushort lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Divide(ushort lhs, char rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ushort lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(ushort lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(ushort lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(int lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(int lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(int lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(int lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(int lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(int lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(int lhs, ulong rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(int lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(int lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(int lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(int lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(uint lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(uint lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(uint lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(uint lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(uint lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(uint lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(uint lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(uint lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(uint lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(uint lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(uint lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(long lhs, ulong rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(long lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(long lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(long lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(long lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ulong lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ulong lhs, short rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ulong lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ulong lhs, int rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ulong lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ulong lhs, long rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ulong lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(ulong lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ulong lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(ulong lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(ulong lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Divide(char lhs, byte rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short Divide(char lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort Divide(char lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Divide(char lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Divide(char lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Divide(char lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Divide(char lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Divide(char lhs, char rhs) => (char)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(char lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(char lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(char lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(double lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(double lhs, decimal rhs) => (decimal)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(float lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(float lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(float lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(float lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(float lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(float lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(float lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(float lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Divide(float lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divide(float lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(float lhs, decimal rhs) => (decimal)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, double rhs) => lhs / (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, float rhs) => lhs / (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Divide(decimal lhs, decimal rhs) => lhs / rhs;
    }
}
