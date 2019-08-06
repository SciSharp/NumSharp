using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities.Maths
{
    /// <summary>
    ///     A class that provides math operations that return the exact expected type according to numpy.
    /// </summary>
    [SuppressMessage("ReSharper", "UnusedMember.Local")]
    [SuppressMessage("ReSharper", "ArrangeTypeMemberModifiers")]
    class Operator
    {

#if _REGEN //We manually fixed types that naturally do not match with casting.
        %import "C:\Users\Eli-PC\Desktop\SciSharp\NumSharp\src\NumSharp.Core\bin\Debug\netstandard2.0\NumSharp.Core.dll"
        %import NumSharp.np as np
        %ops = ["+", "-", "%", "*", "/"]
        %names = ["Add", "Subtract", "Mod", "Multiply", "Divide"]
        
        %foreach ops,names%
        %foreach supported_numericals,supported_numericals_lowercase%
        %foreach supported_numericals,supported_numericals_lowercase%
        [MethodImpl(MethodImplOptions.AggressiveInlining)] #(np.find_common_type(#101.ToString(), #201.ToString())) #2(#101 lhs, #201 rhs) => lhs #1 rhs;
        %
        %
        %
#else

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Add(byte lhs, byte rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Add(byte lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Add(byte lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(byte lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(byte lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(byte lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(byte lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Add(byte lhs, char rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(byte lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(byte lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(byte lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Add(short lhs, byte rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Add(short lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(short lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(short lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(short lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(short lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(short lhs, ulong rhs) => lhs + (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Add(short lhs, char rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(short lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(short lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(short lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Add(ushort lhs, byte rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(ushort lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Add(ushort lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(ushort lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(ushort lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(ushort lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ushort lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Add(ushort lhs, char rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ushort lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(ushort lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(ushort lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(int lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(int lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(int lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(int lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(int lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(int lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(int lhs, ulong rhs) => lhs + (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(int lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(int lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(int lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(int lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(uint lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(uint lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(uint lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(uint lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(uint lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(uint lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(uint lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(uint lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(uint lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(uint lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(uint lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(long lhs, ulong rhs) => lhs + (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(long lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(long lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(long lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(long lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ulong lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ulong lhs, short rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ulong lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ulong lhs, int rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ulong lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ulong lhs, long rhs) => lhs + (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ulong lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(ulong lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ulong lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(ulong lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(ulong lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Add(char lhs, byte rhs) => (byte)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Add(char lhs, short rhs) => (short)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Add(char lhs, ushort rhs) => (ushort)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Add(char lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Add(char lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Add(char lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Add(char lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        char Add(char lhs, char rhs) => (char)(lhs + rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(char lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(char lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(char lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(double lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(double lhs, decimal rhs) => (decimal)lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(float lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(float lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(float lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(float lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(float lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(float lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(float lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(float lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Add(float lhs, double rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Add(float lhs, float rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(float lhs, decimal rhs) => (decimal)lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, byte rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, short rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, ushort rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, int rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, uint rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, long rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, ulong rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, char rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, double rhs) => lhs + (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, float rhs) => lhs + (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Add(decimal lhs, decimal rhs) => lhs + rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Subtract(byte lhs, byte rhs) => (byte)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Subtract(byte lhs, short rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Subtract(byte lhs, ushort rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(byte lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(byte lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(byte lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(byte lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Subtract(byte lhs, char rhs) => (byte)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(byte lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(byte lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(byte lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Subtract(short lhs, byte rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Subtract(short lhs, short rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(short lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(short lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(short lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(short lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(short lhs, ulong rhs) => lhs - (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Subtract(short lhs, char rhs) => (short)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(short lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(short lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(short lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Subtract(ushort lhs, byte rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(ushort lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Subtract(ushort lhs, ushort rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(ushort lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(ushort lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(ushort lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ushort lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Subtract(ushort lhs, char rhs) => (ushort)(lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ushort lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(ushort lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(ushort lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(int lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(int lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(int lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(int lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(int lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(int lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(int lhs, ulong rhs) => lhs - (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(int lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(int lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(int lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(int lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(uint lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(uint lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(uint lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(uint lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(uint lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(uint lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(uint lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(uint lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(uint lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(uint lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(uint lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(long lhs, ulong rhs) => lhs - (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(long lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(long lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(long lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(long lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ulong lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ulong lhs, short rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ulong lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ulong lhs, int rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ulong lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ulong lhs, long rhs) => lhs - (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ulong lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(ulong lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ulong lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(ulong lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(ulong lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Subtract(char lhs, byte rhs) => (byte) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Subtract(char lhs, short rhs) => (short) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Subtract(char lhs, ushort rhs) => (ushort) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Subtract(char lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Subtract(char lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Subtract(char lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Subtract(char lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        char Subtract(char lhs, char rhs) => (char) (lhs - rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(char lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(char lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(char lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(double lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(double lhs, decimal rhs) => (decimal)lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(float lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(float lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(float lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(float lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(float lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(float lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(float lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(float lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Subtract(float lhs, double rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Subtract(float lhs, float rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(float lhs, decimal rhs) => (decimal)lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, byte rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, short rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, ushort rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, int rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, uint rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, long rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, ulong rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, char rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, double rhs) => lhs - (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, float rhs) => lhs - (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Subtract(decimal lhs, decimal rhs) => lhs - rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Mod(byte lhs, byte rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Mod(byte lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Mod(byte lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(byte lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(byte lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(byte lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(byte lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Mod(byte lhs, char rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(byte lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(byte lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(byte lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Mod(short lhs, byte rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Mod(short lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(short lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(short lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(short lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(short lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(short lhs, ulong rhs) => lhs % (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Mod(short lhs, char rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(short lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(short lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(short lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Mod(ushort lhs, byte rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(ushort lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Mod(ushort lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(ushort lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(ushort lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(ushort lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ushort lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Mod(ushort lhs, char rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ushort lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(ushort lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(ushort lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(int lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(int lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(int lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(int lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(int lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(int lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(int lhs, ulong rhs) => lhs % (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(int lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(int lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(int lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(int lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(uint lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(uint lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(uint lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(uint lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(uint lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(uint lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(uint lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(uint lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(uint lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(uint lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(uint lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(long lhs, ulong rhs) => lhs % (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(long lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(long lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(long lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(long lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ulong lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ulong lhs, short rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ulong lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ulong lhs, int rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ulong lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ulong lhs, long rhs) => lhs % (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ulong lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(ulong lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ulong lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(ulong lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(ulong lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Mod(char lhs, byte rhs) => (byte)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Mod(char lhs, short rhs) => (short)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Mod(char lhs, ushort rhs) => (ushort)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Mod(char lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Mod(char lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Mod(char lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Mod(char lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        char Mod(char lhs, char rhs) => (char)(lhs % rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(char lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(char lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(char lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(double lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(double lhs, decimal rhs) => (decimal)lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(float lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(float lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(float lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(float lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(float lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(float lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(float lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(float lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Mod(float lhs, double rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Mod(float lhs, float rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(float lhs, decimal rhs) => (decimal)lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, byte rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, short rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, ushort rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, int rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, uint rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, long rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, ulong rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, char rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, double rhs) => lhs % (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, float rhs) => lhs % (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Mod(decimal lhs, decimal rhs) => lhs % rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Multiply(byte lhs, byte rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Multiply(byte lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Multiply(byte lhs, ushort rhs) => (ushort) (lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(byte lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(byte lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(byte lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(byte lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Multiply(byte lhs, char rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(byte lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(byte lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(byte lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Multiply(short lhs, byte rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Multiply(short lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(short lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(short lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(short lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(short lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(short lhs, ulong rhs) => lhs * (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Multiply(short lhs, char rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(short lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(short lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(short lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Multiply(ushort lhs, byte rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(ushort lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Multiply(ushort lhs, ushort rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(ushort lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(ushort lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(ushort lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ushort lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Multiply(ushort lhs, char rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ushort lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(ushort lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(ushort lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(int lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(int lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(int lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(int lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(int lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(int lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(int lhs, ulong rhs) => lhs * (int)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(int lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(int lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(int lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(int lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(uint lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(uint lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(uint lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(uint lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(uint lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(uint lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(uint lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(uint lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(uint lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(uint lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(uint lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(long lhs, ulong rhs) => lhs * (long)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(long lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(long lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(long lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(long lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ulong lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ulong lhs, short rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ulong lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ulong lhs, int rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ulong lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ulong lhs, long rhs) => lhs * (ulong)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ulong lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(ulong lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ulong lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(ulong lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(ulong lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Multiply(char lhs, byte rhs) => (byte)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Multiply(char lhs, short rhs) => (short)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Multiply(char lhs, ushort rhs) => (ushort)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Multiply(char lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Multiply(char lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Multiply(char lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Multiply(char lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        char Multiply(char lhs, char rhs) => (char)(lhs * rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(char lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(char lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(char lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(double lhs, decimal rhs) => (decimal)lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(float lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(float lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(float lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(float lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(float lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(float lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(float lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(float lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(float lhs, double rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Multiply(float lhs, float rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(float lhs, decimal rhs) => (decimal)lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, byte rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, short rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, ushort rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, int rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, uint rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, long rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, ulong rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, char rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, double rhs) => lhs * (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, float rhs) => lhs * (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Multiply(decimal lhs, decimal rhs) => lhs * rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Divide(byte lhs, byte rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Divide(byte lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Divide(byte lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(byte lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(byte lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(byte lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(byte lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Divide(byte lhs, char rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(byte lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(byte lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(byte lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Divide(short lhs, byte rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Divide(short lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(short lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(short lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(short lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(short lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(short lhs, ulong rhs) => (double)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Divide(short lhs, char rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(short lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(short lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(short lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Divide(ushort lhs, byte rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(ushort lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Divide(ushort lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(ushort lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(ushort lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(ushort lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ushort lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Divide(ushort lhs, char rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ushort lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(ushort lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(ushort lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(int lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(int lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(int lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(int lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(int lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(int lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(int lhs, ulong rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(int lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(int lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(int lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(int lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(uint lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(uint lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(uint lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(uint lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(uint lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(uint lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(uint lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(uint lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(uint lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(uint lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(uint lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(long lhs, ulong rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(long lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(long lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(long lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(long lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ulong lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ulong lhs, short rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ulong lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ulong lhs, int rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ulong lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ulong lhs, long rhs) => lhs / (double)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ulong lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(ulong lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ulong lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(ulong lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(ulong lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        byte Divide(char lhs, byte rhs) => (byte)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        short Divide(char lhs, short rhs) => (short)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ushort Divide(char lhs, ushort rhs) => (ushort)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        int Divide(char lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        uint Divide(char lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        long Divide(char lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        ulong Divide(char lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        char Divide(char lhs, char rhs) => (char)(lhs / rhs);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(char lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(char lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(char lhs, decimal rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(double lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(double lhs, decimal rhs) => (decimal)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(float lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(float lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(float lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(float lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(float lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(float lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(float lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(float lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Divide(float lhs, double rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        float Divide(float lhs, float rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(float lhs, decimal rhs) => (decimal)lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, byte rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, short rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, ushort rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, int rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, uint rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, long rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, ulong rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, char rhs) => lhs / rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, double rhs) => lhs / (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, float rhs) => lhs / (decimal)rhs;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        decimal Divide(decimal lhs, decimal rhs) => lhs / rhs;
#endif
    }
}
