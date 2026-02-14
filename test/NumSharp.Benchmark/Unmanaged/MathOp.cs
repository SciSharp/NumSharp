using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark.Unmanaged
{
    //|                        Method | RunStrategy | UnrollFactor |        Mean |         Error |       StdDev |      Median |         Min |         Max | Ratio | RatioSD |
    //|------------------------------ |------------ |------------- |------------:|--------------:|-------------:|------------:|------------:|------------:|------:|--------:|
    //|                        Vector |   ColdStart |            1 |    764.1 us |    388.503 us |   256.971 us |    681.8 us |    661.6 us |  1,494.7 us |  1.33 |    0.15 |
    //|                     ScalarAdd |   ColdStart |            1 | 24,250.7 us |    545.400 us |   360.748 us | 24,099.2 us | 24,074.8 us | 25,201.6 us | 43.65 |    5.15 |
    //|                  MathComputer |   ColdStart |            1 |  1,923.5 us |  2,651.523 us | 1,753.818 us |  1,359.1 us |  1,351.5 us |  6,914.1 us |  3.12 |    1.75 |
    //|             ScalarAddSwitched |   ColdStart |            1 | 42,071.5 us |  1,442.499 us |   954.124 us | 41,771.3 us | 41,511.7 us | 44,690.4 us | 75.65 |    8.53 |
    //|     ScalarAddSwitchedTypeCode |   ColdStart |            1 | 34,097.8 us |  1,199.540 us |   793.422 us | 33,840.8 us | 33,619.7 us | 36,282.4 us | 61.31 |    6.93 |
    //| ScalarAddSwitchedTypeCodeFull |   ColdStart |            1 | 13,901.0 us |  1,555.486 us | 1,028.857 us | 13,476.9 us | 13,350.7 us | 16,689.1 us | 24.87 |    2.17 |
    //|                ScalarAddBoxed |   ColdStart |            1 | 23,568.5 us |  1,229.228 us |   813.058 us | 23,332.3 us | 23,177.9 us | 25,877.2 us | 42.32 |    4.44 |
    //|                        Direct |   ColdStart |            1 |    566.5 us |    154.867 us |   102.435 us |    529.9 us |    524.1 us |    853.6 us |  1.00 |    0.00 |
    //|                               |             |              |             |               |              |             |             |             |       |         |
    //|                        Vector |  Throughput |           16 |    910.8 us |      9.546 us |     5.681 us |    909.5 us |    904.2 us |    920.5 us |  1.71 |    0.02 |
    //|                     ScalarAdd |  Throughput |           16 | 24,295.1 us |     74.301 us |    38.861 us | 24,297.8 us | 24,246.7 us | 24,341.1 us | 45.57 |    0.44 |
    //|                  MathComputer |  Throughput |           16 |  1,448.9 us |     22.500 us |    14.883 us |  1,447.1 us |  1,433.4 us |  1,472.9 us |  2.71 |    0.02 |
    //|             ScalarAddSwitched |  Throughput |           16 | 48,124.5 us | 10,660.187 us | 7,051.054 us | 44,428.0 us | 42,639.8 us | 58,423.9 us | 90.11 |   12.64 |
    //|     ScalarAddSwitchedTypeCode |  Throughput |           16 | 32,102.8 us |    473.380 us |   313.111 us | 32,090.6 us | 31,744.4 us | 32,790.6 us | 60.16 |    0.70 |
    //| ScalarAddSwitchedTypeCodeFull |  Throughput |           16 | 13,782.7 us |    119.358 us |    71.028 us | 13,765.8 us | 13,696.0 us | 13,876.7 us | 25.84 |    0.18 |
    //|                ScalarAddBoxed |  Throughput |           16 | 24,268.1 us |    905.324 us |   598.816 us | 24,153.6 us | 23,333.3 us | 25,062.3 us | 45.47 |    0.97 |
    //|                        Direct |  Throughput |           16 |    533.7 us |      7.040 us |     4.657 us |    535.0 us |    526.3 us |    541.0 us |  1.00 |    0.00 |

    //RELEASE-OPTIMIZE:
    //|                        Method | RunStrategy | UnrollFactor |        Mean |        Error |        StdDev |      Median |         Min |          Max |  Ratio | RatioSD |
    //|------------------------------ |------------ |------------- |------------:|-------------:|--------------:|------------:|------------:|-------------:|-------:|--------:|
    //|                        Vector |   ColdStart |            1 |   161.74 us |   392.447 us |   259.5796 us |    75.20 us |    75.10 us |    899.70 us |   1.07 |    0.28 |
    //|                     ScalarAdd |   ColdStart |            1 |   118.70 us |   202.644 us |   134.0368 us |    75.20 us |    75.10 us |    500.10 us |   0.95 |    0.08 |
    //|                 MathComputer_ |   ColdStart |            1 |   366.91 us |   244.173 us |   161.5052 us |   305.25 us |   300.30 us |    815.30 us |   3.72 |    0.97 |
    //|             ScalarAddSwitched |   ColdStart |            1 | 3,732.36 us | 2,355.715 us | 1,558.1598 us | 3,231.10 us | 3,156.80 us |  8,161.00 us |  37.91 |    8.43 |
    //|     ScalarAddSwitchedTypeCode |   ColdStart |            1 | 1,969.92 us |   485.245 us |   320.9597 us | 1,874.25 us | 1,822.90 us |  2,881.40 us |  21.47 |    5.79 |
    //| ScalarAddSwitchedTypeCodeFull |   ColdStart |            1 | 4,820.32 us | 1,010.627 us |   668.4671 us | 4,644.75 us | 4,368.50 us |  6,677.60 us |  52.94 |   14.77 |
    //|                ScalarAddBoxed |   ColdStart |            1 | 8,551.20 us | 1,604.233 us | 1,061.1011 us | 8,206.40 us | 7,972.80 us | 11,516.00 us |  94.22 |   26.24 |
    //|                        Direct |   ColdStart |            1 |   122.29 us |   197.099 us |   130.3688 us |    78.35 us |    75.20 us |    492.70 us |   1.00 |    0.00 |
    //|                               |             |              |             |              |               |             |             |              |        |         |
    //|                        Vector |  Throughput |           16 |    78.41 us |     6.015 us |     3.5794 us |    76.24 us |    75.73 us |     86.06 us |   1.02 |    0.05 |
    //|                     ScalarAdd |  Throughput |           16 |    77.26 us |     2.029 us |     1.3422 us |    77.03 us |    75.82 us |     79.73 us |   1.01 |    0.02 |
    //|                 MathComputer_ |  Throughput |           16 |   304.39 us |     3.479 us |     2.0704 us |   303.53 us |   302.04 us |    308.87 us |   3.97 |    0.07 |
    //|             ScalarAddSwitched |  Throughput |           16 | 3,244.63 us |    61.622 us |    36.6700 us | 3,240.81 us | 3,209.54 us |  3,327.22 us |  42.32 |    0.84 |
    //|     ScalarAddSwitchedTypeCode |  Throughput |           16 | 1,917.91 us |    17.642 us |    11.6690 us | 1,912.49 us | 1,905.10 us |  1,940.88 us |  25.03 |    0.32 |
    //| ScalarAddSwitchedTypeCodeFull |  Throughput |           16 | 4,489.44 us |    52.478 us |    34.7112 us | 4,490.34 us | 4,433.23 us |  4,541.00 us |  58.59 |    0.61 |
    //|                ScalarAddBoxed |  Throughput |           16 | 7,772.17 us |    78.140 us |    51.6847 us | 7,744.84 us | 7,727.88 us |  7,869.93 us | 101.44 |    1.53 |
    //|                        Direct |  Throughput |           16 |    76.63 us |     1.483 us |     0.9810 us |    76.31 us |    75.48 us |     78.35 us |   1.00 |    0.00 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class MathOp
    {
        private const int iterations = 300_000;
        private double d = 15.3d;
        private unsafe double* dptr = (double*)Marshal.AllocHGlobal(sizeof(double));
        double a = 5;
        double b = 3;

        [Benchmark]
        public void Vector()
        {
            Vector<double> ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = new Vector<double>(d) + new Vector<double>(d);
            }
        }

        //This used to be a local copy of Vector<T> but it turned out to be so slow - probably because Vector<T> is optimized behind the scenes.
        //[Benchmark]
        //public unsafe void NewVector() {
        //    NewVector<double> ret;
        //    for (int i = 0; i < iterations; i++) {
        //        ret = new NewVector<double>(dptr) + new NewVector<double>(dptr);
        //    }
        //}

        [Benchmark]
        public void ScalarAdd()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = ScalarAdd<double>(a, b);
            }
        }

        [Benchmark]
        public void MathComputer_()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = new MathComputer<double>(a) + new MathComputer<double>(b);
            }
        }

        [Benchmark]
        public void ScalarAddSwitched()
        {
            for (int i = 0; i < iterations; i++)
            {
                var doubletype = a.GetType();
#if _REGEN
                %foreach supported_numericals_lowercase%
                if (doubletype == typeof(#1)) {
                    var ret = ScalarAdd<#1>(0,0);
                } else 
                %
#else
                if (doubletype == typeof(byte))
                {
                    var ret = ScalarAdd<byte>(0, 0);
                }
                else if (doubletype == typeof(short))
                {
                    var ret = ScalarAdd<short>(0, 0);
                }
                else if (doubletype == typeof(ushort))
                {
                    var ret = ScalarAdd<ushort>(0, 0);
                }
                else if (doubletype == typeof(int))
                {
                    var ret = ScalarAdd<int>(0, 0);
                }
                else if (doubletype == typeof(uint))
                {
                    var ret = ScalarAdd<uint>(0, 0);
                }
                else if (doubletype == typeof(long))
                {
                    var ret = ScalarAdd<long>(0, 0);
                }
                else if (doubletype == typeof(ulong))
                {
                    var ret = ScalarAdd<ulong>(0, 0);
                }
                else if (doubletype == typeof(char))
                {
                    var ret = ScalarAdd<char>('\0', '\0');
                }
                else if (doubletype == typeof(float))
                {
                    var ret = ScalarAdd<float>(0, 0);
                }
                else if (doubletype == typeof(decimal))
                {
                    var ret = ScalarAdd<decimal>(0, 0);
                }
                else
#endif
                if (doubletype == typeof(double))
                {
                    var ret = ScalarAdd<double>(a, b);
                }
            }
        }

        [Benchmark]
        public void ScalarAddSwitchedTypeCode()
        {
            IComparable ret;
            for (int i = 0; i < iterations; i++)
            {
                var doubletype = Type.GetTypeCode(a.GetType());
                switch (doubletype)
                {
#if _REGEN
                %foreach supported_numericals%
	                case TypeCode.#1: {
                        ret = ScalarAdd<#1>(0, 0);
                        break;
                    }
                %
#else

                    case TypeCode.Byte:
                    {
                        ret = ScalarAdd<Byte>(0, 0);
                        break;
                    }

                    case TypeCode.Int16:
                    {
                        ret = ScalarAdd<Int16>(0, 0);
                        break;
                    }

                    case TypeCode.UInt16:
                    {
                        ret = ScalarAdd<UInt16>(0, 0);
                        break;
                    }

                    case TypeCode.Int32:
                    {
                        ret = ScalarAdd<Int32>(0, 0);
                        break;
                    }

                    case TypeCode.UInt32:
                    {
                        ret = ScalarAdd<UInt32>(0, 0);
                        break;
                    }

                    case TypeCode.Int64:
                    {
                        ret = ScalarAdd<Int64>(0, 0);
                        break;
                    }

                    case TypeCode.UInt64:
                    {
                        ret = ScalarAdd<UInt64>(0, 0);
                        break;
                    }

                    case TypeCode.Char:
                    {
                        ret = ScalarAdd<Char>('\0', '\0');
                        break;
                    }

                    case TypeCode.Single:
                    {
                        ret = ScalarAdd<Single>(0, 0);
                        break;
                    }

                    case TypeCode.Decimal:
                    {
                        ret = ScalarAdd<Decimal>(0, 0);
                        break;
                    }

                    case TypeCode.Double:
                    {
                        ret = ScalarAdd<Double>(a, b);
                        break;
                    }
#endif

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        [Benchmark]
        public void ScalarAddSwitchedTypeCodeFull()
        {
            IComparable ret;
            for (int i = 0; i < iterations; i++)
            {
                var doubletype = Type.GetTypeCode(a.GetType());
                switch (doubletype)
                {
#if _REGEN
                %foreach supported_numericals%
	                case TypeCode.#1: {
                       return ScalarAdd<#1>(0, 0);
                        }
                %
#else

                    case TypeCode.Byte:
                    {
                        ret = ScalarAddSwitch<Byte>(0, 0);
                        break;
                    }

                    case TypeCode.Int16:
                    {
                        ret = ScalarAddSwitch<Int16>(0, 0);
                        break;
                    }

                    case TypeCode.UInt16:
                    {
                        ret = ScalarAddSwitch<UInt16>(0, 0);
                        break;
                    }

                    case TypeCode.Int32:
                    {
                        ret = ScalarAddSwitch<Int32>(0, 0);
                        break;
                    }

                    case TypeCode.UInt32:
                    {
                        ret = ScalarAddSwitch<UInt32>(0, 0);
                        break;
                    }

                    case TypeCode.Int64:
                    {
                        ret = ScalarAddSwitch<Int64>(0, 0);
                        break;
                    }

                    case TypeCode.UInt64:
                    {
                        ret = ScalarAddSwitch<UInt64>(0, 0);
                        break;
                    }

                    case TypeCode.Char:
                    {
                        ret = ScalarAddSwitch<Char>('\0', '\0');
                        break;
                    }

                    case TypeCode.Single:
                    {
                        ret = ScalarAddSwitch<Single>(0, 0);
                        break;
                    }

                    case TypeCode.Decimal:
                    {
                        ret = ScalarAddSwitch<Decimal>(0, 0);
                        break;
                    }

                    case TypeCode.Double:
                    {
                        ret = ScalarAddSwitch<Double>(a, b);
                        break;
                    }
#endif

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        [Benchmark]
        public void ScalarAddBoxed()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = (double)ScalarAddBoxed(a, b);
            }
        }

        [Benchmark(Baseline = true)]
        public void Direct()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = a + b;
            }
        }

        /// <summary>
        ///     this was stolen from Vector{T}
        ///     This is fast because typeof vs typeof is evaluation during compilation time or as soon as JIT optimizes it.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static T ScalarAdd<T>(T left, T right) where T : IComparable
        {
            if (typeof(T) == typeof(byte))
                return (T)(IComparable)(byte)((uint)(byte)(IComparable)left + (uint)(byte)(IComparable)right);
            if (typeof(T) == typeof(sbyte))
                return (T)(IComparable)(sbyte)((int)(sbyte)(IComparable)left + (int)(sbyte)(IComparable)right);
            if (typeof(T) == typeof(ushort))
                return (T)(IComparable)(ushort)((uint)(ushort)(IComparable)left + (uint)(ushort)(IComparable)right);
            if (typeof(T) == typeof(short))
                return (T)(IComparable)(short)((int)(short)(IComparable)left + (int)(short)(IComparable)right);
            if (typeof(T) == typeof(uint))
                return (T)(IComparable)(uint)((int)(uint)(IComparable)left + (int)(uint)(IComparable)right);
            if (typeof(T) == typeof(int))
                return (T)(IComparable)((int)(IComparable)left + (int)(IComparable)right);
            if (typeof(T) == typeof(ulong))
                return (T)(IComparable)(ulong)((long)(ulong)(IComparable)left + (long)(ulong)(IComparable)right);
            if (typeof(T) == typeof(long))
                return (T)(IComparable)((long)(IComparable)left + (long)(IComparable)right);
            if (typeof(T) == typeof(float))
                return (T)(IComparable)((float)(IComparable)left + (float)(IComparable)right);
            if (typeof(T) == typeof(double))
                return (T)(IComparable)((double)(IComparable)left + (double)(IComparable)right);
            return default;
        }

#if _REEGEN
                %foreach supported_numericals%
	                case TypeCode.#1: {
                        return 
                    }
                %
#else

#endif
        /// <summary>
        ///     this was stolen from Vector{T}
        ///     This is fast because typeof vs typeof is evaluation during compilation time or as soon as JIT optimizes it.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static T ScalarAddSwitch<T>(T left, T right) where T : IComparable
        {
            //if (typeof(T) == typeof(byte))
            //    return (T) (IComparable) (byte) ((uint) (byte) (IComparable) left + (uint) (byte) (IComparable) right);
            //if (typeof(T) == typeof(sbyte))
            //    return (T) (IComparable) (sbyte) ((int) (sbyte) (IComparable) left + (int) (sbyte) (IComparable) right);
            //if (typeof(T) == typeof(ushort))
            //    return (T) (IComparable) (ushort) ((uint) (ushort) (IComparable) left + (uint) (ushort) (IComparable) right);
            //if (typeof(T) == typeof(short))
            //    return (T) (IComparable) (short) ((int) (short) (IComparable) left + (int) (short) (IComparable) right);
            //if (typeof(T) == typeof(uint))
            //    return (T) (IComparable) (uint) ((int) (uint) (IComparable) left + (int) (uint) (IComparable) right);
            //if (typeof(T) == typeof(int))
            //    return (T) (IComparable) ((int) (IComparable) left + (int) (IComparable) right);
            //if (typeof(T) == typeof(ulong))
            //    return (T) (IComparable) (ulong) ((long) (ulong) (IComparable) left + (long) (ulong) (IComparable) right);
            //if (typeof(T) == typeof(long))
            //    return (T) (IComparable) ((long) (IComparable) left + (long) (IComparable) right);
            //if (typeof(T) == typeof(float))
            //    return (T) (IComparable) ((float) (IComparable) left + (float) (IComparable) right);
            //if (typeof(T) == typeof(double))
            //    return (T)(IComparable)((double)(IComparable)left + (double)(IComparable)right);
            switch (Type.GetTypeCode(typeof(T)))
            {
                case TypeCode.Byte:
                {
                    return (T)(IComparable)(byte)((uint)(byte)(IComparable)left + (uint)(byte)(IComparable)right);
                }

                case TypeCode.Int16:
                {
                    return (T)(IComparable)(short)((int)(short)(IComparable)left + (int)(short)(IComparable)right);
                }

                case TypeCode.UInt16:
                {
                    return (T)(IComparable)(ushort)((uint)(ushort)(IComparable)left + (uint)(ushort)(IComparable)right);
                }

                case TypeCode.Int32:
                {
                    return (T)(IComparable)((int)(IComparable)left + (int)(IComparable)right);
                }

                case TypeCode.UInt32:
                {
                    return (T)(IComparable)(uint)((int)(uint)(IComparable)left + (int)(uint)(IComparable)right);
                }

                case TypeCode.Int64:
                {
                    return (T)(IComparable)((long)(IComparable)left + (long)(IComparable)right);
                }

                case TypeCode.UInt64:
                {
                    return (T)(IComparable)(ulong)((long)(ulong)(IComparable)left + (long)(ulong)(IComparable)right);
                }

                case TypeCode.Char:
                {
                    return default;
                }

                case TypeCode.Single:
                {
                    return (T)(IComparable)((float)(IComparable)left + (float)(IComparable)right);
                }

                case TypeCode.Double:
                {
                    return (T)(IComparable)((double)(IComparable)left + (double)(IComparable)right);
                }

                case TypeCode.Decimal:
                {
                    return default;
                }
            }

            throw new NotSupportedException();
        }

        /// <summary>
        ///     this was stolen from Vector{T}
        ///     This is fast because T vs T is evaluation during compilation time or as soon as JIT optimizes it.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static object ScalarAddBoxed(object left, object right)
        {
            var T = left.GetType();
            if (T == typeof(byte))
                return (IComparable)(byte)((uint)(byte)(IComparable)left + (uint)(byte)(IComparable)right);
            if (T == typeof(sbyte))
                return (IComparable)(sbyte)((int)(sbyte)(IComparable)left + (int)(sbyte)(IComparable)right);
            if (T == typeof(ushort))
                return (IComparable)(ushort)((uint)(ushort)(IComparable)left + (uint)(ushort)(IComparable)right);
            if (T == typeof(short))
                return (IComparable)(short)((int)(short)(IComparable)left + (int)(short)(IComparable)right);
            if (T == typeof(uint))
                return (IComparable)(uint)((int)(uint)(IComparable)left + (int)(uint)(IComparable)right);
            if (T == typeof(int))
                return (IComparable)((int)(IComparable)left + (int)(IComparable)right);
            if (T == typeof(ulong))
                return (IComparable)(ulong)((long)(ulong)(IComparable)left + (long)(ulong)(IComparable)right);
            if (T == typeof(long))
                return (IComparable)((long)(IComparable)left + (long)(IComparable)right);
            if (T == typeof(float))
                return (IComparable)((float)(IComparable)left + (float)(IComparable)right);
            if (T == typeof(double))
                return (IComparable)((double)(IComparable)left + (double)(IComparable)right);
            return default;
        }

        public struct MathComputer<T> where T : struct
        {
            public T Value;

            [MethodImpl(OptimizeAndInline)]
            public MathComputer(T val)
            {
                Value = val;
            }

            [MethodImpl(OptimizeAndInline)]
            public static T operator +(MathComputer<T> lhs, T rhs)
            {
                return lhs + new MathComputer<T>(rhs);
            }

            [MethodImpl(OptimizeAndInline)]
            public static T operator +(MathComputer<T> left, MathComputer<T> right)
            {
                if (typeof(T) == typeof(byte))
                    return ((T)(ValueType)(byte)((uint)(byte)(ValueType)left.Value + (uint)(byte)(ValueType)right.Value));
                if (typeof(T) == typeof(sbyte))
                    return ((T)(ValueType)(sbyte)((int)(sbyte)(ValueType)left.Value + (int)(sbyte)(ValueType)right.Value));
                if (typeof(T) == typeof(ushort))
                    return ((T)(ValueType)(ushort)((uint)(ushort)(ValueType)left.Value + (uint)(ushort)(ValueType)right.Value));
                if (typeof(T) == typeof(short))
                    return ((T)(ValueType)(short)((int)(short)(ValueType)left.Value + (int)(short)(ValueType)right.Value));
                if (typeof(T) == typeof(uint))
                    return ((T)(ValueType)(uint)((int)(uint)(ValueType)left.Value + (int)(uint)(ValueType)right.Value));
                if (typeof(T) == typeof(int))
                    return ((T)(ValueType)((int)(ValueType)left.Value + (int)(ValueType)right.Value));
                if (typeof(T) == typeof(ulong))
                    return ((T)(ValueType)(ulong)((long)(ulong)(ValueType)left.Value + (long)(ulong)(ValueType)right.Value));
                if (typeof(T) == typeof(long))
                    return ((T)(ValueType)((long)(ValueType)left.Value + (long)(ValueType)right.Value));
                if (typeof(T) == typeof(float))
                    return ((T)(ValueType)((float)(ValueType)left.Value + (float)(ValueType)right.Value));
                if (typeof(T) == typeof(double))
                    return ((T)(ValueType)((double)(ValueType)left.Value + (double)(ValueType)right.Value));
                return default;
            }
        }
    }
}
