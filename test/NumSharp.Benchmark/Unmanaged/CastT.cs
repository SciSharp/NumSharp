using System;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark.Unmanaged
{
    //RELEASE-OPTIMIZE:
    //|                            Method |        Job | RunStrategy | UnrollFactor |        Mean |      Error |     StdDev |      Median |         Min |         Max | Ratio | RatioSD |
    //|---------------------------------- |----------- |------------ |------------- |------------:|-----------:|-----------:|------------:|------------:|------------:|------:|--------:|
    //|         '(double) (object) value' | Job-ZQPUVT |   ColdStart |            1 |   161.11 us | 198.714 us | 131.437 us |    90.45 us |    83.50 us |   504.60 us |  1.00 |    0.00 |
    //|      '(double) (ValueType) value' | Job-ZQPUVT |   ColdStart |            1 |   212.42 us | 193.659 us | 128.093 us |   171.65 us |   166.80 us |   576.70 us |  1.57 |    0.54 |
    //|    '(double) (IComparable) value' | Job-ZQPUVT |   ColdStart |            1 |   216.02 us | 175.638 us | 116.174 us |   175.10 us |   166.80 us |   540.40 us |  1.60 |    0.52 |
    //| 'Unsafe.As<T, double>(ref value)' | Job-ZQPUVT |   ColdStart |            1 |   290.98 us | 414.401 us | 274.100 us |   180.15 us |   176.60 us | 1,054.40 us |  2.02 |    1.02 |
    //|  'FastCast<TOut>(Holder<T> @in;)' | Job-ZQPUVT |   ColdStart |            1 | 1,481.63 us | 922.324 us | 610.060 us | 1,245.35 us | 1,072.20 us | 3,098.20 us | 11.93 |    5.57 |
    //|                                   |            |             |              |             |            |            |             |             |             |       |         |
    //|         '(double) (object) value' | Job-LTSICV |  Throughput |           16 |   110.81 us |  28.387 us |  18.776 us |   104.37 us |    89.71 us |   140.98 us |  1.00 |    0.00 |
    //|      '(double) (ValueType) value' | Job-LTSICV |  Throughput |           16 |   103.44 us |  26.193 us |  17.325 us |    98.19 us |    86.03 us |   137.14 us |  0.94 |    0.12 |
    //|    '(double) (IComparable) value' | Job-LTSICV |  Throughput |           16 |   103.68 us |  25.812 us |  17.073 us |    96.57 us |    87.68 us |   139.90 us |  0.95 |    0.17 |
    //| 'Unsafe.As<T, double>(ref value)' | Job-LTSICV |  Throughput |           16 |    90.26 us |   4.679 us |   3.095 us |    89.34 us |    86.36 us |    95.58 us |  0.84 |    0.14 |
    //|  'FastCast<TOut>(Holder<T> @in;)' | Job-LTSICV |  Throughput |           16 | 1,715.69 us | 150.547 us |  89.588 us | 1,717.07 us | 1,577.88 us | 1,817.06 us | 16.20 |    2.70 |

    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class CastT
    {
        private const int iterations = 300_000;
        private double d = 15.3d;
        double a = 5;
        double b = 3;
        private double h = 15.3d;


        [Benchmark(Baseline = true, Description = "(double) (object) value")]
        public double BoxCast()
        {
            double ret = default;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionB(h);
            }

            return ret;
        }

        [Benchmark(Description = "(double) (ValueType) value")]
        public double ValueTypeCast()
        {
            double ret = default;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionC(h);
            }

            return ret;
        }

        [Benchmark(Description = "(double) (IComparable) value")]
        public double IComparableCast()
        {
            double ret = default;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionD(h);
            }

            return ret;
        }

        [Benchmark(Description = "Unsafe.As<T, double>(ref value)")]
        public double UnsafeAs()
        {
            double ret = default;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionE(h);
            }

            return ret;
        }


        [Benchmark(Description = "FastCast<TOut>(Holder<T> @in;)")]
        public double FastCastFunction()
        {
            double ret = default;
            for (int i = 0; i < iterations; i++)
            {
                ret = Function(h);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        private double Function<T>(T value)
        {
            return FastCast<double>(value);
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionB<T>(T value)
        {
            return (double)(object)value;
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionC<T>(T value) where T : struct
        {
            return (double)(ValueType)value;
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionD<T>(T value) where T : IComparable
        {
            return (double)(IComparable)value;
        }


        [MethodImpl((MethodImplOptions)768)]
        private double FunctionE<T>(T value) where T : IComparable
        {
            return Unsafe.As<T, double>(ref value);
        }


        [MethodImpl((MethodImplOptions)768)]
        private TOut FastCast<TOut>(object @in)
        {
            return (TOut)@in;
        }

        public struct Holder<T>
        {
            public T Value;

            /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
            public Holder(T value)
            {
                Value = value;
            }
        }
    }
}
