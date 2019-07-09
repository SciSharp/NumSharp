using System;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark.Unmanaged
{
    //RELEASE-OPTIMIZE:
    //|                      Method | RunStrategy | UnrollFactor |        Mean |       Error |      StdDev |      Median |         Min |         Max | Ratio | RatioSD |
    //|---------------------------- |------------ |------------- |------------:|------------:|------------:|------------:|------------:|------------:|------:|--------:|
    //|    '(double) (object) Tval' |   ColdStart |            1 |   125.96 us | 186.4836 us | 123.3473 us |    75.20 us |    75.10 us |   468.90 us |  0.08 |    0.03 |
    //| '(double) (ValueType) Tval' |   ColdStart |            1 |   114.82 us | 180.4753 us | 119.3732 us |    75.20 us |    75.10 us |   454.30 us |  0.07 |    0.02 |
    //| '(double) (IComparab) Tval' |   ColdStart |            1 |   115.18 us | 188.9671 us | 124.9900 us |    75.25 us |    75.10 us |   470.90 us |  0.07 |    0.03 |
    //|     'FastCast((TOut) @in;)' |   ColdStart |            1 | 1,491.73 us | 985.5014 us | 651.8481 us | 1,285.55 us | 1,163.00 us | 3,326.30 us |  1.00 |    0.00 |
    //|                             |             |              |             |             |             |             |             |             |       |         |
    //|    '(double) (object) Tval' |  Throughput |           16 |    75.44 us |   0.1344 us |   0.0703 us |    75.41 us |    75.38 us |    75.58 us |  0.07 |    0.00 |
    //| '(double) (ValueType) Tval' |  Throughput |           16 |    76.64 us |   1.5446 us |   1.0216 us |    76.31 us |    75.48 us |    78.39 us |  0.07 |    0.00 |
    //| '(double) (IComparab) Tval' |  Throughput |           16 |    75.89 us |   0.6555 us |   0.4335 us |    75.79 us |    75.48 us |    76.70 us |  0.07 |    0.00 |
    //|     'FastCast((TOut) @in;)' |  Throughput |           16 | 1,126.62 us |  20.0607 us |  11.9378 us | 1,125.58 us | 1,113.78 us | 1,150.95 us |  1.00 |    0.00 |

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
        private Holder<double> h = new Holder<double>(15.3d);


        [Benchmark(Description = "(double) (object) Tval")]
        public void BoxCast()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionB(h);
            }
        }

        [Benchmark(Description = "(double) (ValueType) Tval")]
        public void ValueTypeCast()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionC(h);
            }
        }

        [Benchmark(Description = "(double) (IComparable) Tval")]
        public void IComparableCast()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = FunctionD(h);
            }
        }


        [Benchmark(Baseline = true, Description = "FastCast<TOut>(Holder<T> @in;)")]
        public void FastCastFunction()
        {
            double ret;
            for (int i = 0; i < iterations; i++)
            {
                ret = Function(h);
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        private double Function<T>(Holder<T> tval)
        {
            return FastCast<double>(tval.Value);
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionB<T>(Holder<T> tval)
        {
            return (double)(object)tval.Value;
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionC<T>(Holder<T> tval) where T : struct
        {
            return (double)(ValueType)tval.Value;
        }

        [MethodImpl((MethodImplOptions)768)]
        private double FunctionD<T>(Holder<T> tval) where T : IComparable
        {
            return (double)(IComparable)tval.Value;
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
