using System;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Benchmark.Unmanaged
{
    //|              Method |     Mean |    Error |    StdDev |   Median |      Min |        Max | Ratio | RatioSD |
    //|-------------------- |---------:|---------:|----------:|---------:|---------:|-----------:|------:|--------:|
    //|       Action_Method | 897.2 us | 4.271 us |  57.96 us | 880.4 us | 869.5 us | 1,871.3 us |  7.02 |    0.65 |
    //| Action_Method_stack | 897.6 us | 3.880 us |  52.65 us | 882.9 us | 875.1 us | 2,137.8 us |  7.03 |    0.64 |
    //|          Expression | 636.3 us | 2.689 us |  36.49 us | 627.5 us | 625.1 us | 1,479.9 us |  4.98 |    0.43 |
    //|    Expression_stack | 645.7 us | 7.794 us | 105.78 us | 626.9 us | 595.4 us | 4,599.9 us |  5.05 |    0.89 |
    //|              Action | 886.5 us | 2.888 us |  39.19 us | 878.1 us | 875.1 us | 1,598.8 us |  6.94 |    0.56 |
    //|        Action_stack | 882.3 us | 1.731 us |  23.49 us | 877.1 us | 875.1 us | 1,265.3 us |  6.91 |    0.51 |
    //|        Method_stack | 760.5 us | 2.665 us |  36.16 us | 753.5 us | 750.6 us | 1,270.7 us |  5.95 |    0.49 |
    //|              Method | 905.8 us | 4.362 us |  59.19 us | 880.7 us | 875.1 us | 1,528.6 us |  7.09 |    0.68 |
    //|          Switchcase | 378.8 us | 1.252 us |  17.00 us | 375.4 us | 357.3 us |   841.9 us |  2.96 |    0.23 |
    //|             Inlined | 128.7 us | 1.048 us |  14.22 us | 125.2 us | 125.1 us |   389.6 us |  1.00 |    0.00 |

    [SimpleJob(RunStrategy.ColdStart, warmupCount: 5, targetCount: 2000)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class NestingCalls
    {
        private Func<int, int> action;
        private Func<int, int> action_method;
        private int a;
        Expression<Func<int, int>> expression;
        Func<int, int> expressionComp;

        [GlobalSetup]
        public void Setup()
        {
            expression = x => x + 1 - 1;
            expressionComp = expression.Compile();
            action = x => x + 1 - 1;
            action_method = method;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        int method(int a)
        {
            return a + 1 - 1;
        }

        [Benchmark]
        public void Action_Method()
        {
            for (int i = 0; i < 500_000; i++)
            {
                action_method(a);
            }
        }


        [Benchmark]
        public void Action_Method_stack()
        {
            var stackvar = action_method;
            for (int i = 0; i < 500_000; i++)
            {
                stackvar(a);
            }
        }

        [Benchmark]
        public void Expression()
        {
            for (int i = 0; i < 500_000; i++)
            {
                expressionComp(a);
            }
        }


        [Benchmark]
        public void Expression_stack()
        {
            var stackvar = expressionComp;
            for (int i = 0; i < 500_000; i++)
            {
                stackvar(a);
            }
        }


        [Benchmark]
        public void Action()
        {
            for (int i = 0; i < 500_000; i++)
            {
                action(a);
            }
        }


        [Benchmark]
        public void Action_stack()
        {
            var stackvar = action;
            for (int i = 0; i < 500_000; i++)
            {
                stackvar(a);
            }
        }

        [Benchmark]
        public void Method_stack()
        {
            Func<int, int> stackmethod = method;
            for (int i = 0; i < 500_000; i++)
            {
                stackmethod(a);
            }
        }

        [Benchmark]
        public void Method()
        {
            for (int i = 0; i < 500_000; i++)
            {
                method(a);
            }
        }

        private static NPTypeCode code = NPTypeCode.Complex;

        [Benchmark]
        public void Switchcase()
        {
            int b = 0;
            for (int i = 0; i < 500_000; i++)
            {
                switch (code)
                {
                    case NPTypeCode.Empty:
                        break;
                    case NPTypeCode.Boolean:
                        break;
                    case NPTypeCode.Char:
                        break;
                    case NPTypeCode.Byte:
                        break;
                    case NPTypeCode.Int16:
                        break;
                    case NPTypeCode.UInt16:
                        break;
                    case NPTypeCode.Int32:
                        break;
                    case NPTypeCode.UInt32:
                        break;
                    case NPTypeCode.Int64:
                        break;
                    case NPTypeCode.UInt64:
                        break;
                    case NPTypeCode.Single:
                        break;
                    case NPTypeCode.Double:
                        break;
                    case NPTypeCode.Decimal:
                        break;
                    case NPTypeCode.String:
                        break;
                    case NPTypeCode.Complex:
                        b = a + 1 - 1;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                var s = b;
            }
        }

        [Benchmark(Baseline = true)]
        public void Inlined()
        {
            int b = 0;
            for (int i = 0; i < 500_000; i++)
            {
                b = a + 1 - 1;
            }

            var s = b;
        }
    }
}
