using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Benchmark
{
    [SimpleJob(RunStrategy.ColdStart, targetCount: 3)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 3)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class ArrayAssignmentUnspecifiedType
    {
        public object[] randomCoinedDouble;
        public object[] allFalseArray;
        protected Boolean[] _arrayBoolean;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            // first array
            randomCoinedDouble = new object[10000];

            for (int i = 0; i < randomCoinedDouble.Length; i++)
            {
                randomCoinedDouble[i] = rnd.NextDouble() > 0.5d ? 1d : 0d;
            }

            _arrayBoolean = new bool[10000];
            allFalseArray = new object[10000];
            for (int i = 0; i < allFalseArray.Length; i++)
            {
                allFalseArray[i] = false;
            }
        }


        [Benchmark(Baseline = true)]
        public void Convert()
        {
            for (int i = 0; i < 10000; i++)
            {
                var val = allFalseArray[i];
                _arrayBoolean[i] = System.Convert.ToBoolean(val);
            }
        }

        [Benchmark]
        public void AsConvert()
        {
            for (int i = 0; i < 10000; i++)
            {
                var val = allFalseArray[i];
                _arrayBoolean[i] = val as Boolean? ?? System.Convert.ToBoolean(val);
            }
        }

        [Benchmark]
        public void ConvertRandomValue()
        {
            for (int i = 0; i < 10000; i++)
            {
                var val = randomCoinedDouble[i];
                _arrayBoolean[i] = System.Convert.ToBoolean(val);
            }
        }

        [Benchmark]
        public void AsConvertRandomValue()
        {
            for (int i = 0; i < 10000; i++)
            {
                var val = randomCoinedDouble[i];
                _arrayBoolean[i] = val as Boolean? ?? System.Convert.ToBoolean(val);
            }
        }
    }
}
