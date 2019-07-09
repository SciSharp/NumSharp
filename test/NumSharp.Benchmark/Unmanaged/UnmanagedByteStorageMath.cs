using System.Linq;
using BenchmarkDotNet.Attributes;
using OOMath;

namespace NumSharp.Benchmark.Unmanaged
{
    //|        Method |         Mean |       Error |      StdDev |       Median |           Min |          Max |    Ratio | RatioSD |
    //|-------------- |-------------:|------------:|------------:|-------------:|--------------:|-------------:|---------:|--------:|
    //|        DArray |     3.891 us |   0.2508 us |   0.6992 us |     3.900 us |     2.5000 us |     6.000 us |     2.96 |    0.78 |
    //|   DArrayLarge | 2,942.060 us | 106.4932 us | 303.8311 us | 2,820.150 us | 2,616.8000 us | 3,954.100 us | 2,231.85 |  460.13 |
    //|   DirectLarge | 3,963.200 us |  30.9955 us |  88.4319 us | 3,944.650 us | 3,836.9000 us | 4,212.300 us | 3,015.53 |  592.78 |
    //|        Direct |     1.365 us |   0.0957 us |   0.2685 us |     1.300 us |     0.9000 us |     2.000 us |     1.00 |    0.00 |
    //|      NDArray_ |    12.508 us |   0.4499 us |   1.2764 us |    12.500 us |    10.5000 us |    16.500 us |     9.50 |    2.04 |
    //| NDArray_Large |   972.432 us |   5.4667 us |  14.9649 us |   967.100 us |   958.7500 us | 1,039.550 us |   743.08 |  134.96 |
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [SimpleJob(launchCount: 2, warmupCount: 10, targetCount: 50)]
    public class DArrayMath
    {
        private const int iterations = 10;
        UnmanagedByteStorage<int> a;
        UnmanagedByteStorage<int> b;
        UnmanagedByteStorage<int> a_large;
        UnmanagedByteStorage<int> b_large;
        NDArray nd1;
        NDArray nd2;
        NDArray nd1_large;
        NDArray nd2_large;
        int[] arr_a;
        int[] arr_b;
        int[] arr_a_large;
        int[] arr_b_large;

        [IterationSetup]
        public void Setup()
        {
            nd1 = (NDArray)Enumerable.Range(0, 25).ToArray();
            nd2 = (NDArray)Enumerable.Range(0, 25).ToArray();
            nd1_large = (NDArray)Enumerable.Range(0, 100000).ToArray();
            nd2_large = (NDArray)Enumerable.Range(0, 100000).ToArray();
            a = new UnmanagedByteStorage<int>(new Shape(5, 5), 0);
            b = new UnmanagedByteStorage<int>(Enumerable.Range(0, 25).ToArray(), new Shape(5, 5));
            a_large = new UnmanagedByteStorage<int>(new Shape(100000, 5), 0);
            b_large = new UnmanagedByteStorage<int>(new Shape(100000, 5), 0);
            arr_a = Enumerable.Range(0, 25).ToArray();
            arr_b = Enumerable.Range(0, 25).ToArray();
            arr_a_large = Enumerable.Range(0, count: 500000).ToArray();
            arr_b_large = Enumerable.Range(0, count: 500000).ToArray();

            var _ = UnmanagedByteStorage<int>.TypeCode;
        }

        [Benchmark]
        public void DArray()
        {
            UnmanagedByteStorage<int> c;
            //for (int i = 0; i < iterations; i++) {
            c = a + b;
            c = a + b;
            c = a + b;
            c = a + b;
            c = a + b;
            //}
        }

        [Benchmark]
        public void DArrayLarge()
        {
            UnmanagedByteStorage<int> c;
            //for (int i = 0; i < iterations; i++) {
            c = a_large + b_large;
            c = a_large + b_large;
            c = a_large + b_large;
            c = a_large + b_large;
            c = a_large + b_large;
            //}
        }

        [Benchmark]
        public void DirectLarge()
        {
            //for (int j = 0; j < iterations; j++) {
            var len = arr_a_large.Length;
            var ret = new int[len];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a_large[i] + arr_b_large[i];
            }

            len = arr_a_large.Length;
            ret = new int[len];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a_large[i] + arr_b_large[i];
            }

            len = arr_a_large.Length;
            ret = new int[len];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a_large[i] + arr_b_large[i];
            }

            len = arr_a_large.Length;
            ret = new int[len];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a_large[i] + arr_b_large[i];
            }

            len = arr_a_large.Length;
            ret = new int[len];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a_large[i] + arr_b_large[i];
            }

            // }
        }

        [Benchmark(Baseline = true)]
        public void Direct()
        {
            //for (int j = 0; j < iterations; j++) {
            var len = arr_a.Length;
            var ret = new int[25];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a[i] + arr_b[i];
            }

            len = arr_a.Length;
            ret = new int[25];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a[i] + arr_b[i];
            }

            len = arr_a.Length;
            ret = new int[25];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a[i] + arr_b[i];
            }

            len = arr_a.Length;
            ret = new int[25];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a[i] + arr_b[i];
            }

            len = arr_a.Length;
            ret = new int[25];
            for (int i = 0; i < len; i++)
            {
                ret[i] = arr_a[i] + arr_b[i];
            }

            //}
        }

        [Benchmark()]
        public void NDArray_()
        {
            NDArray ret;
            //for (int j = 0; j < iterations; j++) {
            ret = nd1 + nd2;
            ret = nd1 + nd2;
            ret = nd1 + nd2;
            ret = nd1 + nd2;
            ret = nd1 + nd2;
            //}
        }

        [Benchmark()]
        public void NDArray_Large()
        {
            NDArray ret;
            //for (int j = 0; j < iterations; j++) {
            ret = nd1_large + nd2_large;
            ret = nd1_large + nd2_large;
            ret = nd1_large + nd2_large;
            ret = nd1_large + nd2_large;
            ret = nd1_large + nd2_large;
            //}
        }
    }
}
